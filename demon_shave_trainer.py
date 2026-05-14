import torch
import torch.nn as nn
import json
import os
import copy

try:
    import torch_pruning as tp
except ImportError:
    print("Error: 'torch-pruning' library is required for Demon Shaving.")
    print("Please install it using: pip install torch-pruning")

class TrainingDemonShaver:
    """
    A version of the Demon Shaver designed to integrate directly into a training loop.
    It monitors activations during an epoch and can physically prune 'dead' neurons.
    """
    def __init__(self, model, example_input=None, device='cpu', threshold=1e-6):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.hooks = []
        self.stats = {}
        self.example_input = example_input if example_input is not None else torch.randn(1, 3, 192, 192).to(device)
        
        # We only target ReLU and similar activations as per original finder
        self.target_types = (nn.ReLU, nn.ReLU6, nn.Hardswish)
        self._register_hooks()

    def _register_hooks(self):
        """Registers hooks on all ReLU and Hardswish activation layers."""
        self.remove_hooks()
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_types):
                self.stats[name] = None
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            # output shape: [batch, channels, height, width] or [batch, channels]
            # active: True if non-zero anywhere in the batch/spatial dims
            if len(output.shape) == 4: # Conv2d output
                active = (output.abs() > self.threshold).any(dim=0).any(dim=1).any(dim=1)
            elif len(output.shape) == 2: # Linear output
                active = (output.abs() > self.threshold).any(dim=0)
            else:
                return

            if self.stats[name] is None:
                self.stats[name] = active.float()
            else:
                # Accumulate activity (Logical OR)
                self.stats[name] += active.float()
        return hook

    def reset_stats(self):
        """Reset the activation counters at the start of an epoch."""
        for name in self.stats:
            self.stats[name] = None

    def remove_hooks(self):
        """Clean up hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def find_dead_neurons(self):
        """Returns a dictionary of {layer_name: [dead_indices]}."""
        dead_neurons = {}
        for name, activity in self.stats.items():
            if activity is not None:
                # activity == 0 means it never woke up during the whole epoch
                dead_indices = torch.where(activity == 0)[0].tolist()
                if len(dead_indices) > 0:
                    dead_neurons[name] = dead_indices
        return dead_neurons

    def shave(self, verbose=True):
        """
        Physically prunes the dead neurons identified in the current epoch.
        Returns the pruned model and the list of pruned neurons.
        """
        dead_neurons = self.find_dead_neurons()
        if not dead_neurons:
            if verbose: print("No dead neurons (Demons) found this epoch. Shaving skipped.")
            return self.model, {}

        if verbose:
            total_dead = sum(len(indices) for indices in dead_neurons.values())
            print(f"🔪 👺 Demon Shaver: Found {total_dead} dead neurons across {len(dead_neurons)} layers.")

        # Build dependency graph
        self.model.eval()
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=self.example_input)

        pruning_plan = []
        for activation_name, dead_indices in dead_neurons.items():
            target_module = None
            for name, module in self.model.named_modules():
                if name == activation_name:
                    target_module = module
                    break
            
            if target_module is None: continue

            try:
                group = DG.get_pruning_group(target_module, tp.prune_conv_out_channels, idxs=dead_indices)
                if DG.check_pruning_group(group):
                    pruning_plan.append(group)
            except Exception as e:
                if verbose: print(f"Warning: Could not prune {activation_name}: {e}")

        # Execute pruning
        for group in pruning_plan:
            group.exec()

        if verbose:
            print(f"Success! Model physically shaved.")
            print(f"New parameters count: {sum(p.numel() for p in self.model.parameters()):,}")

        # After pruning, we MUST re-register hooks because some modules might have been replaced/altered
        self._register_hooks()
        
        return self.model, dead_neurons

    def export_to_onnx(self, output_path, verbose=True):
        """
        Exports the current (potentially shaved) model to ONNX format.
        This is the format usually required for high-speed inference.
        """
        self.model.eval()
        dummy_input = self.example_input.to(self.device)
        
        if verbose:
            print(f"📦 Exporting model to ONNX: {output_path}")
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        if verbose:
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Success! ONNX model saved. Size: {file_size:.2f} MB")

import torch
import torch.nn as nn
import timm
import json
import os

try:
    import torch_pruning as tp
except ImportError:
    print("Error: 'torch-pruning' library is required for physical shaving.")
    print("Please install it using: pip install torch-pruning")
    exit()

class DemonPhysicalShaver:
    def __init__(self, model_name, num_classes, checkpoint_path=None):
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()

    def physical_shave(self, dead_neurons_json, output_path):
        """
        Actually removes neurons from the model architecture, reducing latency.
        """
        with open(dead_neurons_json, 'r') as f:
            dead_data = json.load(f)

        print(f"--- Physical Shaving (Pruning) ---")
        
        # 1. Build a dependency graph
        # This is the 'Monster' part: it automatically finds which layers need to be
        # resized together to keep the model functional.
        example_inputs = torch.randn(1, 3, 192, 192)
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=example_inputs)

        # 2. Plan the pruning
        pruning_plan = []
        for layer_name, dead_indices in dead_data.items():
            # Find the layer
            layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer = module
                    break
            
            if layer is None:
                # If we targeted an activation, we look for the preceding conv/linear
                # TorchPruning usually works on Conv/Linear/BN
                print(f"Skipping activation-only layer: {layer_name}")
                continue

            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                # Create a pruning group (handles all dependencies)
                group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=dead_indices)
                if DG.check_pruning_group(group):
                    pruning_plan.append(group)
                    print(f"Queued pruning of {len(dead_indices)} channels in {layer_name}")

        # 3. Execute the shaving
        for group in pruning_plan:
            group.exec()

        # 4. Cleanup and Save
        # The model is now physically smaller!
        print("\nSuccess! The model architecture has been physically altered.")
        print(f"New parameters count: {sum(p.numel() for p in self.model.parameters()):,}")
        
        torch.save(self.model, output_path) # Save the whole model object (architecture + weights)
        print(f"Physical model saved to {output_path}")
        print("Note: Use torch.load() to load this model, as its architecture has changed.")

if __name__ == "__main__":
    # Settings
    MODEL_NAME = 'mobilenetv3_small_100'
    NUM_CLASSES = 6
    JSON_PATH = 'dead_neurons.json'
    OUTPUT_PATH = 'model_shaved_physical.pt' # .pt for whole model

    if os.path.exists(JSON_PATH):
        shaver = DemonPhysicalShaver(MODEL_NAME, NUM_CLASSES)
        shaver.physical_shave(JSON_PATH, OUTPUT_PATH)
    else:
        print(f"Target list not found: {JSON_PATH}")
        print("Please run 'demon_shave_target_finder.py' first.")

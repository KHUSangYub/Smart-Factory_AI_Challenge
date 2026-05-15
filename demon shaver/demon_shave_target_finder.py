import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from tqdm import tqdm
import os

class DemonShaver:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.stats = {}
        self._register_hooks()

    def _register_hooks(self):
        """Registers hooks on all ReLU and Hardswish activation layers."""
        for name, module in self.model.named_modules():
            # Check for common activation layers
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish)):
                self.stats[name] = None
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            # output shape: [batch, channels, height, width]
            # We want to check if a channel is ALWAYS zero across the batch and spatial dims.
            # active_mask: [channels] -> True if non-zero anywhere in batch/spatial
            # We use a threshold (e.g., 1e-6) to avoid noise.
            active = (output.abs() > 1e-6).any(dim=0).any(dim=1).any(dim=1)
            
            if self.stats[name] is None:
                self.stats[name] = active.float()
            else:
                # Accumulate activity (logical OR behavior using float sum)
                self.stats[name] += active.float()
        return hook

    def analyze(self, loader, num_batches=10, device='cpu'):
        self.model.to(device)
        self.model.eval()
        
        # Reset counters
        for name in self.stats:
            self.stats[name] = None
            
        count = 0
        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(loader, desc="Shaving Neurons")):
                if i >= num_batches:
                    break
                self.model(images.to(device))
                count += 1
        
        # Calculate results
        results = {}
        for name, activity in self.stats.items():
            if activity is not None:
                # activity / count = % of batches where this neuron was alive at least once
                # if 0.0, it never woke up (Dead)
                dead_indices = torch.where(activity == 0)[0].tolist()
                total_neurons = activity.size(0)
                results[name] = {
                    'dead_count': len(dead_indices),
                    'total': total_neurons,
                    'dead_indices': dead_indices,
                    'dead_percent': (len(dead_indices) / total_neurons) * 100
                }
        return results

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

if __name__ == "__main__":
    # Example usage
    # Change 'mobilenetv3_small_100' to the model your teammates are using
    MODEL_NAME = 'mobilenetv3_small_100'
    NUM_CLASSES = 6
    DATA_PATH = './competition_dataset/NEU-DET_open/train/images' # Adjust as needed
    
    print(f"--- Demon Shaving Analysis: {MODEL_NAME} ---")
    
    # 1. Load Model
    try:
        model = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    except:
        print("Model loading failed. Make sure 'timm' is installed.")
        exit()

    # 2. Setup Data
    transform = T.Compose([
        T.Resize((192, 192)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if os.path.exists(DATA_PATH):
        dataset = ImageFolder(DATA_PATH, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 3. Analyze
        shaver = DemonShaver(model)
        results = shaver.analyze(loader, num_batches=5)
        
        # 4. Report and Export
        import json
        total_dead = 0
        total_neurons = 0
        print("\n--- RESULTS ---")
        export_data = {}
        for layer, res in results.items():
            if res['dead_count'] > 0:
                print(f"Layer: {layer:40s} | Dead: {res['dead_count']:3d}/{res['total']:3d} ({res['dead_percent']:.1f}%)")
                total_dead += res['dead_count']
                export_data[layer] = res['dead_indices']
            total_neurons += res['total']
        
        # Save to JSON for demon_shaver.py
        with open('dead_neurons.json', 'w') as f:
            json.dump(export_data, f)
        
        print("-" * 30)
        print(f"Total Dead Neurons: {total_dead} / {total_neurons} ({ (total_dead/total_neurons)*100 if total_neurons > 0 else 0:.2f}%)")
        print(f"Target list saved to 'dead_neurons.json'")
        
        shaver.remove_hooks()
    else:
        print(f"Data path not found: {DATA_PATH}")
        print("Please check the DATA_PATH variable in the script.")

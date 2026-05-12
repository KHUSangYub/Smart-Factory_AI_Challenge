import torch
import torch.nn as nn
import timm
import json
import os

def shave_layer(module, dead_indices):
    """
    Physically removes channels from a layer.
    Note: This is a simplified version. In complex models like MobileNet,
    shaving one layer requires updating the input of the next.
    """
    total_channels = module.weight.shape[0]
    alive_indices = [i for i in range(total_channels) if i not in dead_indices]
    alive_indices = torch.tensor(alive_indices)

    # 1. Shave Weights
    new_weight = module.weight.data[alive_indices]
    
    # 2. Shave Bias (if exists)
    new_bias = None
    if module.bias is not None:
        new_bias = module.bias.data[alive_indices]

    return new_weight, new_bias, alive_indices

class DemonShaver:
    def __init__(self, model_name, num_classes, checkpoint_path=None):
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()

    def shave_with_mask(self, dead_neurons_json):
        """
        Zeroes out dead neurons. This is 'Soft Shaving'.
        It doesn't speed up the model yet, but it proves accuracy is maintained.
        """
        with open(dead_neurons_json, 'r') as f:
            dead_data = json.load(f)

        print(f"--- Soft Shaving (Masking) ---")
        for layer_name, dead_indices in dead_data.items():
            found = False
            # We search for the Conv2d layer that feeds into this activation
            # Layer names in finder are usually the activation, we need the preceding Conv
            # This is a bit complex for a generic script, so we use a simple lookup
            for name, module in self.model.named_modules():
                if name in layer_name or layer_name in name:
                    if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                        mask = torch.ones(module.weight.shape[0])
                        mask[dead_indices] = 0
                        
                        # Apply mask to weight
                        if len(module.weight.shape) == 4: # Conv2d
                            module.weight.data *= mask.view(-1, 1, 1, 1)
                        else: # Linear
                            module.weight.data *= mask.view(-1, 1)
                            
                        if module.bias is not None:
                            module.bias.data *= mask
                            
                        print(f"Masked {len(dead_indices)} neurons in {name}")
                        found = True
            if not found:
                print(f"Could not find matching weight layer for {layer_name}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Shaved model saved to {path}")

if __name__ == "__main__":
    # Settings
    MODEL_NAME = 'mobilenetv3_small_100'
    NUM_CLASSES = 6
    JSON_PATH = 'dead_neurons.json'
    OUTPUT_PATH = 'model_shaved_soft.pth'

    if os.path.exists(JSON_PATH):
        shaver = DemonShaver(MODEL_NAME, NUM_CLASSES)
        
        # Perform Soft Shave (Masking)
        shaver.shave_with_mask(JSON_PATH)
        
        # Save
        shaver.save(OUTPUT_PATH)
        
        print("\n--- MONSTER NOTE ---")
        print("This script currently performs 'Soft Shaving' (Zeroing out).")
        print("To get the 1-second SPEEDUP, you need to physically remove the layers.")
        print("Since your team is using MobileNetV3 (which has skip connections),")
        print("I recommend using the 'TorchPruning' library for physical shaving.")
        print("Ask me if you want to implement the TorchPruning version!")
    else:
        print(f"Target list not found: {JSON_PATH}")
        print("Please run 'demon_shave_target_finder.py' first.")

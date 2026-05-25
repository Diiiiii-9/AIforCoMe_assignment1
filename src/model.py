"""
Convolutional Neural Network (CNN) architecture definition.
Baseline model: Lightweight, but preserves spatial features for convergence.
"""

import torch
import torch.nn as nn

class MicrostructureCNN(nn.Module):
    def __init__(self):
        super(MicrostructureCNN, self).__init__()
        
        # Feature Extractor: 3 Convolutional Blocks
        # Input shape: (Batch, 1, 65, 65)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # Output: (Batch, 4, 33, 33)
            
            # Block 2
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Output: (Batch, 8, 17, 17)
            
            # Block 3
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Output: (Batch, 16, 9, 9)

            # Global Average Pooling to reduce spatial dimensions while preserving features
            nn.AdaptiveAvgPool2d((1, 1))
            # Output: (Batch, 16, 1, 1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.regressor(x)
        return x

def count_parameters(model):
    """
    Utility function to count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example Usage & Sanity Check:
if __name__ == "__main__":
    model = MicrostructureCNN()
    print(f"Total Trainable Parameters: {count_parameters(model)}")
    
    # Dummy input mimicking a batch of 4 images of size 65x65
    dummy_input = torch.randn(4, 1, 65, 65)
    dummy_output = model(dummy_input)
    print(f"Output shape: {dummy_output.shape}") # Should be (4, 1)
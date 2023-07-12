"""
Author: Angelika Vižintin

################################################################################

Architectures file of Image Depixelation Project.
"""
import torch

class CNN(torch.nn.Module):
    
    def __init__(self, hidden_layers: int = 3, kernels: int = 32, kernel_size: int = 5):
        """CNN for image depixelation.

        :param hidden_layers: number of hidden layers
        :param kernels: number of kernels
        :param kernel_size: size of kernel(s)
        """
        super().__init__()

        in_channels = 2
        cnn = []
        for i in range(hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU())
            in_channels = kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        """Apply CNN to input ``x`` of shape ``(N, 2, H, W)``, where
        ``N=number of samples in a batch``, ``H`` is height of image and ``W`` is width of image."""
        # Apply hidden layers: (N, 2, H, W) -> (N, kernels, H, W)
        output = self.hidden_layers(x)
        # Apply output layer: (N, kernels, H, W) -> (N, 1, H, W)
        predictions = self.output_layer(output)
        return predictions

# Define the Swin Transformer model
from timm.models.swin_transformer import SwinTransformer
import torch
import torch.nn as nn
class SwinTransformerModel(nn.Module):
    def __init__(self, img_size=32, embed_dim=128, num_classes=10):
        super(SwinTransformerModel, self).__init__()

        # Swin Transformer uses patches of images instead of tokens
        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=4,  # Difference: Operates on patches of the input image
            in_chans=3,  # Assume RGB input
            embed_dim=embed_dim,
            depths=(2, 2),  # Similar to number of layers in Transformer
            num_heads=(4, 4),  # Number of attention heads in each stage
            num_classes=num_classes,
        )

    def forward(self, x):
        # Swin transformer directly outputs the class predictions
        return self.swin(x)

# Example Input
input_images = torch.randn(8, 3, 32, 32)  # (batch_size, channels, height, width)
swin_transformer = SwinTransformerModel()
swin_output = swin_transformer(input_images)
print('swin output',swin_output)  # Output shape: (batch_size, num_classes)
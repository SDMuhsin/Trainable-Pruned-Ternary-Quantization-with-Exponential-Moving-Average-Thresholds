import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvNeXtTinyClassificationModel(nn.Module):
    """
    ConvNeXt Tiny wrapper for classification.
    Loads torchvision's ConvNeXt Tiny (no pretrained weights) and adds log_softmax output
    to match the interface used by other models in this project.

    Architecture: ConvNeXt (Liu et al., 2022) - A pure ConvNet for the 2020s
    Parameters: ~28M
    """
    def __init__(self, num_classes=200):
        super(ConvNeXtTinyClassificationModel, self).__init__()
        # Load ConvNeXt Tiny from torchvision without pretrained weights
        self.model = models.convnext_tiny(weights=None, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    from torchinfo import summary

    device = torch.device("cpu")
    model = ConvNeXtTinyClassificationModel(num_classes=200)
    model.float()
    model.to(device)

    # Test with TinyImageNet input size (3x64x64)
    dummy = torch.randn(1, 3, 64, 64)
    out = model(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(summary(model, input_size=(1, 3, 64, 64)))

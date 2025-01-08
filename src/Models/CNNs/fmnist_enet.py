import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            Swish(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.squeeze(x).view(b, c)
        excitation = self.excitation(squeeze).view(b, c, 1, 1)
        return x * excitation

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expansion_factor, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        exp_channels = in_channels * expansion_factor
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            Swish()
        ) if expansion_factor != 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, 
                      padding=kernel_size//2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            Swish()
        )
        
        # Squeeze and Excitation
        self.se = SEBlock(exp_channels, reduction_ratio=int(1/se_ratio))
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.skip:
            x = x + identity
            
        return x

class EfficientNet(nn.Module):
    def __init__(self, input_channels=3, width_multiplier=1.0, depth_multiplier=1.0, 
                 dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()
        
        # Base architecture parameters
        self.blocks_args = [
            # expansion_factor, channels, num_layers, stride
            [1,  16,  1, 1],
            [6,  24,  2, 2],
            [6,  40,  2, 2],
            [6,  80,  3, 2],
            [6, 112,  3, 1],
            [6, 192,  4, 2],
            [6, 320,  1, 1],
        ]
        
        # Initial convolution
        self.conv_stem = nn.Sequential(
            nn.Conv2d(input_channels, int(32 * width_multiplier), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * width_multiplier)),
            Swish()
        )
        
        # Build blocks
        self.blocks = self._build_blocks(width_multiplier, depth_multiplier)
        
        # Head
        pen_channels = int(320 * width_multiplier)
        self.conv_head = nn.Sequential(
            nn.Conv2d(pen_channels, int(1280 * width_multiplier), 1, bias=False),
            nn.BatchNorm2d(int(1280 * width_multiplier)),
            Swish()
        )
        
        # Final linear layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(int(1280 * width_multiplier), num_classes)
        
        # Weight initialization
        self._initialize_weights()

    def _build_blocks(self, width_multiplier, depth_multiplier):
        blocks = []
        for expansion_factor, channels, num_layers, stride in self.blocks_args:
            out_channels = int(channels * width_multiplier)
            layers = int(math.ceil(num_layers * depth_multiplier))
            
            for i in range(layers):
                blocks.append(MBConvBlock(
                    in_channels=blocks[-1].out_channels if blocks else int(32 * width_multiplier),
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    expansion_factor=expansion_factor
                ))
                
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def create_efficientnet(model_name='efficientnet-b0', num_classes=1000, input_channels=3):
    """Create an EfficientNet model based on the model name, number of classes, and input channels."""
    params = {
        'efficientnet-b0': (1.0, 1.0, 0.2),
        'efficientnet-b1': (1.0, 1.1, 0.2),
        'efficientnet-b2': (1.1, 1.2, 0.3),
        'efficientnet-b3': (1.2, 1.4, 0.3),
        'efficientnet-b4': (1.4, 1.8, 0.4),
        'efficientnet-b5': (1.6, 2.2, 0.4),
        'efficientnet-b6': (1.8, 2.6, 0.5),
        'efficientnet-b7': (2.0, 3.1, 0.5),
    }
    
    if model_name not in params:
        raise ValueError(f'Model name {model_name} not supported')
        
    width_multiplier, depth_multiplier, dropout_rate = params[model_name]
    
    return EfficientNet(
        input_channels=input_channels,
        width_multiplier=width_multiplier,
        depth_multiplier=depth_multiplier,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )

if __name__ == '__main__':
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example 1: Standard EfficientNet with 3 input channels
    model_rgb = create_efficientnet('efficientnet-b0', num_classes=1000, input_channels=3)
    model_rgb = model_rgb.to(device)
    
    # Example 2: Grayscale EfficientNet with 1 input channel
    model_gray = create_efficientnet('efficientnet-b0', num_classes=1000, input_channels=1)
    model_gray = model_gray.to(device)
    
    # Print model summary (optional, requires torchinfo)
    # from torchinfo import summary
    # print(summary(model_rgb, input_size=(1, 3, 224, 224)))
    # print(summary(model_gray, input_size=(1, 1, 224, 224)))
    
    print("Models created successfully!")


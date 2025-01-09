"""
    Defines a DenseNet architecture that can be trained on the MNIST dataset or similar image classification tasks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# For Xavier Normal initialization
import numpy as np
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))


class _DenseLayer(nn.Module):
    """
    A single Dense layer (BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv).
    Growth rate is the amount of new channels generated at each layer.
    """
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        """
        :param num_input_features: # of input feature maps
        :param growth_rate: how many feature maps to add
        :param bn_size: multiplicative factor for bottleneck layers
        :param drop_rate: dropout rate
        """
        super(_DenseLayer, self).__init__()
        # 1x1 bottleneck
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        # 3x3 conv
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        # x is of shape (N, num_input_features, H, W)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        # Concatenate input with new feature maps
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    """
    DenseBlock: A series of `_DenseLayer`s where the output of each layer is
    concatenated to its inputs.
    """
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            layers.append(layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    """
    Transition layer between DenseBlocks to down-sample the feature maps
    (using 1x1 conv) and to reduce feature map sizes by 2x2 average pooling.
    """
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNetEncoder(nn.Module):
    """
    Encoder portion of the DenseNet (similar to torchvision's DenseNet-121 configuration).
    """
    def __init__(
            self,
            input_channels=1,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            bn_size=4,
            drop_rate=0.0,
    ):
        """
        :param input_channels: # of channels in the input data
        :param growth_rate: how many feature maps each DenseLayer adds
        :param block_config: tuple with the number of layers in each DenseBlock
        :param bn_size: multiplicative factor for bottleneck layers
        :param drop_rate: dropout rate
        """
        super(DenseNetEncoder, self).__init__()
        # Initial convolution
        # (We mimic ResNet: a 7x7 conv, stride=2, and a 2x2 or 3x3 pooling.)
        
        ''' This version done not fit the small dataset (feature maps reduce to 0 dim too quickly)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 2 * growth_rate, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )'''

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 2 * growth_rate, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            # optional single downsampling
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Each DenseBlock
        num_features = 2 * growth_rate
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:  # no transition after the last block
                out_features = num_features // 2
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=out_features)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = out_features

        # Final batch norm (typical in DenseNet)
        self.features.add_module("final_bn", nn.BatchNorm2d(num_features))

    def forward(self, x):
        x = self.features(x)
        return x


class DenseNetClassificationModel(nn.Module):
    """
    DenseNet that outputs class probabilities (for nb_classes) via log_softmax.
    """
    def __init__(self, input_channels=1, nb_classes=10):
        super(DenseNetClassificationModel, self).__init__()
        # Encoder
        self.encoder = DenseNetEncoder(input_channels=input_channels,
                                       growth_rate=32,
                                       block_config=(6, 12, 24, 16),
                                       bn_size=4,
                                       drop_rate=0.0)

        # Classification
        # We adapt the final classification to match the number of classes
        # (after we perform global average pooling).
        # For DenseNet-121, the final number of feature maps is 1024 by default.
        # But let's confirm from the final layer (depends on block_config).
        # For block_config=(6,12,24,16) and growth_rate=32, the last block has 16 layers.
        # So after the last DenseBlock:
        #   num_features = 2*growth_rate + sum(block_config)*growth_rate
        #   with transitions halving feature map count in between,
        #   net effect = 1024 for standard DenseNet-121.
        # We'll do an adaptive check after the encoder is built, to read out the actual #features.

        # Instead of hardcoding, we will assign it after building the model:
        self.classifier = None

        # A dummy forward pass to figure out the output channels
        # (Because we need the final number of channels after the last BN.)
        with torch.no_grad():
            temp_input = torch.randn(1, input_channels, 224, 224)  # or 28x28, see note below
            temp_output = self.encoder(temp_input)
            final_num_features = temp_output.shape[1]

        self.classifier = nn.Linear(final_num_features, nb_classes)

    def forward(self, x):
        # Pass through the DenseNet encoder
        x = self.encoder(x)

        # Final batch norm + ReLU is often done before pooling in DenseNet
        # (the encoder already has a final BN, so we just ReLU it).
        x = F.relu(x, inplace=True)

        # Global average pooling
        # Depending on input size, the spatial dimension after the final block can vary.
        # For MNIST-like images (28x28), it will be smaller than for 224x224,
        # but we can still use adaptive pooling.
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        # Output (log probabilities)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    device = torch.device("cpu")

    # Create DenseNet model
    model = DenseNetClassificationModel(input_channels=1, nb_classes=10)
    model.apply(weights_init)  # Xavier init if desired
    model = model.to(device)

    # Print summary (using torchinfo)
    # Let's assume MNIST shape of [1, 28, 28]
    summary(model, input_size=(1, 1, 28, 28))


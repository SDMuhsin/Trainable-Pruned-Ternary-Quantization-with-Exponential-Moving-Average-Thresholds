import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# For Xavier Normal initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)
        out = F.relu(out)

        return out

class ResNet50Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNet50Encoder, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 3, stride=1)   # Bottleneck layers
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels // Bottleneck.expansion, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet50ClassificationModel(nn.Module):
    def __init__(self, input_channels=3, nb_classes=10):
        super(ResNet50ClassificationModel, self).__init__()
        
        # Encoder
        self.encoder = ResNet50Encoder(input_channels)

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layer
        self.fc = nn.Linear(512 * Bottleneck.expansion, nb_classes)

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        # Output
        output = F.log_softmax(x, dim=1)
        
        return output

if __name__=='__main__':
    # Device setup
    device = torch.device("cpu")

    # Model initialization for CIFAR datasets (input channels set to 3 for RGB images)
    model = ResNet50ClassificationModel(input_channels=3, nb_classes=10)  
    model.float()
    model.to(device)

    # Summary of the model
    print(summary(model))

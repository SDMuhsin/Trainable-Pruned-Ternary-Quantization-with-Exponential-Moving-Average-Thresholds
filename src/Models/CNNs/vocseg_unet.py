"""
    Defines a U-Net style architecture for semantic segmentation on Pascal VOC dataset
"""
from math import floor
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

# For Xavier Normal initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))

# Class defining the encoder
class VocSegEncoderModel(nn.Module):
    def __init__(self, input_channels=3):
        super(VocSegEncoderModel, self).__init__()
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Store intermediates for skip connections
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        
        return x3, x2, x1  # Return intermediate features for skip connections

# Class defining the segmentation model
class VocSegModel(nn.Module):
    def __init__(self, input_channels=3, nb_classes=21):  # Pascal VOC has 20 classes + background
        super(VocSegModel, self).__init__()
        # Encoder
        self.encoder = VocSegEncoderModel(input_channels)
        
        # Decoder path with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, nb_classes, kernel_size=1)  # Final 1x1 conv for class prediction
        )
        
        # Initialize weights
        self.apply(weights_init)
        
    def forward(self, x):
        # Encoding
        x3, x2, x1 = self.encoder(x)
        
        # Decoding with skip connections
        d3 = self.dec3(x3)
        d3 = torch.cat([d3, x2], dim=1)  # Skip connection from encoder
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, x1], dim=1)  # Skip connection from encoder
        
        # Final convolutions and output
        logits = self.dec1(d2)
        
        # Returning logits directly (no softmax) as we'll use CrossEntropyLoss
        return logits

if __name__=='__main__':
    # Device
    device = torch.device("cpu")
    
    # Model
    model = VocSegModel(input_channels=3, nb_classes=21)
    model.float()
    model = model.to(device)
    
    # Summary of the model
    print(summary(model, input_size=(1, 3, 224, 224)))  # Assuming 224x224 input size

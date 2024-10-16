"""
    Defines a simple (2D) CNN architecture to be trained on the MNIST dataset
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
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))

# Class defining the encoder
class MnistEncoderModel(nn.Module):
    def __init__(self, input_channels=1, nb_classes=10):
        super(MnistEncoderModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        # Conv Block 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv Block 2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        return x

# Class defining the classification model
class MnistClassificationModel(nn.Module):
    def __init__(self, input_channels=1, nb_classes=10):
        super(MnistClassificationModel, self).__init__()
        # Encoder
        self.encoder = MnistEncoderModel(input_channels, nb_classes)

        # Classification layers
        self.fc1 = nn.Linear(80, 50) # If the input size is (20, 20, 1)
        self.fc2 = nn.Linear(50, nb_classes)
        self.fc_drop = nn.Dropout()

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Classification
        # Reshape
        x = x.view(-1, 80) # If the input size is (20, 20, 1)
        # FC 1
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        # FC 2
        x = self.fc2(x)
        # Output
        output = F.log_softmax(x)

        return output

if __name__=='__main__':
    # Device
    device = torch.device("cpu")

    # Model
    model = MnistClassificationModel(input_channels=1, nb_classes=10)
    model.float()
    model = model.to(device)

    # Summary of the model
    print(summary(model))

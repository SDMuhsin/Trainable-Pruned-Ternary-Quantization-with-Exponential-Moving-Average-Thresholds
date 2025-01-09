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

class BasicConv2d(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

###############################################################################
#                                Inception A                                  #
###############################################################################
class InceptionA(nn.Module):
    """
    Each branch ends with 32 channels => total = 128.
    """
    def __init__(self, in_channels, pool_features=32):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 24, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 24, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)  # 32

        branch3x3 = self.branch3x3_1(x)  # 24
        branch3x3 = self.branch3x3_2(branch3x3)  # 32

        branch3x3dbl = self.branch3x3dbl_1(x)  # 24
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)  # 32
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)  # 32

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)  # 32

        # => total = 32 + 32 + 32 + 32 = 128
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)

###############################################################################
#                               Reduction A                                   #
###############################################################################
class ReductionA(nn.Module):
    """
    Sum of:
      - branch3x3: 128
      - branch3x3dbl: 128
      - branch_pool: passes in_channels
    => If in_channels=128 => 128 + 128 + 128 = 384
    """
    def __init__(self, in_channels):
        super(ReductionA, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 128, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 128, kernel_size=3, stride=2)
        # pool => F.max_pool2d

    def forward(self, x):
        branch3x3 = self.branch3x3(x)  # => 128

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)  # => 128

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        # => cat => 128 + 128 + in_channels
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)

###############################################################################
#                               Inception B                                   #
###############################################################################
class InceptionB(nn.Module):
    """
    4 branches each with 128 out => total 512 channels
    """
    def __init__(self, in_channels, pool_features=128):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 128, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0))
        self.branch7x7_3 = BasicConv2d(64, 128, kernel_size=(1,7), padding=(0,3))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0))
        self.branch7x7dbl_3 = BasicConv2d(64, 64, kernel_size=(1,7), padding=(0,3))
        self.branch7x7dbl_4 = BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0))
        self.branch7x7dbl_5 = BasicConv2d(64, 128, kernel_size=(1,7), padding=(0,3))

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        # branch1x1 => 128
        branch1x1 = self.branch1x1(x)

        # branch7x7 => 128
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        # branch7x7dbl => 128
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # branch_pool => 128
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # => total = 128 + 128 + 128 + 128 = 512
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)

###############################################################################
#                               Reduction B                                   #
###############################################################################
class ReductionB(nn.Module):
    """
    Sum of:
      - branch3x3 => 192
      - branch7x7 => 192
      - branch_pool => in_channels
    => If in_channels=512 => final = 512 + 192 + 192 = 896
    """
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 192, kernel_size=3, stride=2)

        self.branch7x7_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(128, 128, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = BasicConv2d(128, 128, kernel_size=(7,1), padding=(3,0))
        self.branch7x7_4 = BasicConv2d(128, 192, kernel_size=3, stride=2)
        # branch_pool => F.max_pool2d

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)  # => 192

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_4(branch7x7)   # => 192

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        # => total = 192 + 192 + in_channels
        return torch.cat([branch3x3, branch7x7, branch_pool], 1)

###############################################################################
#                               Inception C                                   #
###############################################################################
class InceptionC(nn.Module):
    """
    We must carefully handle the intermediate concatenations:
      - branch1x1 => 192
      - branch3x3 => 384
      - branch3x3dbl => 384
      - pool => 128
    => total = 1088
    """
    def __init__(self, in_channels, pool_features=128):
        super(InceptionC, self).__init__()
        
        # 1x1 branch => out=192
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        # 3x3 branch => first 1x1 => 192, then splits into (1,3) & (3,1)
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(192, 192, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2b = BasicConv2d(192, 192, kernel_size=(3,1), padding=(1,0))

        # Double 3x3 branch => we have two splits along the way:
        # First split: in_channels -> 128, then (1,3) & (3,1) -> cat => 256
        # Second split: that 256 -> (1,3) & (3,1) -> cat => 384
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch3x3dbl_2a = BasicConv2d(128, 128, kernel_size=(1,3), padding=(0,1))
        self.branch3x3dbl_2b = BasicConv2d(128, 128, kernel_size=(3,1), padding=(1,0))
        # Now we have 256 channels to feed next convs:
        self.branch3x3dbl_3a = BasicConv2d(256, 192, kernel_size=(1,3), padding=(0,1))
        self.branch3x3dbl_3b = BasicConv2d(256, 192, kernel_size=(3,1), padding=(1,0))

        # Pool => out=128
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        # 1) branch1x1 => 192
        branch1x1 = self.branch1x1(x)

        # 2) branch3x3 => 384
        b3x3 = self.branch3x3_1(x)       # => 192
        b3x3_a = self.branch3x3_2a(b3x3) # => 192
        b3x3_b = self.branch3x3_2b(b3x3) # => 192
        branch3x3 = torch.cat([b3x3_a, b3x3_b], 1) # => 384

        # 3) branch3x3dbl => 384
        b3x3dbl = self.branch3x3dbl_1(x)       # => 128
        b3x3dbl_a = self.branch3x3dbl_2a(b3x3dbl) # => 128
        b3x3dbl_b = self.branch3x3dbl_2b(b3x3dbl) # => 128
        b3x3dbl = torch.cat([b3x3dbl_a, b3x3dbl_b], 1) # => 256

        b3x3dbl_a = self.branch3x3dbl_3a(b3x3dbl) # => 192
        b3x3dbl_b = self.branch3x3dbl_3b(b3x3dbl) # => 192
        branch3x3dbl = torch.cat([b3x3dbl_a, b3x3dbl_b], 1) # => 384

        # 4) branch_pool => 128
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # => total = 192 + 384 + 384 + 128 = 1088
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)

###############################################################################
#                             InceptionV4 Encoder                             #
###############################################################################
class InceptionV4Encoder(nn.Module):
    """
    Build the Inception-V4-style encoder with correct channel transitions.
    """
    def __init__(self, input_channels=3):
        super(InceptionV4Encoder, self).__init__()
        
        # Stem (simplified for smaller inputs like 32x32)
        self.stem = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)  # => 64
        )

        # Inception-A (3 blocks), each => 128 channels out
        self.inception_a = nn.Sequential(
            InceptionA(64,  pool_features=32),   # => 128
            InceptionA(128, pool_features=32),   # => 128
            InceptionA(128, pool_features=32)    # => 128
        )

        # Reduction-A => from 128 to 128+128+128=384
        self.reduction_a = ReductionA(128)  # => 384

        # Inception-B (3 blocks), each => 512 out
        self.inception_b = nn.Sequential(
            InceptionB(384, pool_features=128),  # => 512
            InceptionB(512, pool_features=128),  # => 512
            InceptionB(512, pool_features=128)   # => 512
        )

        # Reduction-B => from 512 to 512+192+192=896
        self.reduction_b = ReductionB(512)  # => 896

        # Inception-C (3 blocks), each => 1088 out
        self.inception_c = nn.Sequential(
            InceptionC(896,  pool_features=128), # => 1088
            InceptionC(1088, pool_features=128), # => 1088
            InceptionC(1088, pool_features=128)  # => 1088
        )

    def forward(self, x):
        x = self.stem(x)          # => 64
        x = self.inception_a(x)   # => 128
        x = self.reduction_a(x)   # => 384
        x = self.inception_b(x)   # => 512
        x = self.reduction_b(x)   # => 896
        x = self.inception_c(x)   # => 1088
        return x

###############################################################################
#                           InceptionV4 Classification                        #
###############################################################################
class InceptionV4ClassificationModel(nn.Module):
    def __init__(self, input_channels=3, nb_classes=10):
        super(InceptionV4ClassificationModel, self).__init__()
        self.encoder = InceptionV4Encoder(input_channels)
        # After last Inception-C => 1088 channels
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1088, nb_classes)

    def forward(self, x):
        x = self.encoder(x)      # => (N, 1088, H', W')
        x = self.avgpool(x)      # => (N, 1088, 1, 1)
        x = torch.flatten(x, 1)  # => (N, 1088)
        x = self.fc(x)           # => (N, nb_classes)
        return F.log_softmax(x, dim=1)

###############################################################################
#                                     MAIN                                    #
###############################################################################
if __name__ == '__main__':
    device = torch.device("cpu")
    model = InceptionV4ClassificationModel(input_channels=3, nb_classes=10)
    
    # Apply Xavier init
    model.apply(weights_init)
    model.float().to(device)

    # Show summary for an example input size (1, 3, 32, 32)
    print(summary(model, input_size=(1, 3, 32, 32)))


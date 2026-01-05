import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic Convolutional Block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    """Upsampling Block: Upsample -> Concat -> ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: current decoder feature, x2: skip connection feature
        x1 = self.up(x1)
        
        # Handle padding issues if sizes don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class SiameseUNet(nn.Module):
    """
    Siamese U-Net for Change Detection (FC-Siam-diff type).
    
    Structure:
    1. Siamese Encoder (shared weights) extracts features at multiple scales.
    2. Difference of features at each scale is computed.
    3. Decoder upsamples and fuses difference features.
    """
    def __init__(self, in_channels=3, n_classes=1):
        super(SiameseUNet, self).__init__()
        
        # Encoder (Shared Weights)
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        # Input channels = 1024 (from bottleneck) + 512 (from skip) = 1536? 
        # Wait, standard U-Net: Up(1024)->512, Concat(512, 512)->1024.
        # In Siamese Diff: Skip connection is |Enc1_A - Enc1_B|. 
        # So skip channels are same as encoder channels.
        
        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward_one(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        return e1, e2, e3, e4, b

    def forward(self, x1, x2):
        # Encoder 1
        e1_1, e2_1, e3_1, e4_1, b_1 = self.forward_one(x1)
        # Encoder 2
        e1_2, e2_2, e3_2, e4_2, b_2 = self.forward_one(x2)
        
        # Feature Difference (Absolute Diff)
        d1 = torch.abs(e1_1 - e1_2)
        d2 = torch.abs(e2_1 - e2_2)
        d3 = torch.abs(e3_1 - e3_2)
        d4 = torch.abs(e4_1 - e4_2)
        db = torch.abs(b_1 - b_2)
        
        # Decoder
        # We use the difference of the bottleneck as the starting point
        # And concatenate with the difference of encoder features
        x = self.up4(db, d4)
        x = self.up3(x, d3)
        x = self.up2(x, d2)
        x = self.up1(x, d1)
        
        out = self.final_conv(x)
        # Note: We return logits. Sigmoid is applied in loss function (BCEWithLogits) or prediction.
        return out

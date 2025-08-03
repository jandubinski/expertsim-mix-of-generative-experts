import torch
import torch.nn as nn


import torch
import torch.nn as nn


class AuxReg(nn.Module):
    def __init__(self, strength, output_dim=2, **kwargs):
        super(AuxReg, self).__init__()
        self.name = "regressor_v3_changed_loss_log_cosh"
        self.feature_extractor = FeatureExtractor()
        self.strength = strength
        # Fixed feature dimension from the feature extractor
        feature_dim = 64

        # Simplified regressor with 2 FC layers
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        features = self.feature_extractor(x)
        coords = self.regressor(features)
        return coords

    @staticmethod
    def regressor_loss(real_coords, fake_coords):
        return torch.mean(torch.log(torch.cosh(fake_coords - real_coords)))


# Improved feature extractor with residual blocks
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.name = "baseline_resnet_feature_extractor_v3_kernel_size_changed"

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # First pooling (56x30 -> 28x15)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # First residual block (32->32 channels)
        self.res1 = ResidualBlock(32, 32, kernel_size=5, stride=2)

        # Second pooling (28x15 -> 14x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Second residual block (32->64 channels)
        self.res2 = ResidualBlock(32, 64, kernel_size=5, stride=2)

        # Final pooling (14x7 -> 7x3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, 56, 30]
        x = self.pool1(x)  # [B, 32, 28, 15]

        x = self.res1(x)  # [B, 32, 28, 15]
        x = self.pool2(x)  # [B, 32, 14, 7]

        x = self.res2(x)  # [B, 64, 14, 7]
        x = self.pool3(x)  # [B, 64, 7, 3]

        # Global average pooling
        features = x.mean([2, 3])  # [B, 64]
        return features

# Baseline residual block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2

        # First convolutional path
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Second convolutional path
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection with dimensionality matching if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



#
# OLD MODEL
#
# class AuxReg(nn.Module):
#     feature_shape_conv_channels = 256
#
#     def __init__(self, strength, **kwargs):
#         super(AuxReg, self).__init__()
#         self.name = "aux-architecture-2-with_droppout_batchnorm"
#         self.strength = strength
#         # Feature extraction layers
#         self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.leaky3 = nn.LeakyReLU(0.1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#
#         self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.leaky4 = nn.LeakyReLU(0.1)
#         self.pool4 = nn.MaxPool2d((2, 1))
#
#         self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
#         self.bn5 = nn.BatchNorm2d(128)
#         self.leaky5 = nn.LeakyReLU(0.1)
#         self.pool5 = nn.MaxPool2d((2, 1))
#
#         self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.leaky6 = nn.LeakyReLU(0.1)
#
#         # Dropout layers (separated from feature path)
#         self.dropout = nn.Dropout(0.2)
#
#         # Final layers
#         self.flatten = nn.Flatten()
#         self.dense = nn.Linear(256 * 3 * 8, 2)  # Update dimensions based on input size
#
#     def forward(self, x):
#         # Original forward pass with dropout
#         x = self.pool3(self.dropout(self.leaky3(self.bn3(self.conv3(x)))))
#         x = self.pool4(self.dropout(self.leaky4(self.bn4(self.conv4(x)))))
#         x = self.pool5(self.dropout(self.leaky5(self.bn5(self.conv5(x)))))
#         x = self.dropout(self.leaky6(self.bn6(self.conv6(x))))
#         x = self.flatten(x)
#         return self.dense(x)
#
#     def get_features(self, img):
#         x = self.pool3(self.leaky3(self.bn3(self.conv3(img))))
#         x = self.pool4(self.leaky4(self.bn4(self.conv4(x))))
#         x = self.pool5(self.leaky5(self.bn5(self.conv5(x))))
#         x = self.leaky6(self.bn6(self.conv6(x)))
#         features = x.mean([2, 3])  # Global average pooling
#         return features  # [112, 256] E.g.

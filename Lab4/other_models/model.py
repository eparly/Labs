import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallScaleModel(nn.Module):
    def __init__(self, num_classes=21):
        super(SmallScaleModel, self).__init__()

        # Convolutional layer block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer block 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, padding=1, stride=1)

        # Upsampling layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.conv4(x)  # No activation or pooling here

        # Upsample to match the input size
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        """
        Initialize the segmentation model.
        Args:
            num_classes (int): Number of classes for segmentation.
        """
        super(SegmentationModel, self).__init__()

        # Encoder
        self.encoder1 = self._conv_block(3, 32)
        self.encoder2 = self._conv_block(32, 64)
        self.encoder3 = self._conv_block(64, 128)

        # ASPP-like Module
        self.aspp1 = nn.Conv2d(128, 128, kernel_size=1)
        self.aspp2 = nn.Conv2d(128, 128, kernel_size=3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(128, 128, kernel_size=3, padding=12, dilation=12)
        self.aspp4 = nn.AdaptiveAvgPool2d(1)

        self.aspp_out = nn.Conv2d(512, 128, kernel_size=1)

        # Decoder
        self.up1 = self._upsample_block(128, 64)
        self.up2 = self._upsample_block(64, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2))

        # ASPP
        aspp1 = self.aspp1(e3)
        aspp2 = self.aspp2(e3)
        aspp3 = self.aspp3(e3)
        aspp4 = F.interpolate(self.aspp4(e3), size=e3.shape[2:], mode='bilinear', align_corners=False)
        aspp_out = torch.cat([aspp1, aspp2, aspp3, aspp4], dim=1)
        aspp_out = self.aspp_out(aspp_out)

        # Decoder
        d1 = self.up1(aspp_out)
        d2 = self.up2(d1)
        out = self.final(d2)

        return F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _upsample_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, base_channels=32):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.dec3 = self.upconv_block(base_channels * 8, base_channels * 4)
        self.dec2 = self.upconv_block(base_channels * 8, base_channels * 2)
        self.dec1 = self.upconv_block(base_channels * 4, base_channels)

        # Output layer
        self.out_conv = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """Two convolutional layers followed by ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """Upsampling followed by a conv_block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        # print(enc1.shape)
        enc2 = self.enc2(self.pool(enc1))
        # print(enc2.shape)
        enc3 = self.enc3(self.pool(enc2))
        # print(enc3.shape)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        # print(bottleneck.shape)

        # Decoding path
        dec3 = self.dec3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        # print(dec3.shape)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        # print(dec2.shape)
        dec1 = self.dec1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        # print(dec1.shape)

        # Output layer
        output = self.out_conv(dec1)
        # print(output)
        return output


class UNetLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(UNetLite, self).__init__()
        self.enc1 = self.encoder_block(in_channels, 32)
        self.enc2 = self.encoder_block(32, 64)
        self.enc3 = self.encoder_block(64, 128)

        self.bottleneck = self.encoder_block(128, 256)

        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(256, 64)
        self.dec1 = self.decoder_block(128, 32)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        # print(enc1.shape)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        # print(enc2.shape)
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        # print(enc3.shape)

        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc3))
        # print(bottleneck.shape)

        dec3 = self.dec3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        # print(dec3.shape)

        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        # print(dec2.shape)

        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        # print(dec1.shape)

        output = self.final_conv(dec1)
        # print(output.shape)
        return output

if __name__ == "__main__":
    # model = UNet(num_classes=21)
    from small_pspnet import PSPNetLite
    model = PSPNetLite()
    model = model.to('cuda')
    info = summary(model, (3, 256, 256))
    sample_input = torch.randn((1, 3, 256, 256)).to('cuda')
    # output = model(sample_input)
    # print(model)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SmallPSPNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super(SmallPSPNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Initial convolutional layer to adapt input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Base convolutional blocks
        self.base_block1 = self._make_convolutional_block([32, 32, 64], '1')
        self.base_block2 = self._make_convolutional_block([64, 64, 128], '2')
        self.base_block3 = self._make_convolutional_block([128, 128, 256], '3')

        # Pyramid pooling module
        self.red_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling
        self.red_conv = nn.Conv2d(256, 64, kernel_size=1)

        self.yellow_pool = nn.AdaptiveAvgPool2d(2)
        self.yellow_conv = nn.Conv2d(256, 64, kernel_size=1)

        self.blue_pool = nn.AdaptiveAvgPool2d(4)
        self.blue_conv = nn.Conv2d(256, 64, kernel_size=1)

        self.green_pool = nn.AdaptiveAvgPool2d(8)
        self.green_conv = nn.Conv2d(256, 64, kernel_size=1)

        # Final classifier
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=3, padding=1)

    def _make_convolutional_block(self, filters, block_identifier):
        """
        Returns a sequential block of convolutional layers with skip connections.
        """
        in_channels, mid_channels, out_channels = filters
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, dilation=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, dilation=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward_base_block(self, x):
        """
        Pass input through the three base convolutional blocks.
        """
        x = self.base_block1(x)
        x = self.base_block2(x)
        x = self.base_block3(x)
        return x

    def pyramid_pooling(self, x):
        """
        Apply pyramid pooling on the input feature map.
        """
        h, w = x.size(2), x.size(3)

        # Red pooling (Global pooling)
        red = self.red_pool(x)
        red = self.red_conv(red)
        red = F.interpolate(red, size=(h, w), mode='bilinear', align_corners=False)

        # Yellow pooling
        yellow = self.yellow_pool(x)
        yellow = self.yellow_conv(yellow)
        yellow = F.interpolate(yellow, size=(h, w), mode='bilinear', align_corners=False)

        # Blue pooling
        blue = self.blue_pool(x)
        blue = self.blue_conv(blue)
        blue = F.interpolate(blue, size=(h, w), mode='bilinear', align_corners=False)

        # Green pooling
        green = self.green_pool(x)
        green = self.green_conv(green)
        green = F.interpolate(green, size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate results
        return torch.cat([x, red, yellow, blue, green], dim=1)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        # Initial layer to adjust input channels
        x = self.initial_conv(x)

        # Base convolutional block
        base_features = self.forward_base_block(x)

        # Pyramid pooling
        pooled_features = self.pyramid_pooling(base_features)

        # Classifier
        logits = self.final_conv(pooled_features)
        return logits, base_features



# Example usage
if __name__ == "__main__":
    input_shape = (1, 3, 256, 256)  # Batch size, channels, height, width
    model = SmallPSPNet(input_channels=3, num_classes=21)  # Change num_classes for your dataset
    model.to('cuda')
    model.eval()  # Set to evaluation mode for testing

    dummy_input = torch.randn(input_shape).to('cuda')
    logits, features = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    info = summary(model, (3, 256, 256))
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A simple convolutional block: Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PSPNetLite(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(PSPNetLite, self).__init__()
        # Encoder with more layers and filters
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Pyramid Pooling Module
        self.pyramid_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size) for output_size in [1, 2, 4, 8]
        ])
        self.ppm_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),  # Removed BatchNorm2d
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # Decoder with added layers
        self.decoder = nn.Sequential(
            nn.Conv2d(512 + 4 * 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)  # Logits
        )

        # Final Upsampling
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)  # Input height and width

        # Encoder
        enc = self.encoder(x)

        # Pyramid Pooling
        ppm_outs = [F.interpolate(self.ppm_convs[i](pool(enc)),
                                  size=enc.size()[2:], mode='bilinear', align_corners=True)
                    for i, pool in enumerate(self.pyramid_pooling)]
        ppm_out = torch.cat(ppm_outs + [enc], dim=1)

        # Decoder
        decoder_out = self.decoder(ppm_out)

        # Upsample to match input size
        output = self.upsample(decoder_out)

        return output

# Example
if __name__ == "__main__":
    model = PSPNetLite(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 21, 256, 256]
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
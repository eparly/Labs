import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class DilatedSkipNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(DilatedSkipNet, self).__init__()
        
        # Encoder with dilated convolutions
        self.enc1 = self.encoder_block(in_channels, 32, dilation=1)
        self.enc2 = self.encoder_block(32, 64, dilation=2)
        self.enc3 = self.encoder_block(64, 128, dilation=4)
        
        # Bottleneck with dilated convolutions
        self.bottleneck = self.encoder_block(128, 256, dilation=8)
        
        # Decoder with skip connections
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(256, 64)
        self.dec1 = self.decoder_block(128, 32)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Boundary refinement blocks
        self.refinement1 = self.boundary_refinement_block(256)
        self.refinement2 = self.boundary_refinement_block(128)
        self.refinement3 = self.boundary_refinement_block(64)
    
    def encoder_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
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
    
    def boundary_refinement_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [64, 256, 256]
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))  # [128, 128, 128]
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))  # [256, 64, 64]
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2))  # [512, 32, 32]
        
        # Decoder
        dec3 = self.dec3(bottleneck)  # [256, 64, 64]
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec3 = self.refinement1(dec3)  # Refinement block
        
        dec2 = self.dec2(dec3)  # [128, 128, 128]
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.refinement2(dec2)  # Refinement block
        
        dec1 = self.dec1(dec2)  # [64, 256, 256]
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.refinement3(dec1)  # Refinement block
        
        # Final output
        output = self.final_conv(dec1)  # [out_channels, 256, 256]
        return output


# used for feature based learning 
class FeatureDilatedSkipNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(FeatureDilatedSkipNet, self).__init__()
        
        # Encoder with dilated convolutions
        self.enc1 = self.encoder_block(in_channels, 32, dilation=1)   # Matches ResNet's Conv2_x
        self.enc2 = self.encoder_block(32, 64, dilation=2)            # Matches ResNet's Conv3_x
        self.enc3 = self.encoder_block(64, 128, dilation=4)           # Matches ResNet's Conv4_x
        
        # Bottleneck with dilated convolutions
        self.bottleneck = self.encoder_block(128, 256, dilation=8)    # Matches ResNet's Conv5_x
        
        # Decoder with skip connections
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(256, 64)
        self.dec1 = self.decoder_block(128, 32)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Boundary refinement blocks
        self.refinement1 = self.boundary_refinement_block(256)
        self.refinement2 = self.boundary_refinement_block(128)
        self.refinement3 = self.boundary_refinement_block(64)
        
        self.feature_maps = {} # extract features from each layer
    
    def encoder_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
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
    
    def boundary_refinement_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [32, 256, 256]
        self.feature_maps["enc1"] = enc1
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))  # [64, 128, 128]
        self.feature_maps["enc2"] = enc2
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))  # [128, 64, 64]
        self.feature_maps["enc3"] = enc3
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2))  # [256, 32, 32]
        self.feature_maps["bottleneck"] = bottleneck
        
        
        # Decoder
        dec3 = self.dec3(bottleneck)  # [128, 64, 64]
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec3 = self.refinement1(dec3)  # Refinement block
        
        dec2 = self.dec2(dec3)  # [64, 128, 128]
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.refinement2(dec2)  # Refinement block
        
        dec1 = self.dec1(dec2)  # [32, 256, 256]
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.refinement3(dec1)  # Refinement block
        
        # Final output
        output = self.final_conv(dec1)  # [out_channels, 256, 256]
        return output

if __name__ == "__main__":
    # model = UNet(num_classes=21)
    model = FeatureDilatedSkipNet(in_channels=3, out_channels=21)
    model = model.to('cuda')
    # info = summary(model, (3, 256, 256))
    sample_input = torch.randn((1, 3, 256, 256)).to('cuda')
    output = model(sample_input)
    # print(model)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, DeformableBlock, AttentionModule


class EnhancedGenerator(nn.Module):
    """
    Enhanced Generator (EP-GAN) - exact architecture as in user's file.
    Input: 4 channels (RGB + edge), Output: 3-channel RGB restored image.
    """
    def __init__(self):
        super(EnhancedGenerator, self).__init__()
        # Encoder
        self.conv_in = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3)
        self.enc1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # downsample
        self.enc2 = DeformableBlock(128, 256)
        self.enc3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1) # downsample


        # Residual tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(6)])
        self.attn = AttentionModule(256)


        # Decoder
        self.dec3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dec2 = DeformableBlock(256, 128)
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()


    def forward(self, x):
        # x: (B,4,H,W)
        x = F.leaky_relu(self.conv_in(x), 0.2)
        x = F.leaky_relu(self.enc1(x), 0.2)
        x = self.enc2(x)
        x = F.leaky_relu(self.enc3(x), 0.2)


        x = self.res_blocks(x)
        x = self.attn(x)


        x = F.leaky_relu(self.dec3(x), 0.2)
        x = self.dec2(x)
        x = F.leaky_relu(self.dec1(x), 0.2)
        x = self.conv_out(x)
        return self.tanh(x)
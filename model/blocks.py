import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual Block as in original code
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(16, channels)


    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.2)
        out = self.gn2(self.conv2(out))
        return out + residual


# DeformableBlock placeholder - original uses torchvision.ops.DeformConv2d if available
try:
    from torchvision.ops import DeformConv2d
    class DeformableBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(DeformableBlock, self).__init__()
            self.offset_conv = nn.Conv2d(in_ch, 18, kernel_size=3, stride=1, padding=1)
            self.deform = DeformConv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.gn = nn.GroupNorm(16, out_ch)
            self.act = nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x):
            offset = self.offset_conv(x)
            x = self.deform(x, offset)
            x = self.gn(x)
            return self.act(x)
except Exception:
    # Fallback to standard conv if DeformConv2d not available (keeps interface)
    class DeformableBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(DeformableBlock, self).__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            self.gn = nn.GroupNorm(16, out_ch)
            self.act = nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x):
            x = self.conv(x)
            x = self.gn(x)
            return self.act(x)


    # Attention Module as per original
    class AttentionModule(nn.Module):
        def __init__(self, channels):
            super(AttentionModule, self).__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=1)
            self.sig = nn.Sigmoid()
        def forward(self, x):
            return x * self.sig(self.conv(x))
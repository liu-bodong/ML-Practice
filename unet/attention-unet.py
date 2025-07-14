import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile
from thop import clever_format

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

   
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up_conv(x)
    

class AttentionGate(nn.Module):
    """This attention gate resamples the output from sigmoid"""
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(1)
        )
        
        # upsample back to F_x channels
        self.resampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, g):
        x1 = self.W_x(x)
        g = self.W_g(g)
        print(g.shape)
        psi = self.psi(torch.relu(x1 + g))
        psi = torch.sigmoid(psi)
        a =  self.resampler(psi)
        return a * x


class AttnUNet(nn.Module):
    def __init__(self, input_channels, out_channels, channels = [64, 128, 256, 512]):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_e1 = ConvBlock(input_channels, channels[0])
        self.conv_e2 = ConvBlock(channels[0], channels[1])
        self.conv_e3 = ConvBlock(channels[1], channels[2])
        self.conv_e4 = ConvBlock(channels[2], channels[3])
        
        self.ag1 = AttentionGate(F_g = channels[3], F_x = channels[2], F_int = channels[2])
        self.up_conv1 = UpConvBlock(channels[3], channels[2])
        self.conv_d1 = ConvBlock(channels[3], channels[2])
        
        self.ag2 = AttentionGate(F_g = channels[2], F_x = channels[1], F_int = channels[1])
        self.up_conv2 = UpConvBlock(channels[2], channels[1])
        self.conv_d2 = ConvBlock(channels[2], channels[1])
        
        self.ag3 = AttentionGate(F_g = channels[1], F_x = channels[0], F_int = channels[0])
        self.up_conv3 = UpConvBlock(channels[1], channels[0])
        self.conv_d3 = ConvBlock(channels[1], channels[0])
        
        self.conv_1x1 = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=1, stride=1)
        
    def forward(self, x):
        # contracting path
        print("encoder")
        print(x.shape)
        
        x1 = self.conv_e1(x)
        print(x1.shape)
        
        x2 = self.conv_e2(self.maxpool(x1))
        print(x2.shape)

        x3 = self.conv_e3(self.maxpool(x2))
        print(x3.shape)
        
        x4 = self.conv_e4(self.maxpool(x3))
        print(x4.shape)
        
        # expansive path
        print("decoder")
        print("a1")
        a1 = self.ag1(x3, x4)
        print(a1.shape)
        
        d1 = torch.cat((self.up_conv1(x4), a1), dim=1)
        print(d1.shape)
        
        d1 = self.conv_d1(d1)
        print(d1.shape)

        print("a2")
        a2 = self.ag2(x2, d1)
        print(a2.shape)
        
        d2 = torch.cat((self.up_conv2(d1), a2), dim=1)
        print(d2.shape)
        
        d2 = self.conv_d2(d2)
        print(d2.shape)
        
        print("a3")
        a3 = self.ag3(x1, d2)
        print(a3.shape)
        
        d3 = torch.cat((self.up_conv3(d2), a3), dim=1)
        print(d3.shape)
        
        d3 = self.conv_d3(d3)
        print(d3.shape)
        
        return self.conv_1x1(d3)
        
         
if __name__ == "__main__":
    with torch.no_grad():
        input = torch.rand(1, 3, 256, 256).to("cuda:0")
        model = AttnUNet(3, 1).to("cuda:0")
        flops, params = profile(model, (input,))

        print("-" * 30)
        print(f'Flops  = {clever_format(flops, format="%.5f")}')
        print(f'Params = {clever_format(params, format="%.5f")}')
        
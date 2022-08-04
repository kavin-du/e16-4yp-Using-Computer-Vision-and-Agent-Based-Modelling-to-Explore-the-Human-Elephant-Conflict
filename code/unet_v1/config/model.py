import torch
import torchvision 

import numpy as np

class ConvRelu(torch.nn.Module):
    
    def __init__(self, in_depth, out_depth):
        super(ConvRelu, self).__init__()
        self.conv = torch.nn.Conv2d(in_depth, out_depth, kernel_size=3, stride=1, padding=1)
        self.activation = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(torch.nn.Module):
    
    def __init__(self, in_depth, middle_depth, out_depth):
        super(DecoderBlock, self).__init__()
        self.conv_relu = ConvRelu(in_depth, middle_depth)
        self.conv_transpose = torch.nn.ConvTranspose2d(middle_depth, out_depth, kernel_size=4, stride=2, padding=1)
        self.activation = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv_relu(x)
        x = self.conv_transpose(x)
        x = self.activation(x)
        return x  

class UNetResNet(torch.nn.Module):
    
    def __init__(self, num_classes):

        super(UNetResNet, self).__init__()
        
        self.encoder = torchvision.models.resnet101(pretrained=True)
        
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        
        self.pool = torch.nn.MaxPool2d(2, 2)      
        self.center = DecoderBlock(2048, 512, 256)
        
        self.dec5 = DecoderBlock(2048 + 256, 512, 256)
        self.dec4 = DecoderBlock(1024 + 256, 512, 256)
        self.dec3 = DecoderBlock(512 + 256, 256, 64)
        self.dec2 = DecoderBlock(256 + 64, 128, 128)
        self.dec1 = DecoderBlock(128, 128, 32)
        self.dec0 = ConvRelu(32, 32)
        self.final = torch.nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    #(convolution => [BN] => ReLU) * 2

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    #Downscaling with maxpool then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    #Upscaling then double conv

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
"""        
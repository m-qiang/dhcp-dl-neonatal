import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self, C_in=1, C_hid=[16,32,64,128,128], C_out=1, K=3):
        super(UNet, self).__init__()

        """
        3D U-Net for cortical ribbon segmentation.
        
        Args:
        - C_in: input channels
        - C_hid: hidden channels
        - C_out: output channels
        - K: kernel size
        
        Inputs:
        - x: input volume (B,C_in,D1,D2,D3)
        
        Returns:
        - x: output volume (B,C_out,D1,D2,D3)
        """
        
        # convolutional encoder
        self.conv1 = nn.Conv3d(in_channels=C_in, out_channels=C_hid[0],
                               kernel_size=K, stride=1, padding=K//2)
        self.conv2 = nn.Conv3d(in_channels=C_hid[0], out_channels=C_hid[1],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv3 = nn.Conv3d(in_channels=C_hid[1], out_channels=C_hid[2],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv4 = nn.Conv3d(in_channels=C_hid[2], out_channels=C_hid[3],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv5 = nn.Conv3d(in_channels=C_hid[3], out_channels=C_hid[4],
                               kernel_size=K, stride=1, padding=K//2)
        
        # convolutional decoder
        self.deconv4 = nn.Conv3d(in_channels=C_hid[4]+C_hid[3],
                                 out_channels=C_hid[3], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv3 = nn.Conv3d(in_channels=C_hid[3]+C_hid[2],
                                 out_channels=C_hid[2], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv2 = nn.Conv3d(in_channels=C_hid[2]+C_hid[1], 
                                 out_channels=C_hid[1], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv1 = nn.Conv3d(in_channels=C_hid[1]+C_hid[0],
                                 out_channels=C_hid[0], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv0 = nn.Conv3d(in_channels=C_hid[0], out_channels=C_out,
                                 kernel_size=K, stride=1, padding=K//2)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                              align_corners=False)
        
    def forward(self, x):
        # encode
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)

        # decode
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)

        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        
        x = self.deconv0(x)
        
        return x

"""
Spherical U-Net

The code is retrieved from and modified based on:
https://github.com/zhaofenqiang/SphericalUNetPackage/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class conv_layer(nn.Module):
    """
    The convolutional layer on icosahedron discretized sphere 
    using 1-ring filter.
    
    Args:
    - C_in: input channels
    - C_out: output channels
    - neigh_orders: indices of neighborhoods

    Input: 
    - x: input features, (B,|V|,C_in) torch.Tensor
    
    Return:
    - x: output features, (B,|V|,C_out) torch.Tensor
    """  
    def __init__(self, C_in, C_out, neigh_orders):
        super(conv_layer, self).__init__()
        
        self.C_in = C_in
        self.C_out = C_out
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7 * C_in, C_out)
        
    def forward(self, x):
        mat = x[:,self.neigh_orders].reshape(
            x.shape[0], x.shape[1], 7*self.C_in)
        out = self.weight(mat)
        return out
    
    
class pool_layer(nn.Module):
    """
    The pooling layer on icosahedron discretized sphere
    using 1-ring filter.
    
    Args:
    - neigh_orders: indices of neighborhoods
    - pooling_type: ['mean', 'max']

    Input: 
    - x: input features, (B,|V|,C) torch.Tensor
    
    Return:
    - x: output features, (B,(|V|+6)/4,C) torch.Tensor
    """
    
    def __init__(self, neigh_orders, pooling_type='mean'):
        super(pool_layer, self).__init__()
        
        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type
        
    def forward(self, x):
        num_nodes = (x.size(1)+6)//4
        feat_num = x.size(2)
        x = x[:, self.neigh_orders[0:num_nodes*7]].reshape(
            x.shape[0], num_nodes, feat_num, 7)
        if self.pooling_type == "mean":
            return x.mean(dim=-1)
        elif self.pooling_type == "max":
            return x.max(dim=-1)[0]
        
        
class upconv_layer(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere
    using 1-ring filter
    
    Args:
    - C_in: input channels
    - C_out: output channels
    - upconv_top_index: indices for upsampling
    - upconv_down_index: indices for upsampling

    Input: 
    - x: input features, (B,|V|,C_in) torch.Tensor
    
    Return:
    - x: output features, (B,4|V|-6,C_out) torch.Tensor
    """  
    def __init__(self,C_in, C_out, upconv_top_index, upconv_down_index):
        super(upconv_layer, self).__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        self.weight = nn.Linear(C_in, 7 * C_out)
        
    def forward(self, x):
        n_batch = x.shape[0]
        n_vert = x.shape[1]
        x = self.weight(x)
        x = x.reshape(n_batch, n_vert * 7, self.C_out)
        x1 = x[:,self.upconv_top_index]
        x2 = x[:,self.upconv_down_index].reshape(n_batch, -1, self.C_out, 2)
        x = torch.cat([x1, x2.mean(dim=-1)], dim=1)
        return x
    
    
class conv_block(nn.Module):
    """
    The convolutional blocks: (Conv => LeakyReLU) * 2
    
    Args:
    - C_in: input channels
    - C_out: output channels
    - neigh_orders: indices of neighborhoods

    Input: 
    - x: input features, (B,|V|,C_in) torch.Tensor
    
    Return:
    - x: output features, (B,|V|,C_out) torch.Tensor
    """  
    
    def __init__(self, C_in, C_out, neigh_orders):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            conv_layer(C_in, C_out, neigh_orders),
            # nn.LayerNorm(C_out),
            nn.LeakyReLU(0.2),
            conv_layer(C_out, C_out, neigh_orders),
            # nn.LayerNorm(c_out),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.block(x)

"""
Spherical U-Net

The code is retrieved from and modified based on:
https://github.com/zhaofenqiang/SphericalUNetPackage/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib

from sphere.net.layers import (
    conv_block,
    conv_layer,
    pool_layer,
    upconv_layer)

from sphere.net.interp import (
    sphere_interpolate,
    get_bi_interp)

from sphere.net.utils import (
    get_neighs_order,
    get_upconv_index)



class SUNet(nn.Module):
    """
    Spherical U-Net (SUNet).
    
    Args:
    - C_in: input channels
    - C_hid: hidden channels
    - neigh_orders: indices of neighborhoods
    - upconv_top_index: indices for upsampling
    - upconv_down_index: indices for upsampling
    
    Input: 
    - x: input features, (B,|V|,C_in) torch.Tensor
    
    Return:
    - svf: spherical stationary velocity field (SVF),
    (B,|V|,3) torch.Tensor
    """
    
    def __init__(self,
                 C_in=3, 
                 C_hid=[32, 64, 128, 256, 512],
                 neigh_orders=None,
                 upconv_top_index=None,
                 upconv_down_index=None):

        super(SUNet, self).__init__()
        # Ni = {163842, 40962, 10242, 2562, 642, 162}

        self.conv1 = conv_block(C_in, C_hid[0], neigh_orders[0])
        self.conv2 = conv_block(C_hid[0], C_hid[1], neigh_orders[1])
        self.conv3 = conv_block(C_hid[1], C_hid[2], neigh_orders[2])
        self.conv4 = conv_block(C_hid[2], C_hid[3], neigh_orders[3])
        self.conv5 = conv_block(C_hid[3], C_hid[4], neigh_orders[4])
        self.conv6 = conv_block(C_hid[4], C_hid[4], neigh_orders[4])

        self.pool1 = pool_layer(neigh_orders[0])
        self.pool2 = pool_layer(neigh_orders[1])
        self.pool3 = pool_layer(neigh_orders[2])
        self.pool4 = pool_layer(neigh_orders[3])
        
        self.deconv5 = conv_block(C_hid[4]+C_hid[4], C_hid[4], neigh_orders[4])
        self.deconv4 = conv_block(C_hid[4]+C_hid[3], C_hid[3], neigh_orders[3])
        self.deconv3 = conv_block(C_hid[3]+C_hid[2], C_hid[2], neigh_orders[2])
        self.deconv2 = conv_block(C_hid[2]+C_hid[1], C_hid[1], neigh_orders[1])
        self.deconv1 = conv_block(C_hid[1]+C_hid[0], C_hid[0], neigh_orders[0])
        
        self.up4 = upconv_layer(C_hid[4], C_hid[4],
                                upconv_top_index[3], upconv_down_index[3])
        self.up3 = upconv_layer(C_hid[3], C_hid[3],
                                upconv_top_index[2], upconv_down_index[2])
        self.up2 = upconv_layer(C_hid[2], C_hid[2],
                                upconv_top_index[1], upconv_down_index[1])
        self.up1 = upconv_layer(C_hid[1], C_hid[1],
                                upconv_top_index[0], upconv_down_index[0])

        self.flow = conv_layer(C_hid[0], 3, neigh_orders[0])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        x = self.conv6(x5)
        
        x = torch.cat([x, x5], dim=-1)
        x = self.deconv5(x)
        
        x = self.up4(x)
        x = torch.cat([x, x4], dim=-1)
        x = self.deconv4(x)
        
        x = self.up3(x)
        x = torch.cat([x, x3], dim=-1)
        x = self.deconv3(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=-1)
        x = self.deconv2(x)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=-1)
        x = self.deconv1(x)
        svf = self.flow(x)
        
        return svf
    
    
class SphereDeform(nn.Module):
    """
    Spherical diffeomorphic deformation network for
    spherical mapping.
    
    Args:
    - C_in: input channels
    - C_hid: hidden channels
    
    Input: 
    - feat_in: input features, (B,|V|,C_in) torch.Tensor
    - vert_in: vertices of input sphere, (B,|V|,3) torch.Tensor
    
    Return:
    - vert_pred: vertices of deformed sphere, (B,|V|,3) torch.Tensor
    """
    
    def __init__(self,
                 C_in=3,
                 C_hid=[32, 64, 128, 256, 512],
                 device='cpu'):
        super(SphereDeform, self).__init__()
        
        # load template sphere for integration
        n_vert = 163842

        self.bi_interp = get_bi_interp(n_vert, device)
        vert_fixed = nib.load(
            './template/sphere_'+str(n_vert)+'.surf.gii').agg_data('pointset')
        self.vert_fixed = torch.Tensor(vert_fixed).float().to(device)

        # load multiscale spherical u-net
        self.neigh_orders = get_neighs_order()
        upconv_top_index, upconv_down_index = get_upconv_index()
        
        self.msunet = SUNet(
            C_in, C_hid, self.neigh_orders,
            upconv_top_index, upconv_down_index).to(device)

    def forward(self, feat_in, vert_in, n_steps=7):
        svf = self.msunet(feat_in)
        # project to tangent space
#         svf = svf - self.vert_fixed * (svf * self.vert_fixed).sum(
#             dim=-1, keepdim=True)
        # integrate velocity field
        phi = self.integrate(svf, n_steps)
        
        # smooth deformation fields
        phi = phi[:, self.neigh_orders[0]].reshape(
            -1, phi.shape[1], 7, 3).mean(-2)

        # warp deformation field
        vert_pred = sphere_interpolate(vert_in, phi, self.bi_interp)
        vert_pred = vert_pred / vert_pred.norm(dim=-1, keepdim=True)
        return vert_pred  #, phi_list
    
    def integrate(self, flow, n_steps=7):
        flow = flow / (2**n_steps)
        vert_warped = self.vert_fixed + flow
        vert_warped = vert_warped / vert_warped.norm(dim=-1, keepdim=True)
        # compute exp
        for i in range(n_steps):
            vert_warped = sphere_interpolate(
                vert_warped, vert_warped.clone(), self.bi_interp)
            vert_warped = vert_warped / vert_warped.norm(dim=-1, keepdim=True)
        return vert_warped
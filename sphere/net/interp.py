"""
Spherical U-Net

The code is retrieved from and modified based on:
https://github.com/zhaofenqiang/SphericalUNetPackage/
"""

import numpy as np
import torch
import math


def get_bi_interp(n_vert, device):
    """
    Get indices and weights for bilinear interpolation.
    
    Inputs:
    - n_vert: number of vertices
    - device: [cuda, cpu]
    
    Returns:
    - inter_indices: indices for interpolation
    - inter_weights: weights for interpolation
    """
    inter_indices = np.load('./sphere/net/neigh_indices/img_indices_'+str(n_vert)+'.npy')
    inter_indices = torch.from_numpy(inter_indices.astype(np.int64)).to(device)
    inter_weights = np.load('./sphere/net/neigh_indices/img_weights_'+str(n_vert)+'.npy')
    inter_weights = torch.from_numpy(inter_weights.astype(np.float32)).to(device)
    return inter_indices, inter_weights


def get_latlon_img(bi_interp, feat):
    """
    Get longitude/latitude parameterization of the sphere
    
    Inputs:
    - bi_interp: indices and weights for interpolation
    - feat: input features
    
    Returns:
    - img: 2D longitude/latitude image
    """
    
    inter_indices, inter_weights = bi_interp
    width = int(np.sqrt(len(inter_indices)))
    img = torch.sum(((feat[0,inter_indices.flatten()]).reshape(
        inter_indices.shape[0], inter_indices.shape[1], feat.size(-1))) *\
        ((inter_weights.unsqueeze(2)).repeat(1,1,feat.size(-1))), 1)
    img = img.reshape(width, width, feat.size(-1))
    
    return img


def sphere_interpolate(vert_interp, feat, bi_interp):
    """
    Spherical bilinear interpolation.
    Assume vert_fix are on the standard icosahedron discretized spheres.
    
    Inputs:
    - vert_interp: vertices for interpolation
    - feat: input features to be interpolated
    - bi_interp: indices and weights for interpolation
    
    Returns:
    - feat_interp: interpolated features
    """
    img = get_latlon_img(bi_interp, feat)
    width = img.shape[0]
    
    # Cartesian to spherical coordinate
    x, y, z = vert_interp[:,:,0], vert_interp[:,:,1], vert_interp[:,:,2]
    phi = torch.acos(z.clamp(min=-1+1e-6, max=1-1e-6) / 1.0)
    row = phi / (math.pi / (width-1))

    theta = torch.atan2(y, x)
    theta = theta + math.pi * 2
    theta = torch.remainder(theta, math.pi * 2)

    col = theta / (2*math.pi / (width-1))
    feat_interp = bilinear_interpolate(img, col, row)
    
    return feat_interp


def bilinear_interpolate(img, x, y):
    """
    Bilinear interpolation.
    
    Inputs:
    - img: input image
    - x,y: coordinates
    
    Returns:
    - interpolated value
    """
    x = torch.clamp(x, 1e-4, img.size(1)-1 - 1e-4)
    y = torch.clamp(y, 1e-4, img.size(0)-1 - 1e-4)
    
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    Ia = img[y0.long(), x0.long()]
    Ib = img[y1.long(), x0.long()]
    Ic = img[y0.long(), x1.long()]
    Id = img[y1.long(), x1.long()]

    wa = ((x1-x) * (y1-y)).unsqueeze(-1)
    wb = ((x1-x) * (y-y0)).unsqueeze(-1)
    wc = ((x-x0) * (y1-y)).unsqueeze(-1)
    wd = ((x-x0) * (y-y0)).unsqueeze(-1)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


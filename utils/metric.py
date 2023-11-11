import os
import numpy as np
import nibabel as nib
import subprocess
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from utils.mesh import (
    laplacian_smooth,
    vert_normal
)
from utils.inflate import mris_inflate
from utils.mesh import adjacency_matrix


def metric_dilation(metric, face, roi=None, n_iters=10):
    """
    Metric dilation within the region of interest.
    
    Inputs:
    - metric: surface metric, (1,|V|,1) torch.Tensor
    - face: mesh faces, (1,|V|,3) torch.LongTensor
    - roi: region of interest, (|V|) numpy.array
    - n_iters: number of dilation iterations, int
    """
    
    # compute adjacency matrix
    A = adjacency_matrix(face)
    for n in range(n_iters):
        # find all nonzero metric values
        metric_nonzero = 1. - (metric == 0).float()
        # weighted without the zero metric
        degree = A.bmm(metric_nonzero) + 1e-8
        # only update the metrics with zero values
        metric += (1. - metric_nonzero) * A.bmm(metric) / degree
    metric = metric[0,:,0].cpu().numpy()
    if roi is not None:
        metric = metric * roi
    return metric 


def cortical_thickness(vert_white, vert_pial):
    """
    Estimate cortical thickness between
    wm and pial surfaces.
    
    Inputs:
    - vert_white: vertices of wm surface, (1,|V|,3) torch.Tensor
    - vert_pial: vertices of pial surface, (1,|V|,3) torch.Tensor
    
    Returns:
    - thickness: cortical thickness, (|V|) numpy.array
    """
    thickness = 0
    vert_white_arr = vert_white[0].cpu().numpy()
    vert_pial_arr = vert_pial[0].cpu().numpy()

    kdtree = cKDTree(vert_white_arr, leafsize=20)
    thickness += kdtree.query(vert_pial_arr)[0] / 2
    kdtree = cKDTree(vert_pial_arr, leafsize=20)
    thickness += kdtree.query(vert_white_arr)[0] / 2
    
    return thickness
    
    
def curvature(vert, face, curv_type='mean', smooth_iters=0):
    """
    Estimate curvature of the surface.
    
    This function reimplements the method in the connectome
    workbench commandline. For originial code please see: 
    https://github.com/Washington-University/workbench/blob/
    master/src/Algorithms/AlgorithmSurfaceCurvature.cxx

    Inputs:
    - vert: mesh vertices, (1,|V|,3) torch.Tensor
    - face: mesh faces, (1,|F|,3) torch.LongTensor
    - curv_type: ['mean', 'gaussian']
    - smooth_iters: number of smoothing iterations
    
    Returns:
    - curv: curvature, (|V|) numpy.array
    """
    
    n_vert = vert.shape[1]
    normal = vert_normal(vert, face)
    basis = (normal[:,:,0].abs()>normal[:,:,1].abs()).unsqueeze(-1).float()
    basis = torch.cat([1-basis, basis, torch.zeros_like(basis)], dim=-1)
    ihat = torch.cross(normal, basis, dim=-1)
    ihat = ihat / ihat.norm(dim=-1, keepdim=True)
    jhat = torch.cross(normal, ihat, dim=-1)
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], dim=0).T  # compute edges
    
    # edge[0]: center vertex, edge[1]: neighborhood vertex
    neigh_normal = normal[:,edge[1]]  # find normals for neighborhoods
    neigh_diff = vert[:,edge[1]] - vert[:,edge[0]]
    ihat = ihat[:, edge[0]]
    jhat = jhat[:, edge[0]]

    norm_proj_0 = (neigh_normal*ihat).sum(-1)
    norm_proj_1 = (neigh_normal*jhat).sum(-1)
    diff_proj_0 = (neigh_diff*ihat).sum(-1)
    diff_proj_1 = (neigh_diff*jhat).sum(-1)

    sig_x = diff_proj_0 * diff_proj_0
    sig_xy = diff_proj_0 * diff_proj_1
    sig_y = diff_proj_1 * diff_proj_1
    norm_x = norm_proj_0 * diff_proj_0
    norm_xy = norm_proj_0 * diff_proj_1 + norm_proj_1 * diff_proj_0
    norm_y = norm_proj_1 * diff_proj_1

    # build adjacency matrix
    values = torch.cat([sig_x, sig_xy, sig_y, norm_x, norm_xy, norm_y]).T
    neigh_matrix = torch.sparse_coo_tensor(
        edge, values, (n_vert, n_vert, 6)).unsqueeze(0)
    # sum all neighbors
    values_per_vertex = torch.sparse.sum(neigh_matrix, dim=-2).to_dense()
    sig_x = values_per_vertex[...,0]
    sig_xy = values_per_vertex[...,1]
    sig_y = values_per_vertex[...,2]
    norm_x = values_per_vertex[...,3]
    norm_xy = values_per_vertex[...,4]
    norm_y = values_per_vertex[...,5]

    sig_xy2 = sig_xy * sig_xy
    denom = (sig_x + sig_y) * (-sig_xy2 + sig_x * sig_y)
    denom_ = denom + 1e-8    # avoid divide by 0

    a = (norm_x * (-sig_xy2 + sig_x * sig_y + sig_y * sig_y) -
         norm_xy * sig_xy * sig_y + norm_y * sig_xy2) / denom_
    b = (-norm_x * sig_xy * sig_y + norm_xy * sig_x * sig_y -
         norm_y * sig_x * sig_xy) / denom_
    c = (norm_x * sig_xy2 -norm_xy * sig_x * sig_xy +
         norm_y * (sig_x * sig_x - sig_xy2 + sig_x * sig_y)) / denom_
    trC = a + c
    detC = a * c - b * b
    temp = trC * trC - 4 * detC
    delta = temp.abs().sqrt()
    k1 = (trC + delta) / 2
    k2 = (trC - delta) / 2

    # set curvature to zero if denom=0 or temp<0
    k1[torch.where(denom==0)] = 0.
    k2[torch.where(denom==0)] = 0.
    k1[torch.where(temp<0)] = 0.
    k2[torch.where(temp<0)] = 0.
    
    if curv_type == 'gaussian':
        curv =  k1 * k2
    elif curv_type == 'mean':
        curv = (k1 + k2) / 2
        
    # smooth the curvature
    curv = laplacian_smooth(
        curv.unsqueeze(-1), face, lambd=1.0, n_iters=smooth_iters)
    return curv[0,:,0].cpu().numpy()


def sulcal_depth(vert, face, verbose=False):
    """
    Estimate sulcal depth by inflating the surface.
    
    Inputs:
    - vert: mesh vertices, (1,|V|,3) torch.Tensor
    - face: mesh faces, (1,|F|,3) torch.LongTensor
    - verbose: if report
    
    Returns:
    - sulc: sulcal depth, (|V|) numpy.array
    """
    
    _, sulc = mris_inflate(
        vert, face, track_sulcal_depth=True, verbose=verbose)
    return sulc


def myelin_map(
    subj_dir,
    surf_hemi='left',
    myelin_sigma=5/(2*(np.sqrt(2*np.log(2))))
):
    """
    Estimate the myelin map from the T1/T2 ratio, cortical ribbon,
    cortical thickness and midthickness surface.
    
    Inputs:
    - subj_dir: prefix of the directory for the subject
    - surf_hemi: ['left', 'right']
    - myelin_sigma: standard deviation for volume-to-surface mapping

    Returns:
    - myelin: myelin map, (|V|) numpy.array
    """
    
    # create myelin map
    subprocess.run(
        'wb_command -volume-to-surface-mapping '+\
        subj_dir+'_T1wDividedByT2w.nii.gz '+\
        subj_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii '+\
        subj_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii '+\
        '-myelin-style '+subj_dir+'_ribbon.nii.gz '+\
        subj_dir+'_hemi-'+surf_hemi+'_thickness.shape.gii '+\
        str(myelin_sigma), shell=True)
    
    # remove the gifti file, and save it later to change colormap
    myelin = nib.load(
        subj_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii')
    myelin = myelin.agg_data()
    os.remove(
        subj_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii')
    return myelin
    
    
def smooth_myelin_map(
    subj_dir,
    surf_hemi='left',
    smooth_sigma=4/(2*(np.sqrt(2*np.log(2))))
):
    """
    Smooth the myelin map.
    
    Inputs:
    - subj_dir: prefix of the directory for the subject
    - surf_hemi: ['left', 'right']
    - smooth_sigma: standard deviation for smoothing

    Returns:
    - smoothed_myelin: smoothed myelin map, (|V|) numpy.array
    """

    # smooth the myelin map
    subprocess.run(
        'wb_command -metric-smoothing '+\
        subj_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii '+\
        subj_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii '+\
        str(smooth_sigma)+' '+\
        subj_dir+'_hemi-'+surf_hemi+'_smoothed_myelinmap.shape.gii',
        shell=True)
    
    # remove the gifti file, and save it later to change colormap
    smoothed_myelin = nib.load(
        subj_dir+'_hemi-'+surf_hemi+'_smoothed_myelinmap.shape.gii')
    smoothed_myelin = smoothed_myelin.agg_data()
    os.remove(
        subj_dir+'_hemi-'+surf_hemi+'_smoothed_myelinmap.shape.gii')
    
    return smoothed_myelin
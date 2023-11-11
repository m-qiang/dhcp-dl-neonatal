import torch
import numpy as np
import subprocess
import nibabel as nib
from utils.mesh import (
    area_weighted_smooth,
    vert_normal,
    mesh_area,
    adjacency_matrix,
    neighbor_matrix
)


def wb_inflate(vert, vert0, face,
               strength=1.0,
               n_iters=1,
               n_cycles=1,
               inflation_factor=1.0):
    """
    Cortical surface inflation.
    
    This function reimplements the method in the connectome
    workbench commandline. For originial code please see: 
    https://github.com/Washington-University/workbench/blob/
    master/src/Algorithms/AlgorithmSurfaceInflation.cxx

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - vert0: reference mesh vertices, (1,|V|,3) torch.Tensor
    - strength: strength of mesh smoothing [0,1]
    - n_iters: number of mesh smoothing iterations
    - n_cycles: number of inflation cycles
    - inflation_factor: factor of inflation
    
    Returns:
    - vert: inflated mesh vertices, (1,|V|,3) torch.Tensor
    """
    v_range = vert0.max(1)[0] - vert0.min(1)[0]
    # translate to center of mass
    v_center = vert.mean(1)
    vert = vert - v_center
    scale_factor = inflation_factor - 1.
    for c in range(n_cycles):
        # surface smooth
        vert = area_weighted_smooth(
            vert, face, strength=strength, n_iters=n_iters)
        # surface inflation
        radius = (vert / v_range).norm(dim=-1)
        scale = 1.0 + scale_factor * (1. - radius)
        vert = vert * scale.unsqueeze(-1)
    # translate back
    vert = vert + v_center
    return vert


def match_boundingbox(vert, vert0):
    """
    Match the bounding box of two surfaces.
    
    This function reimplements the method in the connectome
    workbench commandline. For originial code please see: 
    https://github.com/Washington-University/workbench/blob/
    master/src/Algorithms/AlgorithmSurfaceMatch.cxx

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - vert0: reference mesh vertices, (1,|V|,3) torch.Tensor

    Returns:
    - vert: rescaled mesh vertices, (1,|V|,3) torch.Tensor
    """
    vert = vert - vert.min(1)[0]  # translate min to origin
    scale = (vert0.max(1)[0] - vert0.min(1)[0])\
        / (vert.max(1)[0] - vert.min(1)[0])  # match the scale
    vert = vert * scale
    vert = vert +  vert0.min(1)[0]  # translate origin to the target min
    return vert


def generate_inflated_surfaces(vert, face, iter_scale=1.0):
    """
    Generate inflated and very inflated surfaces.
    
    This function reimplements the method in the connectome
    workbench commandline. For originial code please see: 
    https://github.com/Washington-University/workbench/blob/
    master/src/Algorithms/AlgorithmSurfaceGenerateInflated.cxx

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - iter_scale: scale of inflation iterations

    Returns:
    - inflated: inflated mesh vertices, (1,|V|,3) torch.Tensor
    - very_inflated: very inflated mesh vertices, (1,|V|,3) torch.Tensor
    """
    
    vert0 = vert.clone()
    low_smooth = wb_inflate(vert, vert0, face,
                            strength=0.2, n_iters=int(50*iter_scale),
                            n_cycles=1, inflation_factor=1.0)
    inflated = wb_inflate(low_smooth, vert0, face,
                          strength=1.0, n_iters=int(30*iter_scale),
                          n_cycles=2, inflation_factor=1.4)
    very_inflated = wb_inflate(inflated, vert0, face,
                               strength=1.0, n_iters=int(30*iter_scale),
                               n_cycles=4, inflation_factor=1.1)
    # match bounding box
    inflated = match_boundingbox(inflated, vert0)
    very_inflated = match_boundingbox(very_inflated, vert0)
    
    return inflated, very_inflated


def wb_generate_inflated_surfaces(
    save_dir,
    surf_hemi='left',
    iter_scale=1.0,
):
    """
    Generate inflated and very inflated surfaces by
    running connectome workbench commandline.
    
    Inputs:
    - save_dir: prefix of the directory for the subject
    - surf_hemi: ['left', 'right']
    - iter_scale: scale of inflation iterations

    Returns:
    - vert_inflated: inflated mesh vertices, (|V|,3) numpy.array
    - vert_vinflated: very inflated mesh vertices, (|V|,3) numpy.array
    """
    subprocess.run(
        'wb_command -surface-generate-inflated '+\
        save_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii '+\
        save_dir+'_hemi-'+surf_hemi+'_inflated.surf.gii '+\
        save_dir+'_hemi-'+surf_hemi+'_vinflated.surf.gii '+\
        '-iterations-scale '+str(iter_scale),
        shell=True)
    
    # load the vertices of inflated surfaces
    vert_inflated = nib.load(
        save_dir+'_hemi-'+surf_hemi+'_inflated.surf.gii'
    ).agg_data('pointset')
    vert_vinflated = nib.load(
        save_dir+'_hemi-'+surf_hemi+'_vinflated.surf.gii'
    ).agg_data('pointset')
    
    return vert_inflated, vert_vinflated



def mris_inflate(vert, face,
                 radius=2,
                 w_inflate=0.5,
                 w_distort=0.05,
                 eps_stop=0.015,
                 step_size=0.9,
                 n_steps=10,
                 n_levels=[0,6],
                 max_dx=1.0,
                 momentum=0.9,
                 track_sulcal_depth=False,
                 verbose=False):
    """
    Mimic FreeSurfer mris_inflate function.
    
    This function reimplements mirtk deform-mesh -inflate-brain.
    For originial code please see: 
    https://github.com/MIRTK/Deformable.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3), torch.LongTensor
    - radius: number of neighbors, int
    - w_inflate: weight for inflation loss/energy
    - w_distort: weight for distortion loss/energy
    - eps_stop: minimum rmse for the stop criteria
    - step_size: step size of the gradient descent
    - n_steps: number of steps for each integration level
    - n_levels: number of levels for gradient averaging [min_level, max_level]
    - max_dx: maximum displacement of the vertices
    - momentum: momentum for the gradient descent
    - track_sulcal_depth: if track sulcal depth
    - verbose: if report
    
    Returns:
    - vert_t: deformed/inflated mesh vertices, (1,|V|,3) torch.Tensor
    - sulcal_depth: sulcal depth, (|V|) numpy.array
    """
    
    # ------ initialization ------
    n_vert = vert.shape[1]
    # adjacency matrix
    A = adjacency_matrix(face)
    # connection matrix for n-neighbors (n=radius) 
    A_n = neighbor_matrix(face, radius)
    # edge list for adjacency matrix
    edge = A[0].coalesce().indices()
    # edge list for connection matrix
    edge_n = A_n[0].coalesce().indices()
    # initial distance for each edge
    dist_0 = (vert[0,edge_n[1]] -\
              vert[0,edge_n[0]]).norm(dim=-1, keepdim=True)
    # number of neighbors for each vertex
    degree = torch.sparse.sum(A, dim=-1).to_dense().unsqueeze(-1)  
    degree_n = torch.sparse.sum(A_n, dim=-1).to_dense().unsqueeze(-1)
    area_0 = mesh_area(vert, face)  # initial area
    
    # ------ configuration ------
    vert_t = vert.clone()
    h = step_size
    max_level = n_levels[-1]
    min_level = n_levels[0]
    # n levels of gradient averaging
    n_avgs = [2**i for i in range(max_level-2,
                                  np.maximum(min_level, 1)-2, -1)]
    if min_level == 0:
        n_avgs.append(0)   
    disp = 0
    if track_sulcal_depth:
        sulcal_depth = 0   # track sulcal depth

    # ------ optimization ------
    for level in range(len(n_avgs)):
        for t in range(n_steps):
            area_t = mesh_area(vert_t, face)   # compute current area
            scale_t = np.sqrt(area_0 / area_t)
            normal_t = vert_normal(vert_t, face)    # compute normal

            # ------ gradient of inflation energy ------
            grad_inflate = vert_t - A.bmm(vert_t) / degree
            # take out the average components of normal
            magnitude = (grad_inflate * normal_t).sum() / n_vert 
            grad_inflate = grad_inflate - magnitude * normal_t
            w_inflate_ = 2.0 * scale_t * w_inflate
            
            # ------ gradient of metric distortion ------
            # compute distance for each edge
            dist_t = (vert_t[0,edge_n[1]] -\
                      vert_t[0,edge_n[0]]).norm(dim=-1, keepdim=True)
            e_t = vert_t[0,edge_n[1]] - vert_t[0,edge_n[0]]   # direction vector
            
            # distortion=(dist_t-dist_0)^2, build sparse matrix
            distort = torch.sparse_coo_tensor(
                edge_n, (dist_t - dist_0/scale_t) / dist_t * e_t,
                (n_vert, n_vert, 3)).unsqueeze(0)
            grad_distort = torch.sparse.sum(
                distort, dim=-2).to_dense() / degree_n

            # take out the components of normal
            g_dot_n = (grad_distort * normal_t).sum(-1, keepdim=True)  # dot product
            grad_distort = grad_distort - g_dot_n * normal_t
            grad_distort = - grad_distort  # final gradient

            # gradient averaging (consider long geodesic distortion)
            for i in range(n_avgs[level]):
                grad_distort = (grad_distort+A.bmm(grad_distort))/(degree+1)
            w_distort_ = 2.0 * w_distort * np.sqrt(n_avgs[level])
            
            # ------ gradient discent ------
            grad_all = w_inflate_ * grad_inflate + w_distort_ * grad_distort
            
            disp = - h * grad_all + momentum * disp  # momentum
            disp_norm = (disp**2).sum(-1).sqrt()
            disp_scale = torch.min(torch.ones_like(disp_norm),
                                   max_dx / disp_norm).unsqueeze(-1)
            disp = disp * disp_scale    # scale the displacement
            vert_t = vert_t + disp    # update vertices
            if track_sulcal_depth:
                sulcal_depth += (-disp * normal_t).sum(-1)   # track sulcal depth
            
            # ------ inflation stop criteria ------
            e_t = vert_t[0,edge_n[1]] - vert_t[0,edge_n[0]]    # direction vector
            normal = vert_normal(vert_t, face)[0,edge_n[0]]
            dist2 = (e_t ** 2).sum(-1)
            dp = (normal * e_t).sum(-1)
            rmse = ((dp**2) / dist2).mean().sqrt().item()  # root mean suqared error
            
            if verbose:
                # inflation energy
                # compute squared distance
                dist2 =  ((vert_t[0,edge[0]] - vert_t[0,edge[1]])**2).sum(-1)
                # build an adjacency matrix
                Dist2 = torch.sparse_coo_tensor(
                    edge, dist2, (n_vert, n_vert)).unsqueeze(0)
                J_inflate = torch.sparse.sum(Dist2, dim=-1).to_dense().unsqueeze(-1)
                J_inflate = (J_inflate / degree).mean().item()
                # distortion energy
                dist_t = (vert_t[0,edge_n[1]] -\
                      vert_t[0,edge_n[0]]).norm(dim=-1, keepdim=True)
                delta2 = ((dist_t - dist_0/scale_t)**2).sum(-1)
                Delta2 = torch.sparse_coo_tensor(
                    edge_n, delta2, (n_vert, n_vert)).unsqueeze(0)
                J_distort = torch.sparse.sum(Delta2, dim=-1).to_dense().unsqueeze(-1)
                J_distort = (J_distort / degree_n).mean().item()
                J = w_inflate * J_inflate + w_distort * np.sqrt(n_avgs[level])* J_distort
                print("Level:{}, Iter:{}, RMSE:{}, Energy:{}, Inflation:{}, Distortion:{}."\
                      .format(level+1, t+1, np.round(rmse,5),
                              np.round(J, 5), np.round(J_inflate, 5),
                              np.round(J_distort, 5)))
            if rmse < eps_stop:
                break
        if verbose:
            print('')
        if rmse < eps_stop:
            break

    # ------ center and scale the output surface ------
#     vert_t = vert_t - vert_t.mean(1, keepdim=True)
#     area_t = mesh_area(vert_t, face)
#     scale_t = np.sqrt(area_0 / area_t)
#     vert_t = vert_t * scale_t
    vert_center = (vert_t.max(1, keepdim=True)[0] + \
                   vert_t.min(1, keepdim=True)[0])/2
    vert_t = vert_t - vert_center
    scale_t = np.sqrt(area_0 / area_t)
    vert_t = vert_t * scale_t
    
    if track_sulcal_depth:
        sulcal_depth = sulcal_depth[0].cpu().numpy() 
        # if zero mean for sulcal depth
        # sulcal_depth = sulcal_depth - sulcal_depth.mean()
        return vert_t, sulcal_depth
    else:
        return vert_t
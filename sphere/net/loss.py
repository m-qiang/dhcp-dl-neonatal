import torch
import numpy as np
from utils.mesh import face_area


def distortion(metric_sphere, metric_surf):
    """
    Compute metric distortion.

    Inputs:
    - metric_sphere: the metric of the (prediced) sphere, (1,|V|) torch.Tensor
    - metric_surf: the metric of the reference WM surface, (1,|V|) torch.Tensor
    
    Returns:
    - distort: the metric distortion (RMSD), torch.float
    """

    # beta = (metric_sphere * metric_surf).mean() /  (metric_surf**2).mean()
    # distort = ((metric_sphere - beta*metric_surf)**2).mean()
    beta = (metric_sphere * metric_surf).mean() /  (metric_sphere**2).mean()
    distort = ((beta*metric_sphere - metric_surf)**2).mean().sqrt()
    return distort


def edge_distortion(vert_sphere, vert_surf, edge):
    """
    Compute edge distortion.

    Inputs:
    - vert_sphere: the vertices of the (prediced) sphere, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - edge: the edge list of the mesh, (2,|E|) torch.LongTensor
    
    Returns:
    - edge distortion, torch.float
    """
    # compute edge length
    edge_len_sphere = (vert_sphere[:,edge[0]] -\
                       vert_sphere[:,edge[1]]).norm(dim=-1)
    edge_len_surf = (vert_surf[:,edge[0]] -\
                     vert_surf[:,edge[1]]).norm(dim=-1)
    return distortion(edge_len_sphere, edge_len_surf)


def area_distortion(vert_sphere, vert_surf, face):
    """
    Compute area distortion.

    Inputs:
    - vert_sphere: the vertices of the (prediced) sphere, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - face: the mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area distortion, torch.float
    """
    
    # compute face area
    area_sphere = face_area(vert_sphere, face)
    area_surf = face_area(vert_surf, face)
    return distortion(area_sphere, area_surf)


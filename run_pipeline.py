import os
import glob
import time
import argparse
import subprocess
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import ants
from ants.utils.bias_correction import n4_bias_field_correction

from seg.unet import UNet
from surface.net import SurfDeform
from sphere.net.sunet import SphereDeform
from sphere.net.utils import get_neighs_order
from sphere.net.loss import (
    edge_distortion,
    area_distortion)

from utils.mesh import (
    apply_affine_mat,
    taubin_smooth)

from utils.register import (
    registration,
    ants_trans_to_mat)

from utils.io import (
    Logging,
    save_numpy_to_nifti,
    save_gifti_surface,
    save_gifti_metric,
    create_wb_spec)

from utils.inflate import (
    generate_inflated_surfaces,
    wb_generate_inflated_surfaces)

from utils.metric import (
    metric_dilation,
    cortical_thickness,
    curvature,
    sulcal_depth,
    myelin_map,
    smooth_myelin_map)



# ============ load hyperparameters ============
parser = argparse.ArgumentParser(description="dHCP DL Neonatal Pipeline")
parser.add_argument('--in_dir', default='./in_dir/', type=str,
                    help='Diectory containing input images.')
parser.add_argument('--out_dir', default='./out_dir/', type=str,
                    help='Directory for saving the output of the pipeline.')
parser.add_argument('--T2', default='_T2w.nii.gz', type=str,
                    help='Suffix of T2 image file.')
parser.add_argument('--T1', default='_T1w.nii.gz', type=str,
                    help='Suffix of T1 image file.')
parser.add_argument('--device', default='cuda', type=str,
                    help='Device for running the pipeline: [cuda, cpu]')
parser.add_argument('--verbose', action='store_true',
                    help='Print debugging information.')
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
t2_suffix = args.T2
t1_suffix = args.T1
device = args.device
verbose = args.verbose
max_regist_iter = 5
min_regist_dice = 0.9


# ============ load nn model ============
# brain extraction
nn_seg_brain = UNet(
    C_in=1, C_hid=[16,32,32,32,32], C_out=1).to(device)
# ribbon segmentation
nn_seg_ribbon = UNet(
    C_in=1, C_hid=[16,32,64,128,128], C_out=1).to(device)

nn_seg_brain.load_state_dict(
    torch.load('./seg/model/model_seg_brain.pt', map_location=device))
nn_seg_ribbon.load_state_dict(
    torch.load('./seg/model/model_seg_ribbon.pt', map_location=device))

# surface reconstruction
nn_surf_left_wm = SurfDeform(
    C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0, device=device)
nn_surf_right_wm = SurfDeform(
    C_hid=[8,16,32,64,128,128], C_in=1, sigma=1.0, device=device)
nn_surf_left_pial = SurfDeform(
    C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0, device=device)
nn_surf_right_pial = SurfDeform(
    C_hid=[8,16,32,32,32,32], C_in=1, sigma=1.0, device=device)

nn_surf_left_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-left_wm.pt', map_location=device))
nn_surf_right_wm.load_state_dict(
    torch.load('./surface/model/model_hemi-right_wm.pt', map_location=device))
nn_surf_left_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-left_pial.pt', map_location=device))
nn_surf_right_pial.load_state_dict(
    torch.load('./surface/model/model_hemi-right_pial.pt', map_location=device))

# spherical projection
nn_sphere_left = SphereDeform(
    C_in=18, C_hid=[32, 64, 128, 128, 128], device=device)
nn_sphere_right = SphereDeform(
    C_in=18, C_hid=[32, 64, 128, 128, 128], device=device)

nn_sphere_left.load_state_dict(
    torch.load('./sphere/model/model_hemi-left_sphere.pt', map_location=device))
nn_sphere_right.load_state_dict(
    torch.load('./sphere/model/model_hemi-right_sphere.pt', map_location=device))


# ============ load image atlas ============
img_t2_atlas_ants = ants.image_read(
    './template/dhcp_week-40_template_T2w.nii.gz')
# both ants->nibabel and nibabel->ants need to reload the nifiti file
# so here simply load the image again
affine_t2_atlas = nib.load(
    './template/dhcp_week-40_template_T2w.nii.gz').affine


# ============ load input surface ============
surf_left_in = nib.load(
    './template/dhcp_week-40_hemi-left_init.surf.gii')
vert_left_in = surf_left_in.agg_data('pointset')
face_left_in = surf_left_in.agg_data('triangle')
vert_left_in = apply_affine_mat(
    vert_left_in, np.linalg.inv(affine_t2_atlas))
vert_left_in = vert_left_in - [64,0,0]
face_left_in = face_left_in[:,[2,1,0]]
vert_left_in = torch.Tensor(vert_left_in[None]).to(device)
face_left_in = torch.LongTensor(face_left_in[None]).to(device)

surf_right_in = nib.load(
    './template/dhcp_week-40_hemi-right_init.surf.gii')
vert_right_in = surf_right_in.agg_data('pointset')
face_right_in = surf_right_in.agg_data('triangle')
vert_right_in = apply_affine_mat(
    vert_right_in, np.linalg.inv(affine_t2_atlas))
face_right_in = face_right_in[:,[2,1,0]]
vert_right_in = torch.Tensor(vert_right_in[None]).to(device)
face_right_in = torch.LongTensor(face_right_in[None]).to(device)


# ============ load input sphere ============
sphere_left_in = nib.load(
    './template/dhcp_week-40_hemi-left_sphere.surf.gii')
vert_sphere_left_in = sphere_left_in.agg_data('pointset')
vert_sphere_left_in = torch.Tensor(vert_sphere_left_in[None]).to(device)

sphere_right_in = nib.load(
    './template/dhcp_week-40_hemi-right_sphere.surf.gii')
vert_sphere_right_in = sphere_right_in.agg_data('pointset')
vert_sphere_right_in = torch.Tensor(vert_sphere_right_in[None]).to(device)


# ============ load template sphere (160k) ============
sphere_160k = nib.load('./template/sphere_163842.surf.gii')
vert_sphere_160k = sphere_160k.agg_data('pointset')
face_160k = sphere_160k.agg_data('triangle')
vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
face_160k = torch.LongTensor(face_160k[None]).to(device)
neigh_order_160k = get_neighs_order()[0]  # neighbors


# ============ load pre-computed barycentric coordinates ============
# for sphere interpolation
barycentric_left = nib.load('./template/dhcp_week-40_hemi-left_barycentric.gii')
bc_coord_left = barycentric_left.agg_data('pointset')
face_left_id = barycentric_left.agg_data('triangle')

barycentric_right = nib.load('./template/dhcp_week-40_hemi-right_barycentric.gii')
bc_coord_right = barycentric_right.agg_data('pointset')
face_right_id = barycentric_right.agg_data('triangle')


# ============ dHCP DL-based neonatal pipeline ============
if __name__ == '__main__':
    subj_list = sorted(glob.glob(in_dir+'*'+t2_suffix))
    for subj_t2_dir in tqdm(subj_list):
        # extract subject id
        subj_id = subj_t2_dir.split('/')[-1][:-len(t2_suffix)]
        # check if T1 image exists
        if t1_suffix:
            subj_t1_dir = in_dir + subj_id + t1_suffix
        t1_exists = False
        if os.path.exists(subj_t1_dir):
            t1_exists = True

        # directory for saving output: out_dir/subj_id/
        subj_out_dir = out_dir + subj_id + '/'
        # create output directory
        if not os.path.exists(subj_out_dir):
            os.mkdir(subj_out_dir)
            # add subject id as prefix
        subj_out_dir = subj_out_dir + subj_id
        
        # initialize logger
        logger = Logging(subj_out_dir)
        # start processing
        logger.info('========================================')
        logger.info('Start processing subject: {}'.format(subj_id))
        t_start = time.time()
        
        
        # ============ Load Data ============
        logger.info('Load T2 image ...', end=' ')
        # load original T2 image
        img_t2_orig_ants = ants.image_read(subj_t2_dir)
        img_t2_orig = img_t2_orig_ants.numpy()

        # ants image produces inaccurate affine matrix
        # reload the nifti file to get the affine matrix
        img_t2_orig_nib = nib.load(subj_t2_dir)
        affine_t2_orig = img_t2_orig_nib.affine

        # args for converting numpy.array to ants image
        args_t2_orig_ants = (
            img_t2_orig_ants.origin,
            img_t2_orig_ants.spacing,
            img_t2_orig_ants.direction)
        logger.info('Done.')
        
        # load t1 image if exists
        if t1_exists:
            logger.info('Load T1 image ...', end=' ')
            # load original T1 image
            img_t1_orig_ants = ants.image_read(subj_t1_dir)
            img_t1_orig = img_t1_orig_ants.numpy()
            logger.info('Done.')

        
        # ============ brain extraction ============
        logger.info('----------------------------------------')
        logger.info('Brain extraction starts ...')
        t_brain_start = time.time()
        
        vol_t2_orig = torch.Tensor(img_t2_orig[None,None]).to(device)
        vol_t2_orig_down = F.interpolate(
            vol_t2_orig, size=[160,208,208], mode='trilinear')
        vol_in = (vol_t2_orig_down / vol_t2_orig_down.max()).float()
        with torch.no_grad():
            brain_mask_pred = torch.sigmoid(nn_seg_brain(vol_in))
            brain_mask_pred = F.interpolate(
                brain_mask_pred, size=vol_t2_orig.shape[2:], mode='trilinear')
        # threshold to binary mask
        brain_mask_orig = (brain_mask_pred[0,0]>0.5).float().cpu().numpy()
        
        # save brain mask
        save_numpy_to_nifti(
            brain_mask_orig, affine_t2_orig,
            subj_out_dir+'_brain_mask.nii.gz')
        
        t_brain_end = time.time()
        t_brain = t_brain_end - t_brain_start
        logger.info('Brain extraction ends. Runtime: {} sec.'.format(
            np.round(t_brain,4)))
        
        
        # ============ N4 bias field correction ============
        logger.info('----------------------------------------')
        logger.info('Bias field correction starts ...')
        t_bias_start = time.time()

        # N4 bias field correction
        brain_mask_ants = ants.from_numpy(
            brain_mask_orig, *args_t2_orig_ants)
        img_t2_restore_ants = n4_bias_field_correction(
            img_t2_orig_ants, brain_mask_ants,
            shrink_factor=4,
            convergence={"iters": [50, 50, 50], "tol": 0.001},
            spline_param=100,
            verbose=verbose)
        img_t2_restore = img_t2_restore_ants.numpy()
        # brain extracted and bias corrected image
        img_t2_proc = img_t2_restore * brain_mask_orig
        img_t2_proc_ants = ants.from_numpy(
            img_t2_proc, *args_t2_orig_ants)

        # save brain extracted and bias corrected T2w image
        save_numpy_to_nifti(
            img_t2_proc, affine_t2_orig,
            subj_out_dir+'_T2w_restore_brain.nii.gz')
        
        t_bias_end = time.time()
        t_bias = t_bias_end - t_bias_start
        logger.info('T2 bias field correction ends. Runtime: {} sec.'.format(
            np.round(t_bias,4)))
        
        
        # ============ T1-to-T2 ratio ============ 
        if t1_exists:
            logger.info('Compute T1/T2 ratio ...', end=' ')
            img_t1_t2_ratio = (
                img_t1_orig / (img_t2_orig+1e-12)).clip(0,100)
            img_t1_t2_ratio = img_t1_t2_ratio * brain_mask_orig
            save_numpy_to_nifti(
                img_t1_t2_ratio, affine_t2_orig,
                subj_out_dir+'_T1wDividedByT2w.nii.gz')
            logger.info('Done.')

        
        # ============ Cortical Ribbon Segmentation ============
        logger.info('----------------------------------------')
        logger.info('Cortical ribbon segmentation starts ...')
        t_ribbon_start = time.time()
        
        vol_t2_proc = torch.Tensor(img_t2_proc[None,None]).to(device)
        vol_t2_proc_down = F.interpolate(
            vol_t2_proc, size=[160,208,208], mode='trilinear')
        vol_in = (vol_t2_proc_down / vol_t2_proc_down.max()).float()
        with torch.no_grad():
            ribbon_pred = torch.sigmoid(nn_seg_ribbon(vol_in))
            ribbon_pred = F.interpolate(
                ribbon_pred, size=vol_t2_proc.shape[2:], mode='trilinear')
        # threshold to binary mask
        ribbon_orig = (ribbon_pred[0,0]>0.5).float().cpu().numpy()

        # save ribbon file
        save_numpy_to_nifti(
            ribbon_orig, affine_t2_orig, subj_out_dir+'_ribbon.nii.gz')

        t_ribbon_end = time.time()
        t_ribbon = t_ribbon_end - t_ribbon_start
        logger.info('Cortical ribbon segmentation ends. Runtime: {} sec.'.format(
            np.round(t_ribbon, 4)))
        
        
        # ============ Affine Registration ============
        logger.info('----------------------------------------')
        logger.info('Affine registration starts ...')
        t_align_start = time.time()

        # ants affine registration
        img_t2_align_ants, affine_t2_align, _, _, align_dice =\
        registration(
            img_move_ants=img_t2_proc_ants,
            img_fix_ants=img_t2_atlas_ants,
            affine_fix=affine_t2_atlas,
            out_prefix=subj_out_dir,
            max_iter=max_regist_iter,
            min_dice=min_regist_dice,
            verbose=verbose)
        
        # check dice score
        if align_dice >= min_regist_dice:
            logger.info('Dice after registration: {}'.format(align_dice))
        else:
            logger.info('Error! Affine registration failed!')
            logger.info('Expected Dice>{} after registraion, got Dice={}.'.format(
                min_regist_dice, align_dice))

        # args for converting numpy array to ants image
        args_t2_align_ants = (
            img_t2_align_ants.origin,
            img_t2_align_ants.spacing,
            img_t2_align_ants.direction)
        img_t2_align = img_t2_align_ants.numpy()
        # send to gpu for the following processing
        vol_t2_align = torch.Tensor(img_t2_align[None,None]).to(device)
        vol_t2_align = (vol_t2_align / vol_t2_align.max()).float()
        t_align_end = time.time()
        t_align = t_align_end - t_align_start
        logger.info('Affine registration ends. Runtime: {} sec.'.format(
            np.round(t_align, 4)))

        
        for surf_hemi in ['left', 'right']:
            
            # ============ Surface Reconstruction ============
            logger.info('----------------------------------------')
            logger.info('Surface reconstruction ({}) starts ...'.format(surf_hemi))
            t_surf_start = time.time()

            # set model, input vertices and faces
            if surf_hemi == 'left':
                nn_surf_wm = nn_surf_left_wm
                nn_surf_pial = nn_surf_left_pial
                # clip the left hemisphere
                vol_in = vol_t2_align[:,:,64:]
                vert_in = vert_left_in
                face_in = face_left_in
            elif surf_hemi == 'right':
                nn_surf_wm = nn_surf_right_wm
                nn_surf_pial = nn_surf_right_pial
                # clip the right hemisphere
                vol_in = vol_t2_align[:,:,:112]
                vert_in = vert_right_in
                face_in = face_right_in

            # wm and pial surfaces reconstruction
            with torch.no_grad():
                vert_wm = nn_surf_wm(vert_in, vol_in, n_steps=7)
                vert_wm = taubin_smooth(vert_wm, face_in, n_iters=5)
                vert_pial = nn_surf_pial(vert_wm, vol_in, n_steps=7)

            # torch.Tensor -> numpy.array
            vert_wm_align = vert_wm[0].cpu().numpy()
            vert_pial_align = vert_pial[0].cpu().numpy()
            face_align = face_in[0].cpu().numpy()

            # transform vertices to original space
            if surf_hemi == 'left':
                # pad the left hemisphere to full brain
                vert_wm_orig = vert_wm_align + [64,0,0]
                vert_pial_orig = vert_pial_align + [64,0,0]
            elif surf_hemi == 'right':
                vert_wm_orig = vert_wm_align.copy()
                vert_pial_orig = vert_pial_align.copy()
            vert_wm_orig = apply_affine_mat(
                vert_wm_orig, affine_t2_align)
            vert_pial_orig = apply_affine_mat(
                vert_pial_orig, affine_t2_align)
            face_orig = face_align[:,[2,1,0]]
            # midthickness surface
            vert_mid_orig = (vert_wm_orig + vert_pial_orig)/2

            # save as .surf.gii
            save_gifti_surface(
                vert_wm_orig, face_orig,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_wm.surf.gii',
                surf_hemi=surf_hemi, surf_type='wm')
            save_gifti_surface(
                vert_pial_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_pial.surf.gii',
                surf_hemi=surf_hemi, surf_type='pial')
            save_gifti_surface(
                vert_mid_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_midthickness.surf.gii',
                surf_hemi=surf_hemi, surf_type='midthickness')

            # send to gpu for the following processing
            vert_wm = torch.Tensor(vert_wm_orig).unsqueeze(0).to(device)
            vert_pial = torch.Tensor(vert_pial_orig).unsqueeze(0).to(device)
            vert_mid = torch.Tensor(vert_mid_orig).unsqueeze(0).to(device)
            face = torch.LongTensor(face_orig).unsqueeze(0).to(device)

            t_surf_end = time.time()
            t_surf = t_surf_end - t_surf_start
            logger.info('Surface reconstruction ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_surf, 4)))
            
            
            # ============ Surface Inflation ============
            logger.info('----------------------------------------')
            logger.info('Surface inflation ({}) starts ...'.format(surf_hemi))
            t_inflate_start = time.time()

            # create inflated and very_inflated surfaces
            # if device is cpu, use wb_command for inflation (faster)
            if device == 'cpu':
                vert_inflated_orig, vert_vinflated_orig = \
                wb_generate_inflated_surfaces(
                    subj_out_dir, surf_hemi, iter_scale=3.0)
            else:  # cuda acceleration
                vert_inflated, vert_vinflated = generate_inflated_surfaces(
                    vert_mid, face, iter_scale=3.0)
                vert_inflated_orig = vert_inflated[0].cpu().numpy()
                vert_vinflated_orig = vert_vinflated[0].cpu().numpy()

            # save as .surf.gii
            save_gifti_surface(
                vert_inflated_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_inflated.surf.gii',
                surf_hemi=surf_hemi, surf_type='inflated')
            save_gifti_surface(
                vert_vinflated_orig, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_vinflated.surf.gii',
                surf_hemi=surf_hemi, surf_type='vinflated')

            t_inflate_end = time.time()
            t_inflate = t_inflate_end - t_inflate_start
            logger.info('Surface inflation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_inflate, 4)))


            # ============ Spherical Mapping ============
            logger.info('----------------------------------------')
            logger.info('Spherical mapping ({}) starts ...'.format(surf_hemi))
            t_sphere_start = time.time()

            # set model, input vertices and faces
            if surf_hemi == 'left':
                nn_sphere = nn_sphere_left
                vert_sphere_in = vert_sphere_left_in
                bc_coord = bc_coord_left
                face_id = face_left_id
            elif surf_hemi == 'right':
                nn_sphere = nn_sphere_right
                vert_sphere_in = vert_sphere_right_in
                bc_coord = bc_coord_right
                face_id = face_right_id

            # interpolate to 160k template
            vert_wm_160k = (vert_wm_orig[face_id] * bc_coord[...,None]).sum(-2)
            vert_wm_160k = torch.Tensor(vert_wm_160k[None]).to(device)
            # input metric features
            neigh_wm_160k = vert_wm_160k[:, neigh_order_160k].reshape(
                vert_wm_160k.shape[0], vert_wm_160k.shape[1], 7, 3)[:,:,:-1]
            edge_wm_160k = (neigh_wm_160k - vert_wm_160k[:,:,None]).norm(dim=-1)
            area_wm_160k = 0.5*torch.norm(torch.cross(
                neigh_wm_160k[:,:,[0,1,2,3,4,5]] - vert_wm_160k[:,:,None],
                neigh_wm_160k[:,:,[1,2,3,4,5,0]] - vert_wm_160k[:,:,None]), dim=-1)
            # final input features
            feat_160k = torch.cat(
                [vert_sphere_160k, vert_wm_160k, edge_wm_160k, area_wm_160k], dim=-1)

            with torch.no_grad():
                vert_sphere = nn_sphere(
                    feat_160k, vert_sphere_in, n_steps=7)
                
            # compute metric distortion
            edge = torch.cat([
                face[0,:,[0,1]],
                face[0,:,[1,2]],
                face[0,:,[2,0]]], dim=0).T
            edge_distort = edge_distortion(vert_sphere, vert_wm, edge).item()
            area_distort = area_distortion(vert_sphere, vert_wm, face).item()
            logger.info('Edge distortion: {}mm'.format(np.round(edge_distort, 4)))
            logger.info('Area distortion: {}mm^2'.format(np.round(area_distort, 4)))
            
            # save as .surf.gii
            vert_sphere = vert_sphere[0].cpu().numpy()
            save_gifti_surface(
                vert_sphere, face_orig, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sphere.surf.gii',
                surf_hemi=surf_hemi, surf_type='sphere')

            t_sphere_end = time.time()
            t_sphere = t_sphere_end - t_sphere_start
            logger.info('Spherical mapping ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_sphere, 4)))


            # ============ Cortical Feature Estimation ============
            logger.info('----------------------------------------')
            logger.info('Feature estimation ({}) starts ...'.format(surf_hemi))
            t_feature_start = time.time()

            logger.info('Estimate cortical thickness ...', end=' ')
            thickness = cortical_thickness(vert_wm, vert_pial)
            thickness = metric_dilation(
                torch.Tensor(thickness[None,:,None]).to(device),
                face, n_iters=10)
            save_gifti_metric(
                metric=thickness,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_thickness.shape.gii',
                surf_hemi=surf_hemi, metric_type='thickness')
            logger.info('Done.')

            logger.info('Estimate curvature ...', end=' ')
            curv = curvature(vert_wm, face, smooth_iters=5)
            save_gifti_metric(
                metric=curv, 
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_curv.shape.gii',
                surf_hemi=surf_hemi, metric_type='curv')
            logger.info('Done.')

            logger.info('Estimate sulcal depth ...', end=' ')
            sulc = sulcal_depth(vert_wm, face, verbose=False)
            save_gifti_metric(
                metric=sulc,
                save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_sulc.shape.gii',
                surf_hemi=surf_hemi, metric_type='sulc')
            logger.info('Done.')

            
            # ============ myelin map estimation ============
            # estimate myelin map based on
            # t1-to-t2 ratio, midthickness surface, 
            # cortical thickness and cortical ribbon

            if t1_exists:
                logger.info('Estimate myelin map ...', end=' ')
                myelin = myelin_map(
                    subj_dir=subj_out_dir, surf_hemi=surf_hemi)
                # metric dilation
                myelin = metric_dilation(
                    torch.Tensor(myelin[None,:,None]).to(device),
                    face, n_iters=10)
                # save myelin map
                save_gifti_metric(
                    metric=myelin,
                    save_dir=subj_out_dir+'_hemi-'+surf_hemi+'_myelinmap.shape.gii',
                    surf_hemi=surf_hemi, metric_type='myelinmap')
                
                # smooth myelin map
                smoothed_myelin = smooth_myelin_map(
                    subj_dir=subj_out_dir, surf_hemi=surf_hemi)
                save_gifti_metric(
                    metric=smoothed_myelin, 
                    save_dir=subj_out_dir+'_hemi-'+surf_hemi+\
                             '_smoothed_myelinmap.shape.gii',
                    surf_hemi=surf_hemi,
                    metric_type='smoothed_myelinmap')
                logger.info('Done.')

            t_feature_end = time.time()
            t_feature = t_feature_end - t_feature_start
            logger.info('Feature estimation ({}) ends. Runtime: {} sec.'.format(
                surf_hemi, np.round(t_feature, 4)))

        logger.info('----------------------------------------')
        # clean temp data
        os.remove(subj_out_dir+'_rigid_0GenericAffine.mat')
        os.remove(subj_out_dir+'_affine_0GenericAffine.mat')
        # os.remove(subj_out_dir+'_ribbon.nii.gz')
        # if os.path.exists(subj_out_dir+'_T1wDividedByT2w.nii.gz'):
        #     os.remove(subj_out_dir+'_T1wDividedByT2w.nii.gz')
        # create .spec file for visualization
        create_wb_spec(subj_out_dir)
        t_end = time.time()
        logger.info('Finished. Total runtime: {} sec.'.format(
            np.round(t_end-t_start, 4)))
        logger.info('========================================')

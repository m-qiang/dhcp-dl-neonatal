import sys
import os
sys.path.append(os.getcwd())

import shutil
import nibabel as nib
import glob
import argparse
import ants
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.register import registration
from utils.io import (
    Logging,
    save_numpy_to_nifti,
    save_gifti_surface)
import pymeshlab



def split_data(orig_dir, save_dir, seed=12345):
    """split original dataset"""

    subj_list = sorted(glob.glob(orig_dir + 'sub-*/ses-*'))
    print("Number of all data:", len(subj_list))

    # ------ randomly split train/valid/test data ------ 
    np.random.seed(seed)
    subj_permute = np.random.permutation(len(subj_list))
    n_train = int(len(subj_list) * 0.6)
    n_valid = int(len(subj_list) * 0.1)
    n_test = len(subj_list) - n_train - n_valid
    print("Number of training data:", n_train)
    print("Number of validation data:", n_valid)
    print("Number of testing data:", n_test)

    train_list = subj_permute[:n_train]
    valid_list = subj_permute[n_train:n_train+n_valid]
    test_list = subj_permute[n_train+n_valid:]
    data_list = [train_list, valid_list, test_list]
    data_split = ['train', 'valid', 'test']

    for n in range(3):
        for i in data_list[n]:
            subid = subj_list[i].split('/')[-2]
            sesid = subj_list[i].split('/')[-1]
            subj_dir = save_dir+data_split[n]+'/'+subid+'_'+sesid
            if not os.path.exists(subj_dir):
                os.makedirs(subj_dir)
                
                
                
def remesh(vert, face, n_target=160000):
    # compute target edge length
    edge = np.concatenate([
        face[:,[0,1]], face[:,[1,2]], face[:,[2,0]]], axis=0).T
    edge_len = np.sqrt(((vert[edge[0]] - vert[edge[1]])**2).sum(-1)).mean()
    target_len = edge_len * np.sqrt(vert.shape[0] / n_target)

    # remesh the surface
    m = pymeshlab.Mesh(vert, face)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5, adaptive=False, checksurfdist=False,
        targetlen=pymeshlab.AbsoluteValue(target_len))

    vert_remesh = ms[0].vertex_matrix()
    face_remesh = ms[0].face_matrix()
    
    print("num of vertices before/after: {} / {}".format(
        vert.shape[0], vert_remesh.shape[0]))
    return vert_remesh, face_remesh
    
    

def process_data(orig_dir, save_dir):
    data_split = ['train', 'valid', 'test']

    img_fix_ants = ants.image_read(
        './template/dhcp_week-40_template_T2w.nii.gz')
    affine_fix = nib.load(
        './template/dhcp_week-40_template_T2w.nii.gz').affine
    img_fix = img_fix_ants
    min_regist_dice = 0.94
    logger = Logging(save_dir+'creat_dataset')

    for n in range(3):
        subj_list = sorted(os.listdir(save_dir+data_split[n]))
        for i in tqdm(range(len(subj_list))):
            subj_name = subj_list[i]
            subid, sesid = subj_name.split('_')
            subj_orig_dir = orig_dir+subid+'/'+sesid+'/anat/'+subid+'_'+sesid
            subj_save_dir = save_dir+data_split[n]+'/'+subid+'_'+sesid+'/'+subid+'_'+sesid
            logger.info(str(i), end=' ')
            logger.info(subid, end=' ')
            logger.info(sesid)

            # ------ load data ------
            # original t2 image
            img_orig_nib = nib.load(subj_orig_dir+'_T2w.nii.gz')
            affine_orig = img_orig_nib.affine
            # t2 image after bias correction
            img_restore_nib = nib.load(subj_orig_dir+'_desc-restore_T2w.nii.gz')
            # brainmask by bet
            brain_mask_nib = nib.load(subj_orig_dir+'_desc-brain_mask.nii.gz')
            img_orig = img_orig_nib.get_fdata()
            img_restore = img_restore_nib.get_fdata()
            brain_mask = brain_mask_nib.get_fdata()
            # brain-extracted bias-corrected brain MRI
            img_proc = img_restore * brain_mask
            # load tissue label
            tissue_label_nib = nib.load(subj_orig_dir+'_desc-drawem9_dseg.nii.gz')
            tissue_label = tissue_label_nib.get_fdata()
            ribbon = (tissue_label == 2).astype(np.float32)

            # ------ for segmentation ------
            # save original brain mask
            save_numpy_to_nifti(
                brain_mask.astype(np.float32), affine_orig,
                subj_save_dir+'_brain_mask.nii.gz')
            # save original cortical ribbon
            save_numpy_to_nifti(
                ribbon.astype(np.float32), affine_orig,
                subj_save_dir+'_ribbon.nii.gz')
            
            # ------ downsample image ------
            # downsample data
            img_orig_t = torch.Tensor(img_orig[None,None])
            img_orig_down_t = F.interpolate(
                img_orig_t, size=[160,208,208], mode='trilinear')
            img_orig_down = img_orig_down_t[0,0].numpy()
            img_proc_t = torch.Tensor(img_proc[None,None])
            img_proc_down_t = F.interpolate(
                img_proc_t, size=[160,208,208], mode='trilinear')
            img_proc_down = img_proc_down_t[0,0].numpy()
            
            # save downsampled images as training input
            save_numpy_to_nifti(
                img_orig_down.astype(np.float32), affine_orig,
                subj_save_dir+'_T2w_orig_down.nii.gz')
            save_numpy_to_nifti(
                img_proc_down.astype(np.float32), affine_orig,
                subj_save_dir+'_T2w_proc_down.nii.gz')
            
            # ------ affine registration ------
            # use restored brain-extracted image
            img_move_ants = ants.image_read(
                subj_orig_dir+'_desc-restore_T2w.nii.gz')
            img_move_ants = ants.from_numpy(
                img_proc,
                origin=img_move_ants.origin,
                spacing=img_move_ants.spacing,
                direction=img_move_ants.direction)
            
            # ants registration
            img_align_ants, affine_align, _, _, align_dice =\
            registration(
                img_move_ants, img_fix_ants, affine_fix,
                out_prefix=subj_save_dir)
            img_align = img_align_ants.numpy()

            # check dice score
            if align_dice >= min_regist_dice:
                logger.info('Dice after registration: {}'.format(align_dice))
            else:
                logger.info('Error! Affine registration failed!')
                logger.info('Expected Dice>{} after registraion, got Dice={}.'.format(
                    min_regist_dice, align_dice))
            os.remove(subj_save_dir+'_rigid_0GenericAffine.mat')
            os.remove(subj_save_dir+'_affine_0GenericAffine.mat')

            # save affinely aligned brain-extracted restored T2w image 
            save_numpy_to_nifti(
                img_align.astype(np.float32), affine_align,
                subj_save_dir+'_T2w_proc_affine.nii.gz')
            
            # ------ cortical surface remesh ------
            # copy surfaces
            for surf_hemi in ['left', 'right']:
                for surf_type in ['wm', 'pial']:
                    surf_orig_dir = subj_orig_dir+'_hemi-'+surf_hemi+'_'+surf_type+'.surf.gii'
                    surf_save_dir = subj_save_dir+'_hemi-'+surf_hemi+'_'+surf_type+'.surf.gii'
                    # shutil.copy(surf_orig_dir, surf_save_dir)
                    surf = nib.load(surf_orig_dir)
                    vert = surf.agg_data('pointset')
                    face = surf.agg_data('triangle')
                    vert_150k, face_150k = remesh(vert, face)
                    save_gifti_surface(
                        vert_150k, face_150k,
                        save_dir=surf_save_dir[:-9]+'_150k.surf.gii',
                        surf_hemi=surf_hemi, surf_type=surf_type)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Data Proprocessing")

    parser.add_argument('--orig_dir', default='./dhcp_dir/', type=str, help="directory of original dHCP dataset")
    parser.add_argument('--save_dir', default='./data/', type=str, help="directory for saving processed data")

    args = parser.parse_args()
    orig_dir = args.orig_dir
    save_dir = args.save_dir
    
    split_data(orig_dir, save_dir)
    process_data(orig_dir, save_dir)
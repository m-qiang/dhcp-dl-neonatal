import sys
import os
os.chdir('..')
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import nibabel as nib
import glob
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import chamfer_distance

from surface.net import SurfDeform

from utils.mesh import (
    apply_affine_mat,
    adjacent_faces,
    taubin_smooth,
    face_normal)


class SurfDataset(Dataset):
    """
    Dataset class for surface reconstruction
    """
    def __init__(self, args, data_split='train'):
        super(SurfDataset, self).__init__()
        
        # ------ load arguments ------ 
        surf_hemi = args.surf_hemi
        surf_type = args.surf_type
        device = args.device
        tag = args.tag
        sigma = args.sigma

        subj_list = sorted(glob.glob('./data/'+data_split+'/sub-*'))

        # ------ load template input ------ 
        img_temp = nib.load(
            './template/dhcp_week-40_template_T2w.nii.gz')
        surf_temp = nib.load(
            './template/dhcp_week-40_hemi-'+surf_hemi+'_init.surf.gii')
        affine_temp = img_temp.affine
        vert_temp = surf_temp.agg_data('pointset')
        face_temp = surf_temp.agg_data('triangle')
        vert_temp = apply_affine_mat(
            vert_temp, np.linalg.inv(affine_temp))
        face_temp = face_temp[:,[2,1,0]]
        
        # ------ load pre-trained model ------
        if surf_type == 'pial':
            nn_surf = SurfDeform(
                C_hid=[8,16,32,64,128,128], C_in=1,
                inshape=[112,224,160], sigma=sigma, device=device)
            model_dir = './surface/model/model_hemi-'+surf_hemi+'_wm.pt'
            nn_surf.load_state_dict(
                torch.load(model_dir, map_location=device))
        self.data_list = []
        
        for i in tqdm(range(len(subj_list))):
            subj_dir = subj_list[i]
            subj_id = subj_list[i].split('/')[-1]
            
            # ------ load input volume ------
            vol_in = nib.load(
                subj_dir+'/'+subj_id+'_T2w_proc_affine.nii.gz')
            affine_in = vol_in.affine
            vol_in = vol_in.get_fdata()
            vol_in = (vol_in / vol_in.max()).astype(np.float32)

            # clip left/right hemisphere
            if surf_hemi == 'left':
                vol_in = vol_in[None, 64:]
            elif surf_hemi == 'right':
                vol_in = vol_in[None, :112]
                
            # ------ load input surface ------
            # use init suface as input for wm surface recon
            vert_in = vert_temp.copy().astype(np.float32)
            face_in = face_temp.copy()
            if surf_hemi == 'left':
                vert_in[:,0] = vert_in[:,0] - 64
            # for pial surface, use predicted wm surface
            if surf_type == 'pial':
                vert_in = torch.Tensor(vert_in[None]).to(device)
                face_in = torch.LongTensor(face_in[None]).to(device)
                vol_in = torch.Tensor(vol_in[None]).to(device)
                with torch.no_grad():
                    vert_in = nn_surf(vert_in, vol_in, n_steps=7)
                    vert_in = taubin_smooth(vert_in, face_in, n_iters=5)
                vert_in = vert_in[0].cpu().numpy()
                face_in = face_in[0].cpu().numpy()
                vol_in = vol_in[0].cpu().numpy()

            # ------ load gt surface ------
            surf_gt = nib.load(
                subj_dir+'/'+subj_id+'_hemi-'+surf_hemi+'_'+surf_type+'_150k.surf.gii')
            vert_gt = surf_gt.agg_data('pointset')
            face_gt = surf_gt.agg_data('triangle')
            vert_gt = apply_affine_mat(
                vert_gt, np.linalg.inv(affine_in)).astype(np.float32)
            face_gt = face_gt[:,[2,1,0]]
            if surf_hemi == 'left':
                vert_gt[:,0] = vert_gt[:,0] - 64
            surf_data = (vol_in, vert_in, vert_gt, face_in, face_gt)
            self.data_list.append(surf_data)  # add to data list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        surf_data = self.data_list[i]
        return surf_data

    
def train_loop(args):
    # ------ load arguments ------ 
    surf_type = args.surf_type  # wm or pial
    surf_hemi = args.surf_hemi  # left or right
    tag = args.tag
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    sigma = args.sigma  # std for gaussian filter
    w_nc = args.w_nc  # weight for nc loss
    w_edge = args.w_edge  # weight for edge loss
    
    # start training logging
    logging.basicConfig(
        filename='./surface/ckpts/log_hemi-'+surf_hemi+'_'+\
        surf_type+'_'+tag+'.log', level=logging.INFO,
        format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = SurfDataset(args, data_split='train')
    validset = SurfDataset(args, data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # ------ pre-compute adjacency------
    surf_temp = nib.load(
        './template/dhcp_week-40_hemi-'+surf_hemi+'_init.surf.gii')
    face_temp = surf_temp.agg_data('triangle')[:,[2,1,0]]
    face_in = torch.LongTensor(face_temp[None]).to(device)
    # for normal consistency loss
    adj_faces = adjacent_faces(face_in)
    # for edge length loss
    edge_in = torch.cat([face_in[0,:,[0,1]],
                         face_in[0,:,[1,2]],
                         face_in[0,:,[2,0]]], dim=0).T

    # ------ initialize model ------ 
    logging.info("initalize model ...")
    if surf_type == 'wm':
        C_hid = [8,16,32,64,128,128]  # number of channels for each layer
    elif surf_type == 'pial':
        C_hid = [8,16,32,32,32,32]  # fewer params to avoid overfitting
    nn_surf = SurfDeform(
        C_hid=C_hid, C_in=1, inshape=[112,224,160],
        sigma=sigma, device=device)
    optimizer = optim.Adam(nn_surf.parameters(), lr=lr)

    # ------ training loop ------ 
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            vol_in, vert_in, vert_gt, face_in, face_gt = data
            vol_in = vol_in.to(device).float()
            vert_in = vert_in.to(device).float()
            face_in = face_in.to(device).long()
            vert_gt = vert_gt.to(device).float()
            face_gt = face_gt.to(device).long()
            optimizer.zero_grad()
            vert_pred = nn_surf(vert_in, vol_in, n_steps=7)

            # normal consistency loss
            normal = face_normal(vert_pred, face_in)  # face normal
            nc_loss = (1 - normal[:,adj_faces].prod(-2).sum(-1)).mean()
            # edge loss
            vert_i = vert_pred[:,edge_in[0]]
            vert_j = vert_pred[:,edge_in[1]]
            edge_loss = ((vert_i - vert_j)**2).sum(-1).mean() 
            # reconstruction loss
            recon_loss = chamfer_distance(vert_pred, vert_gt)[0]
            loss = recon_loss + w_nc*nc_loss + w_edge*edge_loss

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 10 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                recon_error = []
                for idx, data in enumerate(validloader):
                    vol_in, vert_in, vert_gt, face_in, face_gt = data
                    vol_in = vol_in.to(device).float()
                    vert_in = vert_in.to(device).float()
                    face_in = face_in.to(device).long()
                    vert_gt = vert_gt.to(device).float()
                    face_gt = face_gt.to(device).long()
                    
                    vert_pred = nn_surf(vert_in, vol_in, n_steps=7)
                    recon_loss = chamfer_distance(vert_pred, vert_gt)[0]
                    recon_error.append(recon_loss.item())

            logging.info('epoch:{}'.format(epoch))
            logging.info('recon error:{}'.format(np.mean(recon_error)))
            logging.info('-------------------------------------')
        
            # save model checkpoints
            torch.save(nn_surf.state_dict(),
                       './surface/ckpts/model_hemi-'+surf_hemi+'_'+\
                       surf_type+'_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Surface Recon")
    
    parser.add_argument('--surf_type', default='wm', type=str, help="[wm, pial]")
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda", type=str, help="[cuda, cpu]")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")

    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=100, type=int, help="number of training epochs")

    parser.add_argument('--sigma', default=1.0, type=float, help="standard deviation for gaussian smooth")
    parser.add_argument('--w_nc', default=3.0, type=float, help="weight for normal consistency loss")
    parser.add_argument('--w_edge', default=0.3, type=float, help="weight for edge length loss")

    args = parser.parse_args()
    
    train_loop(args)
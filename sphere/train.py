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

from surface.net import SurfDeform
from sphere.net.sunet import SphereDeform
from sphere.net.utils import get_neighs_order
from sphere.net.loss import(
    edge_distortion,
    area_distortion)
from utils.mesh import (
    face_area,
    apply_affine_mat,
    taubin_smooth)


class SphereDataset(Dataset):
    """
    Dataset class for surface reconstruction
    """
    def __init__(self, args, data_split='train'):
        super(SphereDataset, self).__init__()
        
        # ------ load arguments ------ 
        surf_hemi = args.surf_hemi
        device = args.device

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
        
        # ------ for sphere interpolation ------
        # load pre-computed barycentric coordinates
        barycentric = nib.load('./template/dhcp_week-40_hemi-'+surf_hemi+'_barycentric.gii')
        bc_coord = barycentric.agg_data('pointset')
        face_id = barycentric.agg_data('triangle')

        # ------ load pre-trained model ------
        nn_surf = SurfDeform(
            C_hid=[8,16,32,64,128,128], C_in=1,
            inshape=[112,224,160], sigma=1.0, device=device)
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

            # ------ predict wm surface ------
            vert_in = torch.Tensor(vert_in[None]).to(device)
            face_in = torch.LongTensor(face_in[None]).to(device)
            vol_in = torch.Tensor(vol_in[None]).to(device)
            with torch.no_grad():
                vert_wm = nn_surf(vert_in, vol_in, n_steps=7)
                vert_wm = taubin_smooth(vert_wm, face_in, n_iters=5)
            vert_wm_in = vert_wm[0].cpu().numpy()
            # transform to original space
            if surf_hemi == 'left':
                # pad the left hemisphere to full brain
                vert_wm_in = vert_wm_in + [64,0,0]
            vert_wm_in = apply_affine_mat(vert_wm_in, affine_in)
            # barycentric interpoalataion: resample to 160k template
            vert_wm_160k = (vert_wm_in[face_id] * bc_coord[...,None]).sum(-2)        
            sphere_data = (vert_wm_in, vert_wm_160k)
            self.data_list.append(sphere_data)  # add to data list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        sphere_data = self.data_list[i]
        return sphere_data


def train_loop(args):
    # ------ load arguments ------ 
    surf_hemi = args.surf_hemi  # left or right
    tag = args.tag
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    w_edge = args.w_edge  # weight for edge loss
    w_area = args.w_area  # weight for edge loss
    
    # start training logging
    logging.basicConfig(
        filename='./sphere/ckpts/log_hemi-'+surf_hemi+'_sphere_'+tag+'.log',
        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = SphereDataset(args, data_split='train')
    validset = SphereDataset(args, data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # ------ load input sphere ------
    sphere_in = nib.load(
        './template/dhcp_week-40_hemi-'+surf_hemi+'_sphere.surf.gii')
    vert_sphere_in = sphere_in.agg_data('pointset')
    face_in = sphere_in.agg_data('triangle')
    vert_sphere_in = torch.Tensor(vert_sphere_in[None]).to(device)
    face_in = torch.LongTensor(face_in[None]).to(device)
    edge_in = torch.cat([face_in[0,:,[0,1]],
                         face_in[0,:,[1,2]],
                         face_in[0,:,[2,0]]], dim=0).T

    # ------ load template sphere (160k) ------
    sphere_160k = nib.load('./template/sphere_163842.surf.gii')
    vert_sphere_160k = sphere_160k.agg_data('pointset')
    face_160k = sphere_160k.agg_data('triangle')
    vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
    face_160k = torch.LongTensor(face_160k[None]).to(device)
    neigh_order_160k = get_neighs_order()[0]
    
    # ------ load model ------
    nn_sphere = SphereDeform(
        C_in=18, C_hid=[32, 64, 128, 128, 128], device=device)
    optimizer = optim.Adam(nn_sphere.parameters(), lr=lr)

    # ------ training loop ------ 
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        for idx, data in enumerate(trainloader):
            vert_wm_in, vert_wm_160k = data
            # input coordinates
            vert_wm_in = vert_wm_in.to(device).float()  # (1,|V|,3)
            vert_wm_160k = vert_wm_160k.to(device).float()  # (1,|V|,3)
            
            # input metric features
            neigh_wm_160k = vert_wm_160k[:, neigh_order_160k].reshape(
                vert_wm_160k.shape[0], vert_wm_160k.shape[1], 7, 3)[:,:,:-1]
            # edge length (1,|V|,6)
            edge_wm_160k = (neigh_wm_160k - vert_wm_160k[:,:,None]).norm(dim=-1)
            # face area (1,|V|,6)
            area_wm_160k = 0.5*torch.norm(torch.cross(
                neigh_wm_160k[:,:,[0,1,2,3,4,5]] - vert_wm_160k[:,:,None],
                neigh_wm_160k[:,:,[1,2,3,4,5,0]] - vert_wm_160k[:,:,None]), dim=-1)
            
            # final input features (1,|V|,18)
            feat_160k = torch.cat(
                [vert_sphere_160k, vert_wm_160k, edge_wm_160k, area_wm_160k], dim=-1)

            optimizer.zero_grad()
            vert_sphere_pred = nn_sphere(
                feat_160k, vert_sphere_in, n_steps=7)

            # unsupervised metric distortion loss
            edge_loss = edge_distortion(
                vert_sphere_pred, vert_wm_in, edge_in)
            area_loss = area_distortion(
                vert_sphere_pred, vert_wm_in, face_in)
            loss = w_edge*edge_loss + w_area*area_loss

            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 10 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                edge_error = []
                area_error = []
                for idx, data in enumerate(validloader):
                    vert_wm_in, vert_wm_160k = data
                    vert_wm_in = vert_wm_in.to(device).float()
                    vert_wm_160k = vert_wm_160k.to(device).float()
                    
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

                    vert_sphere_pred = nn_sphere(
                        feat_160k, vert_sphere_in, n_steps=7)

                    # unsupervised metric distortion loss
                    edge_loss = edge_distortion(
                        vert_sphere_pred, vert_wm_in, edge_in)
                    area_loss = area_distortion(
                        vert_sphere_pred, vert_wm_in, face_in)
                    
                    edge_error.append(edge_loss.item())
                    area_error.append(area_loss.item())
                    
            logging.info('epoch:{}'.format(epoch))
            logging.info('edge error:{}'.format(np.mean(edge_error)))
            logging.info('area error:{}'.format(np.mean(area_error)))
            logging.info('-------------------------------------')

            # save model checkpoints
            torch.save(nn_sphere.state_dict(),
                       './sphere/ckpts/model_hemi-'+surf_hemi+'_sphere_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Spherical Mapping")
    
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda", type=str, help="[cuda, cpu]")
    parser.add_argument('--tag', default='0000', type=str, help="identity for experiments")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=100, type=int, help="number of training epochs")
    parser.add_argument('--w_edge', default=1.0, type=float, help="weight for edge distortion loss")
    parser.add_argument('--w_area', default=0.5, type=float, help="weight for area distortion loss")

    args = parser.parse_args()
    
    train_loop(args)
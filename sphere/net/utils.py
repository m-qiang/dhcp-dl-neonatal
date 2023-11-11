"""
Spherical U-Net

The code is retrieved from and modified based on:
https://github.com/zhaofenqiang/SphericalUNetPackage/
"""


import numpy as np
import scipy.io as sio


def get_neighs_order_(order_path):
    """
    Get indices of neighborhoods.
    """
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    return neigh_orders

def get_neighs_order():
    """
    Get indices of neighborhoods.
    """
    n_vert = [163842,40962,10242,2562,642,162,42] #,12]
    neigh_orders_list = []
    for n in n_vert:
        neigh_orders_n = get_neighs_order_(
            './sphere/net/neigh_indices/adj_mat_order_'+str(n)+'.mat')
        neigh_orders_list.append(neigh_orders_n)
    return neigh_orders_list


def get_upconv_index_(order_path):  
    """
    Get indices for upsampling.
    """
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index

def get_upconv_index():
    """
    Get indices for upsampling.
    """
    n_vert = [163842,40962,10242,2562,642,162,42]#,12]
    upconv_top_list = []
    upconv_down_list = []

    for n in n_vert:
        upconv_top_index, upconv_down_index = get_upconv_index_(
            './sphere/net/neigh_indices/adj_mat_order_'+str(n)+'.mat')
        upconv_top_list.append(upconv_top_index)
        upconv_down_list.append(upconv_down_index)
    return upconv_top_list, upconv_down_list
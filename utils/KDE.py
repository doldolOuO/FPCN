"""
Created on Thu Jun 24 21:10:05 2021

@author: admin
"""

import torch
import numpy as np
import math


def knn(x, k):
    x = x.permute(0, 2, 1)
    device = x.device
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def kernel_density_estimation(pts, kpoint = 32, is_norm = False):
    batch_size = pts.size(0)
    num_points = pts.size(1)
    device = pts.device
    point_indices = knn(pts, kpoint)
   
    # idx = torch.cat([batch_indices, point_indices], axis=3)
    # idx = idx.view(batch_size, num_points, kpoint, 2)

    #grouped_pts = torch.gather(pts, dim=0, index=idx.long())
    
    idx_base = torch.arange(0, batch_size, device=device).reshape(-1, 1, 1)*num_points

    idx = point_indices.long() + idx_base

    idx = idx.reshape(-1)
    grouped_pts = pts.reshape(batch_size*num_points, -1)[idx, :]
    grouped_pts = grouped_pts.reshape(batch_size, num_points, kpoint, -1)
    grouped_pts -= pts.unsqueeze(2).repeat(1,1,kpoint,1) # translation normalization

    # grouped_pts = grouped_pts.float()
    mean = torch.mean(grouped_pts, 2).unsqueeze(2).repeat(1, 1, kpoint, 1)
    group_mean = grouped_pts - mean
    std = torch.sum(torch.sum(torch.pow(group_mean, 2), -1), -1)/num_points
    
    sigma = 1.05*std*(num_points)**-0.2
    R = sigma**0.5
    R = R.reshape(batch_size, -1, 1, 1).repeat(1, 1, kpoint, grouped_pts.size(3))
    xRinv = torch.div(grouped_pts, R)
    quadform = torch.sum(torch.pow(xRinv, 2), axis = -1)
    logsqrtdetSigma = torch.log(sigma**0.5).unsqueeze(2).repeat(1, 1, kpoint)
    mvnpdf = torch.exp(-0.5 * quadform - logsqrtdetSigma - math.log(2 * 3.1415926) / 2)
    mvnpdf = torch.sum(mvnpdf, axis = 2, keepdims = True)
    scale = 1.0 / kpoint
    density = scale*mvnpdf
    
    if is_norm:
    #grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
        density_max = torch.max(density, axis = 1, keepdims = True)[0]
        density = torch.div(density, density_max)

    return density

def get_KDE_loss(coarse_ptc, gt, kpoint=32):
    device = gt.device
    den_c = kernel_density_estimation(coarse_ptc, kpoint, is_norm=False).squeeze()
    den_g = kernel_density_estimation(gt, kpoint, is_norm=False).squeeze()
    # indx= den_c.topk(k=nsample, dim=-1)[1]
    # idx_base = torch.arange(0, batch_size, device=device).reshape(-1, 1)*num_points

    # idx = indx.long() + idx_base

    # idx = idx.reshape(-1)
    # den_c = den_c.reshape(batch_size*num_points, -1)[idx, :]
    # den_c = den_c.reshape(batch_size, nsample)
    
    # den_g_mean = torch.mean(den_g, 1)[0].reshape(-1, 1)
    den_g_max = torch.max(den_g, 1)[0].reshape(-1, 1)
    out = torch.max(torch.zeros(1).to(device), den_c - den_g_max)
    
    loss = 0.5*torch.sum(out**2)
    return loss
    
if __name__ == '__main__':
    torch.manual_seed(1518)
    x = torch.rand((6, 4, 3))
    den = kernel_density_estimation(x, kpoint = 2, is_norm=False)
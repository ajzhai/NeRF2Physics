import os
import numpy as np
import open3d as o3d
import time
import torch
from PIL import Image
from utils import *


def get_bounding_box(pts, percentile=1.0, buffer=0.1):
    """Get the bounding box of a point cloud."""
    xyz1 = np.percentile(pts, percentile, axis=0)
    xyz2 = np.percentile(pts, 100 - percentile, axis=0)
    lwh = xyz2 - xyz1
    xyz1 -= buffer * lwh
    xyz2 += buffer * lwh
    return xyz1, xyz2


def get_grid_points(xyz1, xyz2, grid_cell_size):
    """Get grid points."""
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2
    x = np.arange(x1, x2, grid_cell_size)
    y = np.arange(y1, y2, grid_cell_size)
    z = np.arange(z1, z2, grid_cell_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return grid_pts



def project_3d_to_2d(pts, w2c, K, return_dists=False):
    """Project 3D points to 2D (nerfstudio format)."""
    pts = np.array(pts)
    K = np.hstack([K, np.zeros((3, 1))])
    pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts = np.dot(pts, w2c.T)
    pts[:, [1, 2]] *= -1
    if return_dists:
        dists = np.linalg.norm(pts[:, :3], axis=-1)
    pts = np.dot(pts, K.T)
    pts_2d = pts[:, :2] / pts[:, 2:]
    if return_dists:
        return pts_2d, dists
    return pts_2d

def project_3d_to_2d_torch(pts, w2c, K, return_dists=False):
    """Project 3D points to 2D (nerfstudio format)."""
    device = pts.device
    K = torch.cat([K, torch.zeros((3, 1), device=device)], 1)
    pts = torch.cat([pts, torch.ones((pts.shape[0], 1), device=device)], 1)
    pts = torch.matmul(pts, w2c.t())
    pts[:, [1, 2]] *= -1
    if return_dists:
        dists = torch.norm(pts[:, :3], dim=-1)
    pts = torch.matmul(pts, K.t())
    pts_2d = pts[:, :2] / pts[:, 2:]
    if return_dists:
        return pts_2d, dists
    return pts_2d


def depth_to_distance(depth, K):
    """Convert depth map to distance from camera."""
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    pts = np.stack([x, y, np.ones_like(x)], axis=1)
    pts = np.dot(pts, np.linalg.inv(K).T)
    pts *= depth[:, None]
    dists = np.linalg.norm(pts, axis=1)
    dists = dists.reshape(h, w)
    return dists


def depth_to_distance_torch(depth, K):
    """Convert depth map to distance from camera."""
    h, w = depth.shape
    x, y = torch.meshgrid(torch.arange(w), torch.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    pts = torch.stack([x, y, torch.ones_like(x)], dim=1).float().to(depth.device)
    pts = torch.matmul(pts, torch.inverse(K).t())
    pts *= depth[:, None]
    dists = torch.norm(pts, dim=1)
    dists = dists.reshape(h, w)
    return dists


def carve_numpy(pts, masks, depths, w2cs, K, dist_thr):
    n_imgs = len(masks)

    for i in range(n_imgs):
        h, w = masks[i].shape
        pts_2d, dists = project_3d_to_2d(pts, w2cs[i], K, return_dists=True)
        pts_2d = np.round(pts_2d).astype(np.int32)
        pts_2d = np.clip(pts_2d, 0, [w - 1, h - 1])

        observed_dists = depths[i] 

        is_in_mask = masks[i][pts_2d[:, 1], pts_2d[:, 0]]
        is_behind_depth = dists > observed_dists[pts_2d[:, 1], pts_2d[:, 0]] - dist_thr
        pts = pts[is_in_mask & is_behind_depth]

    return pts          

def carve_torch(pts, masks, depths, w2cs, K, dist_thr, mask_only=False):
    n_imgs = len(masks)

    with torch.no_grad():
        mask_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        depth_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        for i in range(n_imgs):
            h, w = masks[i].shape
            pts_2d, dists = project_3d_to_2d_torch(pts, w2cs[i], K, return_dists=True)
            pts_2d = torch.round(pts_2d).long().to(pts.device)
            pts_2d[:, 0] = torch.clamp(pts_2d[:, 0], 0, w - 1)
            pts_2d[:, 1] = torch.clamp(pts_2d[:, 1], 0, h - 1)

            observed_dists = depths[i] 

            is_in_mask = masks[i][pts_2d[:, 1], pts_2d[:, 0]]
            is_behind_depth = dists > observed_dists[pts_2d[:, 1], pts_2d[:, 0]] - dist_thr
            mask_votes[is_in_mask] += 1
            depth_votes[is_behind_depth] += 1
        if mask_only:
            pts = pts[mask_votes == n_imgs]
        else:
            pts = pts[(mask_votes == n_imgs) & (depth_votes == n_imgs)]

    return pts


def get_carved_pts(scene_dir, grid_cell_size_ns=1/512, dist_thr_ns=0.01, verbose=False, device='cuda'):
    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    t_file = os.path.join(scene_dir, 'transforms.json')
    img_dir = os.path.join(scene_dir, 'images')
    depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')

    pts = load_ns_point_cloud(pcd_file, dt_file)
    w2cs, K = parse_transforms_json(t_file, return_w2c=True)
    ns_transform, scale = parse_dataparser_transforms_json(dt_file)
    imgs, masks = load_images(img_dir, return_masks=True)
    depths = load_depths(depth_dir, Ks=None)

    xyz1, xyz2 = get_bounding_box(pts)
    grid_cell_size = grid_cell_size_ns / scale
    grid_pts = get_grid_points(xyz1, xyz2, grid_cell_size)
    dist_thr = dist_thr_ns / scale

    grid_pts = torch.from_numpy(grid_pts).float().to(device)
    masks = [torch.from_numpy(mask).to(device) for mask in masks]
    depths = [torch.from_numpy(depth).to(device) for depth in depths]
    w2cs = [torch.from_numpy(w2c).float().to(device) for w2c in w2cs]
    K = torch.from_numpy(K).float().to(device)

    carved = carve_torch(grid_pts, masks, depths, w2cs, K, dist_thr)
    if verbose:
        print('scene: %s, num. surface points: %d, num. carved points: %d, scale: %.4f' % 
              (scene_name, len(pts), len(carved), scale))
        
    return carved, grid_cell_size


if __name__ == '__main__':
    scene_dir = '/home/azhai/n2p/data/debug/B075X4J15G_ATVPDKIKX0DER'

    carved, grid_cell_size = get_carved_pts(scene_dir)
    carved = carved.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(carved)
    o3d.visualization.draw_geometries([pcd])
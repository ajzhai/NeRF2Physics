
import os
import numpy as np
import json
import open3d as o3d
import gzip
from PIL import Image


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


def parse_transforms_json(t_file, return_w2c=False, different_Ks=False):
    with open(t_file, 'rb') as f:
        transforms = json.load(f)

    if different_Ks:
        Ks = []
        for i in range(len(transforms['frames'])):
            K = np.array([
                [transforms['frames'][i]['fl_x'], 0, transforms['frames'][i]['cx']],
                [0, transforms['frames'][i]['fl_y'], transforms['frames'][i]['cy']],
                [0, 0, 1],
            ])
            Ks.append(K)
        K = Ks
    else:
        K = np.array([
            [transforms['fl_x'], 0, transforms['cx']],
            [0, transforms['fl_y'], transforms['cy']],
            [0, 0, 1],
        ])

    n_frames = len(transforms['frames'])
    c2ws = [np.array(transforms['frames'][i]['transform_matrix']) for i in range(n_frames)]
    if return_w2c:
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        return w2cs, K
    return c2ws, K


def parse_dataparser_transforms_json(dt_file):
    with open(dt_file, "r") as fr:
        dataparser_transforms = json.load(fr)

    ns_transform = np.asarray(dataparser_transforms["transform"])
    scale = dataparser_transforms["scale"]
    return ns_transform, scale


def load_ns_point_cloud(pcd_file, dt_file, ds_size=0.01, viz=False):
    pcd = o3d.io.read_point_cloud(pcd_file)
    if ds_size is not None:
        pcd = pcd.voxel_down_sample(ds_size)

    ns_transform, scale = parse_dataparser_transforms_json(dt_file)
    ns_transform = np.concatenate([ns_transform, np.array([[0, 0, 0, 1/scale]])], 0)
    inv_ns_transform = np.linalg.inv(ns_transform)

    # use open3d to scale and transform
    pcd.transform(inv_ns_transform)

    pts = np.asarray(pcd.points)

    if viz:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([cf, pcd])
    return pts


def load_images(img_dir, bg_change=255, return_masks=False):
    img_files = os.listdir(img_dir)
    img_files.sort()
    imgs = []
    masks = []
    for img_file in img_files:
        # load RGBA image
        img = np.array(Image.open(os.path.join(img_dir, img_file)))
        if return_masks or bg_change is not None:
            mask = img[:, :, 3] > 0
            if bg_change is not None:
                img[~mask] = bg_change
            masks.append(mask)
        imgs.append(img[:, :, :3])
        
    if return_masks:
        return imgs, masks
    return imgs


def load_depths(depth_dir, Ks):
    depth_files = os.listdir(depth_dir)
    depth_files.sort()
    depths = []
    for i, depth_file in enumerate(depth_files):
        # load npy.gz depth file
        with gzip.open(os.path.join(depth_dir, depth_file), 'rb') as f:
            dist = np.load(f)[:, :, 0]
        if Ks is not None:
            depth = distance_to_depth(dist, Ks[i])
        else:
            depth = dist
        depths.append(depth)
    return depths


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


def distance_to_depth(dists, K):
    """Convert distance map to depth map."""
    h, w = dists.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    pts = np.stack([x, y, np.ones_like(x)], axis=1)
    pts = np.dot(pts, np.linalg.inv(K).T)
    divisor = np.linalg.norm(pts, axis=1)
    divisor = divisor.reshape(h, w)
    depth = dists / divisor
    return depth


def get_last_file_in_folder(folder):
    files = os.listdir(folder)
    return os.path.join(folder, sorted(files, reverse=True)[0])


def get_scenes_list(args):
    if args.split != 'all':
        with open(os.path.join(args.data_dir, 'splits.json'), 'r') as f:
            splits = json.load(f)
        if args.split == 'train+val':
            scenes = splits['train'] + splits['val']
        else:
            scenes = splits[args.split]
    else:
        scenes = sorted(os.listdir(os.path.join(args.data_dir, 'scenes')))

    if args.end_idx != -1:
        scenes = scenes[args.start_idx:args.end_idx]
    else:
        scenes = scenes[args.start_idx:]
    return scenes


def unproject_point(pt_2d, depth, c2w, K):
    """Unproject a single point from 2D to 3D (nerfstudio format)."""
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x = (pt_2d[0] - cx) / fx
    y = (pt_2d[1] - cy) / fy
    pt_3d = np.array([x, -y, -1])
    pt_3d *= depth[pt_2d[1], pt_2d[0]]
    pt_3d = np.concatenate([pt_3d, np.ones((1,))], axis=0)
    pt_3d = np.dot(c2w, pt_3d)
    pt_3d = pt_3d[:3]
    return pt_3d

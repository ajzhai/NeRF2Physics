import os
import numpy as np
import json
import torch
import open_clip
from PIL import Image

from utils import *
from arguments import get_args


CLIP_BACKBONE = 'ViT-B-16'
CLIP_CHECKPOINT = 'datacomp_xl_s13b_b90k'
CLIP_INPUT_SIZE = 224
CLIP_OUTPUT_SIZE = 512


def get_patch_features(pts, imgs, depths, w2cs, K, model, preprocess_fn, occ_thr,
                       patch_size=56, batch_size=8, device='cuda'):
    n_imgs = len(imgs)
    n_pts = len(pts)

    patch_features = torch.zeros(n_imgs, n_pts, CLIP_OUTPUT_SIZE, device=device, requires_grad=False)
    is_visible = torch.zeros(n_imgs, n_pts, device=device, dtype=torch.bool, requires_grad=False)
    half_patch_size = patch_size // 2

    K = np.array(K)
    with torch.no_grad(), torch.cuda.amp.autocast():
        model.to(device)

        for i in range(n_imgs):
            h, w, c = imgs[i].shape
            if len(K.shape) == 3:
                curr_K = K[i]
            else:
                curr_K = K
            pts_2d, dists = project_3d_to_2d(pts, w2cs[i], curr_K, return_dists=True)
            pts_2d = np.round(pts_2d).astype(np.int32)

            observed_dists = depths[i]
        
            # loop through pts in batches
            for batch_start in range(0, n_pts, batch_size):
                curr_batch_size = min(batch_size, n_pts - batch_start)
                batch_patches = torch.zeros(curr_batch_size, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, device=device)
                
                for j in range(curr_batch_size):
                    x, y = pts_2d[batch_start + j]

                    if x >= half_patch_size and x < w - half_patch_size and \
                       y >= half_patch_size and y < h - half_patch_size:
                        is_occluded = dists[batch_start + j] > observed_dists[y, x] + occ_thr
                        if not is_occluded:
                            patch = imgs[i][y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size]
                            patch = Image.fromarray(patch)
                            
                            patch = preprocess_fn(patch).unsqueeze(0).to(device)
                            batch_patches[j] = patch
                            is_visible[i, batch_start + j] = True

                if is_visible[i, batch_start:batch_start + batch_size].any():
                    patch_features[i, batch_start:batch_start + curr_batch_size] = model.encode_image(batch_patches)

    return patch_features, is_visible


def process_scene(args, scene_dir, model, preprocess_fn):
    
    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    t_file = os.path.join(scene_dir, 'transforms.json')
    img_dir = os.path.join(scene_dir, 'images')
    depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')

    pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.feature_voxel_size)
    w2cs, K = parse_transforms_json(t_file, return_w2c=True, different_Ks=args.different_Ks)
    ns_transform, scale = parse_dataparser_transforms_json(dt_file)
    imgs = load_images(img_dir)
    depths = load_depths(depth_dir, Ks=None)

    print('scene: %s, points: %d, scale: %.4f' % (scene_name, len(pts), scale))

    with torch.no_grad():
        occ_thr = args.occ_thr * scale
        patch_features, is_visible = get_patch_features(pts, imgs, depths, w2cs, K, 
                                                        model, preprocess_fn, 
                                                        occ_thr, patch_size=args.patch_size, batch_size=args.batch_size, 
                                                        device=args.device)
        
    out_dir = os.path.join(scene_dir, 'features')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(patch_features, os.path.join(out_dir, 'patch_features_%s.pt' % args.feature_save_name))
    torch.save(is_visible, os.path.join(out_dir, 'is_visible_%s.pt' % args.feature_save_name))
    with open(os.path.join(out_dir, 'voxel_size_%s.json' % args.feature_save_name), 'w') as f:
        json.dump({'voxel_size': args.feature_voxel_size}, f, indent=4)

    return pts, patch_features, is_visible
    
    
if __name__ == '__main__':   

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    model.to(args.device)

    for j, scene in enumerate(scenes):
        pts, patch_features, is_visible = process_scene(args, os.path.join(scenes_dir, scene), model, preprocess)


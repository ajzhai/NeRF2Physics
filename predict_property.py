import os
import json
import torch
import open_clip
import numpy as np
import open3d as o3d
import matplotlib as mpl

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from gpt_inference import parse_material_list, parse_material_hardness
from carving import get_carved_pts
from utils import load_ns_point_cloud, parse_dataparser_transforms_json, get_last_file_in_folder, get_scenes_list
from arguments import get_args


@torch.no_grad()
def get_text_features(texts, clip_model, clip_tokenizer, prefix='', suffix='', device='cuda'):
    """Get CLIP text features, optionally with a fixed prefix and suffix."""
    extended_texts = [prefix + text + suffix for text in texts]
    tokenized = clip_tokenizer(extended_texts).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features


@torch.no_grad()
def get_agg_patch_features(patch_features, is_visible):
    """Get aggregated patch features by averaging over visible patches."""
    n_visible = is_visible.sum(0)
    is_valid = n_visible > 0

    visible_patch_features = patch_features * is_visible.unsqueeze(-1)
    avg_visible_patch_features = visible_patch_features.sum(0) / n_visible.unsqueeze(-1)
    avg_visible_patch_features = avg_visible_patch_features / avg_visible_patch_features.norm(dim=1, keepdim=True)
    return avg_visible_patch_features[is_valid], is_valid


@torch.no_grad()
def get_interpolated_values(source_pts, source_vals, inner_pts, batch_size=2048, k=1):
    """Interpolate values by k nearest neighbor."""
    n_inner = len(inner_pts)
    inner_vals = torch.zeros(n_inner, source_vals.shape[1], device=inner_pts.device)
    for batch_start in range(0, n_inner, batch_size):
        curr_batch_size = min(batch_size, n_inner - batch_start)
        curr_inner_pts = inner_pts[batch_start:batch_start + curr_batch_size]

        dists = torch.cdist(curr_inner_pts, source_pts)
        _, idxs = torch.topk(dists, k=k, dim=1, largest=False)
        curr_inner_vals = source_vals[idxs].mean(1)

        inner_vals[batch_start:batch_start + curr_batch_size] = curr_inner_vals
    return inner_vals


@torch.no_grad()
def predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer):
    """Predict the volume integral of a physical property (e.g. for mass). Returns a [low, high] range."""

    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    info_file = os.path.join(scene_dir, '%s.json' % args.mats_load_name)

    with open(info_file, 'r') as f:
        info = json.load(f)

    # loading source point info
    with open(os.path.join(scene_dir, 'features', 'voxel_size_%s.json' % args.feature_load_name), 'r') as f:
        feature_voxel_size = json.load(f)['voxel_size']
    pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=feature_voxel_size)
    source_pts = torch.Tensor(pts).to(args.device)
    patch_features = torch.load(os.path.join(scene_dir, 'features', 'patch_features_%s.pt' % args.feature_load_name))
    is_visible = torch.load(os.path.join(scene_dir, 'features', 'is_visible_%s.pt' % args.feature_load_name))

    # preparing material info
    mat_val_list = info['candidate_materials_%s' % args.property_name]
    mat_names, mat_vals = parse_material_list(mat_val_list)
    mat_vals = torch.Tensor(mat_vals).to(args.device)
    mat_tn_list = info['thickness']
    mat_names, mat_tns = parse_material_list(mat_tn_list)
    mat_tns = torch.Tensor(mat_tns).to(args.device) / 100  # cm to m

    # predictions on source points
    text_features = get_text_features(mat_names, clip_model, clip_tokenizer, device=args.device)
    agg_patch_features, is_valid = get_agg_patch_features(patch_features, is_visible)
    source_pts = source_pts[is_valid]

    similarities = agg_patch_features @ text_features.T

    source_pred_probs = torch.softmax(similarities / args.temperature, dim=1)
    source_pred_mat_idxs = similarities.argmax(1)
    source_pred_vals = source_pred_probs @ mat_vals

    # volume integration
    ns_transform, scale = parse_dataparser_transforms_json(dt_file)
    surface_cell_size = args.sample_voxel_size / scale
    mat_cell_volumes = surface_cell_size**2 * mat_tns
    mat_cell_products = mat_vals * mat_cell_volumes

    if args.volume_method == 'thickness':
        dense_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.sample_voxel_size)
        dense_pts = torch.Tensor(dense_pts).to(args.device)
        
        dense_pred_probs = get_interpolated_values(source_pts, source_pred_probs, dense_pts, batch_size=2048, k=1)
        dense_pred_products = dense_pred_probs @ mat_cell_products
        total_pred_val = (dense_pred_products).sum(0)

        carved, grid_cell_size = get_carved_pts(scene_dir, dist_thr_ns=0.05)
        bound_volume = grid_cell_size ** 3 * len(carved)
        total_volume = (dense_pred_probs @ mat_cell_volumes).max(1)[0].sum(0)
        if total_volume > bound_volume:
            total_pred_val *= bound_volume / total_volume
        total_pred_val *= args.correction_factor

    elif args.volume_method == 'carving':
        carved, grid_cell_size = get_carved_pts(scene_dir)
        carved_pred_probs = get_interpolated_values(source_pts, source_pred_probs, carved, batch_size=2048, k=1)
        carved_pred_vals = carved_pred_probs @ mat_vals
        grid_cell_volume = grid_cell_size ** 3
        total_pred_val = carved_pred_vals.sum(0) * grid_cell_volume * args.correction_factor

        dense_pts = carved
        dense_pred_probs = carved_pred_probs


    else:
        raise NotImplementedError

    print('-' * 50)
    print('scene:', scene_name)
    print('-' * 50)
    print('num. dense points:', len(dense_pts))
    print('caption:', info['caption'])
    print('candidate materials:')
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f kg/m^3, %5.1f -%5.1f cm' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1],
               mat_tns[mat_i][0] * 100, mat_tns[mat_i][1] * 100))

    print('surface cell size: %.4f cm' % (surface_cell_size * 100))
    print('predicted total mass: [%.4f - %.4f kg]' % (total_pred_val[0], total_pred_val[1]))

    if args.show_mat_seg:
        # Visualize material segmentation in open3d
        cmap = mpl.colormaps['tab10']
        mat_colors = [cmap(i)[:3] for i in range(len(mat_names))]
        dense_pred_colors = np.array([mat_colors[i] for i in dense_pred_probs.argmax(1)])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(dense_pred_colors)
        o3d.visualization.draw_geometries([pcd])

    return total_pred_val.tolist()


@torch.no_grad()
def predict_physical_property_query(args, query_pts, scene_dir, clip_model, clip_tokenizer, return_all=False):
    """
    Predict a physical property at given array of 3D query points. query_pts can be set to 'grid'
    instead to automatically generate a grid of query points from source points. If return_all=True, 
    returns various intermediate results. Otherwise, returns [low, high] range for each query point.
    """

    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    info_file = os.path.join(scene_dir, '%s.json' % args.mats_load_name)

    with open(info_file, 'r') as f:
        info = json.load(f)

    # loading source point info
    with open(os.path.join(scene_dir, 'features', 'voxel_size_%s.json' % args.feature_load_name), 'r') as f:
        feature_voxel_size = json.load(f)['voxel_size']
    pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=feature_voxel_size)
    source_pts = torch.Tensor(pts).to(args.device)
    patch_features = torch.load(os.path.join(scene_dir, 'features', 'patch_features_%s.pt' % args.feature_load_name))
    is_visible = torch.load(os.path.join(scene_dir, 'features', 'is_visible_%s.pt' % args.feature_load_name))

    # preparing material info
    mat_val_list = info['candidate_materials_%s' % args.property_name]
    if args.property_name == 'hardness':
        mat_names, mat_vals = parse_material_hardness(mat_val_list)
    else:
        mat_names, mat_vals = parse_material_list(mat_val_list)
    mat_vals = torch.Tensor(mat_vals).to(args.device)

    # predictions on source points
    text_features = get_text_features(mat_names, clip_model, clip_tokenizer, device=args.device)
    agg_patch_features, is_valid = get_agg_patch_features(patch_features, is_visible)
    source_pts = source_pts[is_valid]

    similarities = agg_patch_features @ text_features.T

    source_pred_probs = torch.softmax(similarities / args.temperature, dim=1)
    source_pred_mat_idxs = similarities.argmax(1)
    source_pred_vals = source_pred_probs @ mat_vals

    if query_pts == 'grid':
        query_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.sample_voxel_size)
        query_pts = torch.Tensor(query_pts).to(args.device)
    query_pred_probs = get_interpolated_values(source_pts, source_pred_probs, query_pts, batch_size=2048, k=1)
    query_pred_vals = query_pred_probs @ mat_vals

    print('-' * 50)
    print('scene:', scene_name)
    print('-' * 50)
    print('num. query points:', len(query_pts))
    print('caption:', info['caption'])
    print('candidate materials (%s):' % args.property_name)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))

    if args.show_mat_seg:
        # Visualize material segmentation in open3d
        cmap = mpl.colormaps['tab10']
        mat_colors = [cmap(i)[:3] for i in range(len(mat_names))]
        query_pred_colors = np.array([mat_colors[i] for i in query_pred_probs.argmax(1)])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(query_pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(query_pred_colors)
        o3d.visualization.draw_geometries([pcd])

    if return_all:
        query_features = get_interpolated_values(source_pts, agg_patch_features, query_pts, batch_size=2048, k=1)
        query_similarities = get_interpolated_values(source_pts, similarities, query_pts, batch_size=2048, k=1)
        return {
            'query_pred_probs': query_pred_probs.cpu().numpy(),
            'query_pred_vals': query_pred_vals.cpu().numpy(),
            'query_features': query_features.cpu().numpy(),
            'query_similarities': query_similarities.cpu().numpy(),
            'source_pts': source_pts.cpu().numpy(),
            'mat_names': mat_names,
        }
    return query_pred_vals.cpu().numpy()


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    preds = {}
    for j, scene in enumerate(scenes): 
        scene_dir = os.path.join(scenes_dir, scene)
        if args.prediction_mode == 'integral':
            pred = predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer)
        elif args.prediction_mode == 'grid':
            pred = predict_physical_property_query(args, 'grid', scene_dir, clip_model, clip_tokenizer)
        else:  # use predict_physical_property_query() to query points however you want!
            raise NotImplementedError
        preds[scene] = pred
    
    if args.prediction_mode == 'integral' and args.save_preds:
        os.makedirs('preds', exist_ok=True)
        with open(os.path.join('preds', 'preds_%s.json' % args.preds_save_name), 'w') as f:
            json.dump(preds, f, indent=4)

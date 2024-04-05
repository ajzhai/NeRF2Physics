import argparse


def get_args():
    parser = argparse.ArgumentParser(description='NeRF2Physics')
    
    # General arguments
    parser.add_argument('--data_dir', type=str, default="./data/abo_500/",
                        help='path to data (default: ./data/abo_500/)')
    parser.add_argument('--split', type=str, default="all",
                        help='dataset split, either train, val, train+val, test, or all (default: all)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='starting scene index, useful for evaluating only a few scenes (default: 0)')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='ending scene index, useful for evaluating only a few scenes (default: -1)')
    parser.add_argument('--different_Ks', type=int, default=0,
                        help='whether data has cameras with different intrinsic matrices (default: 0)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device for torch (default: cuda)')

    # NeRF training
    parser.add_argument('--training_iters', type=int, default=20000,
                        help='number of training iterations (default: 20000)')
    parser.add_argument('--near_plane', type=float, default=0.4,
                        help='near plane for ray sampling (default: 0.4)')
    parser.add_argument('--far_plane', type=float, default=6.0,
                        help='far plane for ray sampling (default: 6.0)')
    parser.add_argument('--vis_mode', type=str, default='wandb',
                        help='nerfstudio visualization mode (default: wandb)')
    parser.add_argument('--project_name', type=str, default='NeRF2Physics',
                        help='project name used by wandb (default: NeRF2Physics)')
    
    # NeRF point cloud
    parser.add_argument('--num_points', type=int, default=100000,
                        help='number of points for point cloud (default: 100000)')
    parser.add_argument('--bbox_size', type=float, default=1.0,
                        help='bounding box (cube) size, relative to scaled scene (default: 1.0)')

    # CLIP feature fusion
    parser.add_argument('--patch_size', type=int, default=56,
                        help='patch size (default: 56)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument('--feature_voxel_size', type=int, default=0.01,
                        help='voxel downsampling size for features, relative to scaled scene (default: 0.01)')
    parser.add_argument('--feature_save_name', type=str, default="ps56",
                        help='feature save name (default: ps56)')
    parser.add_argument('--occ_thr', type=float, default=0.01,
                        help='occlusion threshold, relative to scaled scene (default: 0.01)')
    
    # Captioning and view selection
    parser.add_argument('--blip2_model_dir', type=str, default="./blip2-flan-t5-xl",
                        help='path to BLIP2 model directory (default: ./blip2-flan-t5-xl)')
    parser.add_argument('--mask_area_percentile', type=float, default=0.75,
                        help='mask area percentile for canonical view (default: 0.75)')
    parser.add_argument('--caption_save_name', type=str, default="info_new",
                        help='caption save name (default: info_new)')
    
    # Material proposal
    parser.add_argument('--caption_load_name', type=str, default="info_new",
                        help='name of saved caption to load (default: info_new)')
    parser.add_argument('--property_name', type=str, default="density",
                        help='property to predict (default: density)')
    parser.add_argument('--include_thickness', type=int, default=1,
                        help='whether to also predict thickness (default: 1)')
    parser.add_argument('--gpt_model_name', type=str, default="gpt-3.5-turbo",
                        help='GPT model name (default: gpt-3.5-turbo)')
    parser.add_argument('--mats_save_name', type=str, default="info_new",
                        help='candidate materials save name (default: info_new)')
    
    # Physical property prediction (uses property_name argument from above)
    parser.add_argument('--mats_load_name', type=str, default="info",
                        help='candidate materials load name (default: info)')
    parser.add_argument('--feature_load_name', type=str, default="ps56",
                        help='feature load name (default: ps56)')
    parser.add_argument('--prediction_mode', type=str, default="integral",
                        help="can be either 'integral' or 'grid' (default: integral)")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='softmax temperature for kernel regression (default: 0.01)')
    parser.add_argument('--sample_voxel_size', type=float, default=0.005,
                        help='voxel downsampling size for sampled points, relative to scaled scene (default: 0.005)')
    parser.add_argument('--volume_method', type=str, default="thickness",
                        help="method for volume estimation, either 'thickness' or 'carving' (default: thickness)")
    parser.add_argument('--correction_factor', type=float, default=0.6,
                        help='correction factor for integral prediction (default: 0.6)')
    parser.add_argument('--show_mat_seg', type=int, default=0,
                        help="whether to show visualization of material segmentation (default: 0)")
    parser.add_argument('--save_preds', type=int, default=1,
                        help='whether to save predictions (default: 1)')
    parser.add_argument('--preds_save_name', type=str, default="mass",
                        help='predictions save name (default: mass)')
    
    # Evaluation
    parser.add_argument('--preds_json_path', type=str, default="./preds/preds_mass.json",
                        help='path to predictions JSON file (default: ./preds/preds_mass.json)')
    parser.add_argument('--gts_json_path', type=str, default="./data/abo_500/filtered_product_weights.json",
                        help='path to ground truth JSON file (default: ./data/abo_500_50/filtered_product_weights.json)')
    parser.add_argument('--clamp_min', type=float, default=0.01,
                        help='minimum value to clamp predictions (default: 0.01)')
    parser.add_argument('--clamp_max', type=float, default=100.,
                        help='maximum value to clamp predictions (default: 100.)')
    
    # Visualization
    parser.add_argument('--scene_name', type=str,
                        help='scene name for visualization (must be provided)')
    parser.add_argument('--show', type=int, default=1,
                        help='whether to show interactive viewer (default: 1)')
    parser.add_argument('--compositing_alpha', type=float, default=0.2,
                        help='alpha for compositing with RGB image (default: 0.2)')
    parser.add_argument('--cmap_min', type=float, default=500,
                        help='minimum physical property value for colormap (default: 500)')
    parser.add_argument('--cmap_max', type=float, default=3500,
                        help='maximum physical property value for colormap (default: 3500)')
    parser.add_argument('--viz_save_name', type=str, default="tmp",
                        help='visualization save name (default: tmp)')

    args = parser.parse_args()

    return args
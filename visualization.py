import numpy as np 
import open3d as o3d
import os
import torch
import open_clip
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.decomposition import PCA

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from predict_property import predict_physical_property_query
from utils import parse_transforms_json, load_ns_point_cloud, load_images
from arguments import get_args


def features_to_colors(features):
    """Convert feature vectors to RGB colors using PCA."""
    pca = PCA(n_components=3)
    pca.fit(features)
    transformed = pca.transform(features)
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    transformed = (transformed - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    colors = np.clip(transformed, 0, 1)
    return colors


def similarities_to_colors(similarities, temperature=None):
    """Convert CLIP similarity values to RGB colors."""
    cmap = mpl.colormaps['tab10']
    mat_colors = [cmap(i)[:3] for i in range(similarities.shape[1])]
    if temperature is None:
        argmax_similarities = np.argmax(similarities, axis=1)
        colors = np.array([mat_colors[i] for i in argmax_similarities])
    else:
        softmax_probs = torch.softmax(torch.tensor(similarities) / temperature, dim=1)
        colors = softmax_probs @ torch.tensor(mat_colors).float()
        colors = colors.numpy()
    return colors


def values_to_colors(values, low, high):
    """Convert scalar values to RGB colors."""
    cmap = mpl.colormaps['inferno']
    colors = cmap((values - low) / (high - low))
    return colors[:, :3]


def render_pcd(pcd, w2c, K, hw=(1024, 1024), pt_size=8, savefile=None, show=False):
    h, w = hw

    # set pinhole camera parameters from K
    render_camera = o3d.camera.PinholeCameraParameters()
    render_camera.extrinsic = w2c

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(h, w, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    render_camera.intrinsic = intrinsic

    # visualize pcd from camera view with intrinsics set to K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=show)
    
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(render_camera, allow_arbitrary=True)

    # rendering options
    render_option = vis.get_render_option()
    render_option.point_size = pt_size
    render_option.point_show_normal = False
    render_option.light_on = False
    vis.update_renderer()

    if show:
        vis.run()

    if savefile is not None:
        vis.capture_screen_image(savefile, do_render=True)
        vis.destroy_window()
        return Image.open(savefile)
    else:
        render = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        return np.array(render)


def composite_and_save(img1, img2, alpha, savefile):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img = img1 * alpha + img2 * (1 - alpha)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(savefile)
    return img


def make_legend(colors, names, ncol=1, figsize=(2.0, 2.5), savefile=None, show=False):
    plt.style.use('fast')
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    plt.axis('off')
    
    # creating legend with color boxes
    ptchs = []
    for color, name in zip(colors, names):
        if len(name) > 10:  # wrap long names
            name = name.replace(' ', '\n')
        ptchs.append(mpatches.Patch(color=color[:3], label=name))
    leg = plt.legend(handles=ptchs, ncol=ncol, loc='center left', prop={'size': 18}, 
                     handlelength=1, handleheight=1, facecolor='white', framealpha=0)
    plt.tight_layout()

    if show:
        plt.show()
    if savefile is not None:
        plt.savefig(savefile, dpi=400)
    plt.close()


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')

    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    scene_dir = os.path.join(scenes_dir, args.scene_name)
    t_file = os.path.join(scene_dir, 'transforms.json')
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')

    query_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=None)
    query_pts = torch.Tensor(query_pts).to(args.device)

    result = predict_physical_property_query(args, query_pts, scene_dir, clip_model, clip_tokenizer, 
                                             return_all=True)

    out_dir = os.path.join('viz', args.viz_save_name)
    os.makedirs(out_dir, exist_ok=True)

    # legend for materials
    mat_names = result['mat_names']
    cmap_tab10 = mpl.colormaps['tab10']
    make_legend([cmap_tab10(i) for i in range(len(mat_names))], mat_names, 
                 savefile=os.path.join(out_dir, '%s_legend.png' % args.viz_save_name), show=args.show)
    
    # camera for rendering
    w2cs, K = parse_transforms_json(t_file, return_w2c=True)
    view_idx = 0
    w2c = w2cs[view_idx]
    w2c[[1, 2]] *= -1  # convert from nerfstudio to open3d format
    imgs = load_images(os.path.join(scene_dir, 'images'))
    orig_img = imgs[view_idx] / 255.

    # RGB reconstruction
    rgb_pcd = o3d.io.read_point_cloud(pcd_file)
    rgb_pcd.points = o3d.utility.Vector3dVector(query_pts.cpu().numpy())
    render = render_pcd(rgb_pcd, w2c, K, show=args.show)
    if not args.show:
        Image.fromarray(imgs[view_idx]).save(os.path.join(out_dir, '%s_rgb.png' % args.viz_save_name))

    # features PCA
    pca_pcd = o3d.geometry.PointCloud()
    pca_pcd.points = o3d.utility.Vector3dVector(query_pts.cpu().numpy())
    colors_pca = features_to_colors(result['query_features'])
    pca_pcd.colors = o3d.utility.Vector3dVector(colors_pca)
    render = render_pcd(pca_pcd, w2c, K, show=args.show)
    if not args.show:
        combined = composite_and_save(orig_img, render, args.compositing_alpha,
            savefile=os.path.join(out_dir, '%s_pca.png' % args.viz_save_name))
        
    # material segmentation
    seg_pcd = o3d.geometry.PointCloud()
    seg_pcd.points = o3d.utility.Vector3dVector(query_pts.cpu().numpy())
    colors_seg = similarities_to_colors(result['query_similarities'])
    seg_pcd.colors = o3d.utility.Vector3dVector(colors_seg)
    render = render_pcd(seg_pcd, w2c, K, show=args.show)
    if not args.show:
        combined = composite_and_save(orig_img, render, args.compositing_alpha,
            savefile=os.path.join(out_dir, '%s_seg.png' % args.viz_save_name))
        
    # physical property values
    val_pcd = o3d.geometry.PointCloud()
    val_pcd.points = o3d.utility.Vector3dVector(query_pts.cpu().numpy())
    colors_val = values_to_colors(np.mean(result['query_pred_vals'], axis=1), args.cmap_min, args.cmap_max)
    val_pcd.colors = o3d.utility.Vector3dVector(colors_val)
    render = render_pcd(val_pcd, w2c, K, show=args.show)
    if not args.show:
        combined = composite_and_save(orig_img, render, args.compositing_alpha,
            savefile=os.path.join(out_dir, '%s_%s.png' % (args.viz_save_name, args.property_name)))


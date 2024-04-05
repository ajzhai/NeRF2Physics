import os
import subprocess
import shutil
from utils import get_last_file_in_folder, get_scenes_list
from arguments import get_args


def move_files_to_folder(source_dir, target_dir):
    for file in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))


if __name__ == '__main__':
    
    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    for scene in scenes: 
        base_dir = os.path.join(scenes_dir, scene, 'ns')

        # Calling ns-train
        result = subprocess.run([
            'ns-train', 'nerfacto',
            '--data', os.path.join(scenes_dir, scene),
            '--output_dir', base_dir,
            '--vis', args.vis_mode,
            '--project_name', args.project_name,
            '--experiment_name', scene,
            '--max_num_iterations', str(args.training_iters),
            '--pipeline.model.background-color', 'random',
            '--pipeline.datamanager.camera-optimizer.mode', 'off',
            '--pipeline.model.proposal-initial-sampler', 'uniform',
            '--pipeline.model.near-plane', str(args.near_plane),
            '--pipeline.model.far-plane', str(args.far_plane),
            '--steps-per-eval-image', '10000',
        ])

        ns_dir = get_last_file_in_folder(os.path.join(base_dir, '%s/nerfacto' % scene))

        # Copying dataparser_transforms (contains scale)
        result = subprocess.run([
            'scp', '-r', 
            os.path.join(ns_dir, 'dataparser_transforms.json'), 
            os.path.join(base_dir, 'dataparser_transforms.json')
        ])

        half_bbox_size = args.bbox_size / 2

        # Calling ns-export pcd
        result = subprocess.run([
            'ns-export', 'pointcloud',
            '--load-config', os.path.join(ns_dir, 'config.yml'),
            '--output-dir', base_dir,
            '--num-points', str(args.num_points),
            '--remove-outliers', 'True',
            '--normal-method', 'open3d',
            '--use-bounding-box', 'True',
            '--bounding-box-min', str(-half_bbox_size), str(-half_bbox_size), str(-half_bbox_size),
            '--bounding-box-max', str(half_bbox_size), str(half_bbox_size), str(half_bbox_size),
        ])

        # Calling ns-render 
        result = subprocess.run([
            'ns-render', 'dataset',
            '--load-config', os.path.join(ns_dir, 'config.yml'),
            '--output-path', os.path.join(base_dir, 'renders'),
            '--rendered-output-names', 'raw-depth',
            '--split', 'train+test',
        ])

        # Collect all depths in one folder
        os.makedirs(os.path.join(base_dir, 'renders', 'depth'), exist_ok=True)
        move_files_to_folder(os.path.join(base_dir, 'renders', 'test', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))
        move_files_to_folder(os.path.join(base_dir, 'renders', 'train', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))


import os
import time
import json
import openai
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from gpt_inference import gpt_candidate_materials, gpt_thickness, parse_material_list, \
    parse_material_hardness, gpt4v_candidate_materials, parse_material_json
from utils import load_images, get_scenes_list
from arguments import get_args
from my_api_key import OPENAI_API_KEY


BASE_SEED = 100


def gpt_wrapper(gpt_fn, parse_fn, max_tries=10, sleep_time=3):
    """Wrap gpt_fn with error handling and retrying."""
    tries = 0
    # sleep to avoid overloading openai api
    time.sleep(sleep_time)
    try:
        gpt_response = gpt_fn(BASE_SEED + tries)
        result = parse_fn(gpt_response)
    except Exception as error:
        print('error:', error)
        result = None
    while result is None and tries < max_tries:
        tries += 1
        time.sleep(sleep_time)
        print('retrying...')
        try:
            gpt_response = gpt_fn(BASE_SEED + tries)
            result = parse_fn(gpt_response)
        except:
            result = None
    return gpt_response


def show_img_to_caption(scene_dir, idx_to_caption):
    img_dir = os.path.join(scene_dir, 'images')
    imgs = load_images(img_dir, bg_change=None, return_masks=False)
    img_to_caption = imgs[idx_to_caption]
    plt.imshow(img_to_caption)
    plt.show()
    plt.close()
    return


def predict_candidate_materials(args, scene_dir, show=False):
    # load caption info
    with open(os.path.join(scene_dir, '%s.json' % args.caption_load_name), 'r') as f:
        info = json.load(f)
    
    caption = info['caption']

    gpt_fn = lambda seed: gpt_candidate_materials(caption, property_name=args.property_name, 
                                                  model_name=args.gpt_model_name, seed=seed)
    parse_fn = parse_material_hardness if args.property_name == 'hardness' else parse_material_list
    candidate_materials = gpt_wrapper(gpt_fn, parse_fn)

    info['candidate_materials_%s' % args.property_name] = candidate_materials
    
    print('-' * 50)
    print('scene: %s, info:' % os.path.basename(scene_dir), info)
    print('candidate materials (%s):' % args.property_name)
    mat_names, mat_vals = parse_fn(candidate_materials)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    # save info to json
    with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'w') as f:
        json.dump(info, f, indent=4)

    return info


def predict_object_info_gpt4v(args, scene_dir, show=False):
    """(EXPERIMENTAL) Predict materials directly from image with GPT-4V."""
    img_dir = os.path.join(scene_dir, 'images')
    imgs, masks = load_images(img_dir, return_masks=True)
    mask_areas = [np.mean(mask) for mask in masks]

    idx_to_caption = np.argsort(mask_areas)[int(len(mask_areas) * args.mask_area_percentile)]
    img_to_caption = imgs[idx_to_caption]

    # save img_to_caption in img_dir
    img_to_caption = Image.fromarray(img_to_caption)
    img_path = os.path.join(scene_dir, 'img_to_caption.png')
    img_to_caption.save(img_path)

    gpt_fn = lambda seed: gpt4v_candidate_materials(img_path, property_name=args.property_name, seed=seed)
    candidate_materials = gpt_wrapper(gpt_fn, parse_material_json)

    info = {'idx_to_caption': str(idx_to_caption), 
            'candidate_materials_%s' % args.property_name: candidate_materials}
    
    print('-' * 50)
    print('scene: %s, info:' % os.path.basename(scene_dir), info)
    print('candidate materials (%s):' % args.property_name)
    mat_names, mat_vals = parse_material_list(candidate_materials)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    # save info to json
    with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'w') as f:
        json.dump(info, f, indent=4)

    return info


def predict_thickness(args, scene_dir, mode='list', show=False):
    # load info
    with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'r') as f:
        info = json.load(f)
    
    if mode == 'list': 
        caption = info['caption']
    elif mode == 'json':  # json contains caption inside
        caption = None
    else:
        raise NotImplementedError
    candidate_materials = info['candidate_materials_density']

    gpt_fn = lambda seed: gpt_thickness(caption, candidate_materials, 
                                        model_name=args.gpt_model_name,  mode=mode, seed=seed)
    thickness = gpt_wrapper(gpt_fn, parse_material_list)

    info['thickness'] = thickness
    
    print('thickness (cm):')
    mat_names, mat_vals = parse_material_list(thickness)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    # save info to json
    with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'w') as f:
        json.dump(info, f, indent=4)

    return info


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    openai.api_key = OPENAI_API_KEY

    for j, scene in enumerate(scenes): 
        mats_info = predict_candidate_materials(args, os.path.join(scenes_dir, scene))
        if args.include_thickness:
            mats_info = predict_thickness(args, os.path.join(scenes_dir, scene))

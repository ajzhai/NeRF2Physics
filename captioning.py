import os
import numpy as np 
import json
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from utils import load_images, get_scenes_list
from arguments import get_args


CAPTIONING_PROMPT = "Question: Give a detailed description of the object. Answer:"


def load_blip2(model_name, device='cuda'):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    return model, processor


def display_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def generate_text(img, model, processor, prompt=CAPTIONING_PROMPT, device='cuda'):
    if prompt is not None:
        inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
    
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def predict_caption(args, scene_dir, vqa_model, vqa_processor, show=False):
    img_dir = os.path.join(scene_dir, 'images')
    imgs, masks = load_images(img_dir, return_masks=True)
    mask_areas = [np.mean(mask) for mask in masks]

    idx_to_caption = np.argsort(mask_areas)[int(len(mask_areas) * args.mask_area_percentile)]
    img_to_caption = imgs[idx_to_caption]

    with torch.no_grad():
        caption = generate_text(img_to_caption, vqa_model, vqa_processor, device=args.device)

    info = {'idx_to_caption': str(idx_to_caption), 'caption': caption} 

    print('scene: %s, info:' % os.path.basename(scene_dir), info)
    if show:
        plt.imshow(img_to_caption)
        plt.show()
    
    # save info to json
    with open(os.path.join(scene_dir, '%s.json' % args.caption_save_name), 'w') as f:
        json.dump(info, f, indent=4)

    return info


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    model, processor = load_blip2(args.blip2_model_dir, device=args.device)

    for j, scene in enumerate(scenes):
        caption_info = predict_caption(args, os.path.join(scenes_dir, scene), model, processor)
    

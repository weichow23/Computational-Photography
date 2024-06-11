import seaborn as sns
import itertools
import os
from PIL import Image
import numpy as np

def center_crop(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    return image.crop((left, top, right, bottom))

def filter_img_path(paths):
    return [x for x in paths if x.split('.')[-1] in ['jpg', 'jpeg', '.png']]

def init_color(count=10):
    '''
    count: number of color
    '''
    rgb2hex = lambda x: '#%02x%02x%02x' % tuple(x)

    palette = sns.color_palette()
    for i in range(len(palette)):
        color = [int(x * 255) for x in palette[i]]
        palette[i] = rgb2hex(color)
    palette = itertools.cycle(palette)
    colors = [next(palette) for i in range(count)]

    return colors

def init_default_source(source_img_dir, fixed_width, fixed_height):
    source_img_paths = filter_img_path(os.listdir(source_img_dir))
    source_img_paths = [os.path.join(source_img_dir, x) for x in source_img_paths]
    source_pil_imgs = [center_crop(Image.open(x).convert('RGB'), fixed_width, fixed_height) for x in source_img_paths]

    examples = [[i] for i in source_img_paths]
    return source_pil_imgs, examples

def process_mask(mask_dict, fixed_width, fixed_height):
    mask = np.zeros((fixed_height, fixed_width), dtype=bool)
    if 'mask' in mask_dict and mask_dict['mask'] is not None:
        mask = np.array(mask_dict['mask']).astype(bool)
        mask_dict['mask'] = None # 清除笔刷 没用，该属性清空了，但是图的显示没变
    return mask

def clean_all_canvas():
    return None, None, None, None

def clean_all_canvas_multi():
    return None, None, None, None, None, None

title = "<center><strong><font size='8'>Interactive Digital Montage<font></strong></center>"
description = "<center><font size='4'>ZJU  3210103790<font></center>"
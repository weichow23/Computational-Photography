import gradio as gr
import os
from PIL import Image
import numpy as np
import seaborn as sns
import yaml
import itertools
from utils import center_crop, filter_img_path
from montage import alpha_beta_swap, create_composite
from termcolor import cprint

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SOURCE_IMG_DIR = config['source_img_dir']
OUTPUT_IMG_DIR = config['output_img_dir']
CANVAS_BORDER_WIDTH = config['canvas_border_width']
BRUSH_WIDTH = config['brush_width']
FIXED_WIDTH = 400
FIXED_HEIGHT = 300

if not os.path.exists(OUTPUT_IMG_DIR):
    os.mkdir(OUTPUT_IMG_DIR)

SOURCE_IMG_PATHS = filter_img_path(os.listdir(SOURCE_IMG_DIR))
SOURCE_IMG_PATHS = [os.path.join(SOURCE_IMG_DIR, x) for x in SOURCE_IMG_PATHS]
SOURCE_PIL_IMGS = [center_crop(Image.open(x).convert('RGB'), FIXED_WIDTH, FIXED_HEIGHT) for x in SOURCE_IMG_PATHS]
CURRENT_SOURCE_IDX = 0
COMPOSITE_ARRAY = np.array(SOURCE_PIL_IMGS[0])
LABEL_MAP_PATH = os.path.join(OUTPUT_IMG_DIR, 'label_map.jpg')

hex2rgb = lambda x: tuple(int(x[i:i + 2], 16) for i in (1, 3, 5))
rgb2hex = lambda x: '#%02x%02x%02x' % tuple(x)
palette = sns.color_palette()
for i in range(len(palette)):
    color = [int(x * 255) for x in palette[i]]
    palette[i] = rgb2hex(color)
palette = itertools.cycle(palette)
COLORS = [next(palette) for i in range(len(SOURCE_PIL_IMGS))]

label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64)

def update_label_map(label_map, mask, idx):
    mask = mask.astype(bool)
    label_map[mask] = idx  # 这样可以区分多张图片的来源. 初始值为全0

def show_label_map(label_map):
    label_map_image = np.zeros(shape=[FIXED_HEIGHT, FIXED_WIDTH, 3], dtype=np.uint8)
    for i in range(len(COLORS)):
        color_array = np.array(list(hex2rgb(COLORS[i])))
        for k in range(3):
            m = label_map_image[:, :, k]
            m[label_map == i] = color_array[k]
            label_map_image[:, :, k] = m
    return Image.fromarray(label_map_image)

all_true_mask = np.ones_like(label_map, dtype=bool)
update_label_map(label_map, all_true_mask, 0)
label_map_image = show_label_map(label_map)

def process_mask(mask_dict):
    mask = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=bool)
    if 'mask' in mask_dict and mask_dict['mask'] is not None:
        print('process mask')
        mask = np.array(mask_dict['mask']).astype(bool)
        mask_dict['mask'] = None # 清除笔刷 没用，该属性清空了，但是图的显示没变
    return mask

def clean_all_canvas():
    return None, None, None, None

def use_default():
    return SOURCE_PIL_IMGS[0], SOURCE_PIL_IMGS[1], None, None

def run(composite_input, source_input):
    global CURRENT_SOURCE_IDX
    composite_mask = process_mask(composite_input)
    source_mask = process_mask(source_input)

    # 将RGB图像转换为灰度图像
    composite_mask = composite_mask[:, :, 0]  # 保留第一个通道
    source_mask = source_mask[:, :, 0]  # 保留第一个通道

    binary_map = alpha_beta_swap(COMPOSITE_ARRAY, np.array(SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX]), composite_mask, source_mask)
    update_label_map(label_map, binary_map, CURRENT_SOURCE_IDX)
    label_map_image = show_label_map(label_map)
    composite_image = create_composite(binary_map=binary_map, source=np.array(SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX]),
                                       target=COMPOSITE_ARRAY)

    return composite_image, label_map_image, SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX]

def run_single(source_input_0, source_input_1):
    global CURRENT_SOURCE_IDX
    composite_mask = process_mask(source_input_0)
    source_mask = process_mask(source_input_1)
    # print(COMPOSITE_ARRAY) # source_input_0['image']
    # print(SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX]) source_input_1['image']

    # 将RGB图像转换为灰度图像
    composite_mask = composite_mask[:, :, 0]  # 保留第一个通道
    source_mask = source_mask[:, :, 0]  # 保留第一个通道

    binary_map = alpha_beta_swap(source_input_0['image'], np.array(source_input_1['image']), composite_mask, source_mask)
    label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64) # todo: 后面要把这个删了，隔离开来
    update_label_map(label_map, binary_map, 1)
    label_map_image = show_label_map(label_map)
    composite_image = create_composite(binary_map=binary_map, source=np.array(source_input_1['image']),
                                       target=source_input_0['image'])
    label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64) # todo: 后面要把这个删了，隔离开来
    return composite_image, label_map_image

def next_image():
    global CURRENT_SOURCE_IDX
    CURRENT_SOURCE_IDX = (CURRENT_SOURCE_IDX + 1) % len(SOURCE_PIL_IMGS)
    # source_canvas.brush_color = '#FFFFFF' # 没用，改不了
    # source_canvas.change()
    return SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX]


def reset_source():
    return SOURCE_PIL_IMGS[CURRENT_SOURCE_IDX]


def reset_composite():
    global COMPOSITE_ARRAY
    COMPOSITE_ARRAY = np.array(SOURCE_PIL_IMGS[0])
    return Image.fromarray(COMPOSITE_ARRAY)


with gr.Blocks(css=".block {padding: 10px;} .gr-button {margin: 5px;}") as demo:
    with gr.Group():
        gr.Markdown("## 两张图片蒙太奇", elem_classes=["block"], elem_id="header1")
        with gr.Row(elem_classes=["block"]):
            source_canvas_0 = gr.Image(label="Source Image 1", tool="sketch", height=FIXED_HEIGHT, width=FIXED_WIDTH,
                                     container=True, brush_color=COLORS[0], value=SOURCE_PIL_IMGS[0])
            source_canvas_1 = gr.Image(label="Source Image 2", tool="sketch", height=FIXED_HEIGHT, width=FIXED_WIDTH,
                                     container=True, brush_color=COLORS[1], value=SOURCE_PIL_IMGS[1])
            label_map_canvas = gr.Image(label="Label Map", height=FIXED_HEIGHT, width=FIXED_WIDTH, container=True)
            composite_canvas = gr.Image(label="Composite Image", height=FIXED_HEIGHT, width=FIXED_WIDTH, container=True)

        with gr.Row(elem_classes=["block"]):
            run_button = gr.Button("Run 🏃‍♂️", elem_classes=["gr-button"])
            clean_all_canvas_button = gr.Button("Clean 🧹", elem_classes=["gr-button"])
            use_default_button = gr.Button("Use Default 🔄", elem_classes=["gr-button"])

        run_button.click(run_single, inputs=[source_canvas_0, source_canvas_1],
                         outputs=[composite_canvas, label_map_canvas])
        use_default_button.click(use_default, outputs=[source_canvas_0, source_canvas_1, composite_canvas, label_map_canvas])
        clean_all_canvas_button.click(clean_all_canvas, outputs=[source_canvas_0, source_canvas_1, composite_canvas, label_map_canvas])

    # -----------------------------------------------------------------------------------------------
    gr.Markdown("---", elem_classes=["block"])  # 添加分隔线

    with gr.Group():
        gr.Markdown("## 多张图片(施工)", elem_classes=["block"], elem_id="header2")
        with gr.Row(elem_classes=["block"]):
            source_canvas_0 = gr.Image(label="Source Image", tool="sketch", height=FIXED_HEIGHT, width=FIXED_WIDTH,
                                     container=True)
            source_canvas_1 = gr.Image(label="Source Image", tool="sketch", height=FIXED_HEIGHT, width=FIXED_WIDTH,
                                     container=True)
            label_map_canvas = gr.Image(label="Label Map", height=FIXED_HEIGHT, width=FIXED_WIDTH, container=True)
            composite_canvas = gr.Image(label="Composite Image", height=FIXED_HEIGHT, width=FIXED_WIDTH, container=True)

        with gr.Row(elem_classes=["block"]):
            run_button = gr.Button("Run 🏃‍", elem_classes=["gr-button"])
            next_button = gr.Button("Next image", elem_classes=["gr-button"])
            reset_source_button = gr.Button("Reset Source", elem_classes=["gr-button"])
            reset_composite_button = gr.Button("Reset Composite", elem_classes=["gr-button"])

        run_button.click(run, inputs=[source_canvas_0, source_canvas_1],
                         outputs=[composite_canvas, label_map_canvas])
        next_button.click(next_image, outputs=source_canvas_1)
        reset_source_button.click(reset_source, outputs=source_canvas_1)
        reset_composite_button.click(reset_composite, outputs=source_canvas_0)

demo.launch()
import gradio as gr
from PIL import Image
import numpy as np
import yaml
from montage.utils import center_crop, filter_img_path, init_color, init_default_source, process_mask, \
    clean_all_canvas, title, description, clean_all_canvas_multi
from montage.montage import alpha_beta_swap, create_composite
from termcolor import cprint
from tqdm import tqdm

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

FIXED_WIDTH = config["fixed_width"]
FIXED_HEIGHT = config["fixed_height"]

SOURCE_PIL_IMGS, examples = init_default_source(source_img_dir=config['source_img_dir'],
                                      fixed_width=FIXED_WIDTH, fixed_height=FIXED_HEIGHT)
hex2rgb = lambda x: tuple(int(x[i:i + 2], 16) for i in (1, 3, 5))
COLORS = init_color()

def update_label_map(label_map, mask, idx):
    mask = mask.astype(bool)
    label_map[mask] = idx  # 这样可以区分多张图片的来源. 初始值为全0
    return label_map

def show_label_map(label_map):
    label_map_image = np.zeros(shape=[FIXED_HEIGHT, FIXED_WIDTH, 3], dtype=np.uint8)
    for i in range(len(COLORS)):
        color_array = np.array(list(hex2rgb(COLORS[i])))
        for k in range(3):
            m = label_map_image[:, :, k]
            m[label_map == i] = color_array[k]
            label_map_image[:, :, k] = m
    return Image.fromarray(label_map_image)

def use_default():
    return SOURCE_PIL_IMGS[0], SOURCE_PIL_IMGS[1], None, None
def use_default_multi():
    return SOURCE_PIL_IMGS[0], SOURCE_PIL_IMGS[1], SOURCE_PIL_IMGS[2], SOURCE_PIL_IMGS[3], None, None

def run_multi(source_input_1, source_input_2, source_input_3, source_input_4):
    source_inputs = [source_input_1, source_input_2, source_input_3, source_input_4]

    source_mask_list = [process_mask(s, fixed_width=FIXED_WIDTH, fixed_height=FIXED_HEIGHT)[:, :, 0] for s in source_inputs]
    label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64)
    composite_image = None
    label_map_image = None
    for idx in tqdm(range(len(source_inputs)-1)): # 从0开始
        if composite_image is None:
            composite_image = source_inputs[0]['image']
        binary_map = alpha_beta_swap(composite=np.array(composite_image),
                                     source=np.array(np.array(source_inputs[idx+1]['image'])),
                                     composite_mask=source_mask_list[idx], source_mask=source_mask_list[idx+1])
        label_map = update_label_map(label_map, binary_map, idx+1)
        label_map_image = show_label_map(label_map)
        composite_image = create_composite(binary_map=binary_map, source=np.array(source_inputs[idx+1]['image']),
                                       target=np.array(composite_image)) # todo:这里检查下source， tar对不对
        composite_image.save(f"image_{idx}.jpg")
        label_map_image.save(f"label_{idx}.jpg")
    return composite_image, label_map_image

def run_single(source_input_0, source_input_1):
    composite_mask = process_mask(source_input_0, fixed_width=FIXED_WIDTH, fixed_height=FIXED_HEIGHT)
    source_mask = process_mask(source_input_1, fixed_width=FIXED_WIDTH, fixed_height=FIXED_HEIGHT)
    # 将RGB图像转换为灰度图像
    composite_mask = composite_mask[:, :, 0]  # 保留第一个通道
    source_mask = source_mask[:, :, 0]  # 保留第一个通道

    binary_map = alpha_beta_swap(source_input_0['image'], np.array(source_input_1['image']), composite_mask, source_mask)
    label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64)
    label_map = update_label_map(label_map, binary_map, 1)
    label_map_image = show_label_map(label_map)
    composite_image = create_composite(binary_map=binary_map, source=np.array(source_input_1['image']),
                                       target=source_input_0['image'])

    return composite_image, label_map_image

with gr.Blocks(css=".block {padding: 10px;} .gr-button {margin: 5px;}") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(title)
            gr.Markdown(description)

    with gr.Tabs():
        with gr.TabItem("两张图片蒙太奇"):
            gr.Markdown("#### 指导")
            gr.Markdown("1. 点击 <Use Default 🔄> 或者自己上传图片")
            gr.Markdown("2. 在两张图片上分别涂抹mask")
            gr.Markdown("3. 点击 <Run 🏃‍> 进行图像蒙太奇")
            gr.Markdown("4. 点击 <Clean 🧹> 就会清空重来")
            with gr.Row(elem_classes=["block"]):
                source_canvas_0 = gr.Image(label="Source Image 1", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                         container=True, brush_color=COLORS[0])
                source_canvas_1 = gr.Image(label="Source Image 2", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                         container=True, brush_color=COLORS[1])
                label_map_canvas = gr.Image(label="Label Map", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)
                composite_canvas = gr.Image(label="Composite Image", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)

            with gr.Row(elem_classes=["block"]):
                run_button = gr.Button("Run 🏃‍♂️", elem_classes=["gr-button"])
                clean_all_canvas_button = gr.Button("Clean 🧹", elem_classes=["gr-button"])
                use_default_button = gr.Button("Use Default 🔄", elem_classes=["gr-button"])

            run_button.click(run_single, inputs=[source_canvas_0, source_canvas_1],
                             outputs=[composite_canvas, label_map_canvas])
            use_default_button.click(use_default, outputs=[source_canvas_0, source_canvas_1, composite_canvas, label_map_canvas])
            clean_all_canvas_button.click(clean_all_canvas, outputs=[source_canvas_0, source_canvas_1, composite_canvas, label_map_canvas])

        with gr.TabItem("bonus 多图像笔刷(施工)"):  # todo: 这边好像涂抹有点问题
            gr.Markdown("#### 指导")
            gr.Markdown("1. 点击 <Use Default 🔄> 或者自己上传图片")
            gr.Markdown("2. 在图片上分别涂抹mask")
            gr.Markdown("3. 点击 <Run 🏃‍> 进行图像蒙太奇")
            gr.Markdown("4. 点击 <Clean 🧹> 就会清空重来")
            with gr.Row(elem_classes=["block"]):
                # source_canvases = [
                #     gr.Image(label=f"Source Image {i + 1}", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                #              container=True, brush_color=COLORS[i % len(COLORS)], value=SOURCE_PIL_IMGS[i])
                #     for i in range(len(SOURCE_PIL_IMGS))]  # 这么写无法修改

                source_canvas_multi1 = gr.Image(label="Source Image 1", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                      container=True, brush_color=COLORS[0])
                source_canvas_multi2 = gr.Image(label="Source Image 2", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                         container=True, brush_color=COLORS[1])
                source_canvas_multi3 = gr.Image(label="Source Image 3", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                         container=True, brush_color=COLORS[2])
                source_canvas_multi4 = gr.Image(label="Source Image 4", tool="sketch", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                         container=True, brush_color=COLORS[3])

            with gr.Row(elem_classes=["block"]):
                label_map_canvas = gr.Image(label="Label Map", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)
                composite_canvas_multi = gr.Image(label="Composite Image", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)

            with gr.Row(elem_classes=["block"]):
                run_button_multi = gr.Button("Run 🏃‍", elem_classes=["gr-button"])
                clean_all_canvas_button_multi = gr.Button("Clean 🧹", elem_classes=["gr-button"])
                use_default_button_multi = gr.Button("Use Default 🔄", elem_classes=["gr-button"])

            run_button_multi.click(run_multi,
                  inputs=[source_canvas_multi1, source_canvas_multi2, source_canvas_multi3, source_canvas_multi4],
                  outputs=[composite_canvas_multi, label_map_canvas])
            use_default_button_multi.click(use_default_multi,
                                outputs=[source_canvas_multi1, source_canvas_multi2, source_canvas_multi3,
                                          source_canvas_multi4, composite_canvas, label_map_canvas])
            clean_all_canvas_button_multi.click(clean_all_canvas_multi,
                                outputs=[source_canvas_multi1, source_canvas_multi2, source_canvas_multi3,
                                          source_canvas_multi4, composite_canvas, label_map_canvas])
        with gr.TabItem("bonus 单一图像笔刷(施工)"):
            gr.Markdown("#### 指导")
            gr.Markdown("1. 点击 <Use Default 🔄> 或者自己上传图片")
            gr.Markdown("2. ")
            gr.Markdown("3. 点击 <Run 🏃‍> 进行图像蒙太奇")
            gr.Markdown("4. 点击 <Clean 🧹> 就会清空重来")

demo.launch()
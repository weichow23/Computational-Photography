import gradio as gr
from PIL import Image
import numpy as np
import yaml
from montage.utils import center_crop, filter_img_path, init_color, init_default_source, process_mask, \
    clean_all_canvas, title, description, clean_all_canvas_multi
from montage.montage import alpha_beta_swap, create_composite
from termcolor import cprint
from tqdm import tqdm
from sam.app.sam_utils import segment_with_points, get_points_with_draw, segment_everything
import os

os.environ["PYTHONPATH"] = './'

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

FIXED_WIDTH = config["fixed_width"]
FIXED_HEIGHT = config["fixed_height"]

SOURCE_PIL_IMGS, examples_o = init_default_source(source_img_dir=config['source_img_dir'],
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

    # mask: (h, w)
    source_mask_list = [process_mask(s, fixed_width=FIXED_WIDTH, fixed_height=FIXED_HEIGHT)[:, :, 0] for s in source_inputs]
    label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64)
    composite_image = None
    label_map_image = None
    histrory_mask = None
    for idx in tqdm(range(len(source_inputs)-1)): # 从0开始
        if composite_image is None:
            composite_image = source_inputs[0]['image']
        if histrory_mask is None:
            assert idx == 0
            histrory_mask = source_mask_list[idx]
        else:
            histrory_mask = np.logical_or(histrory_mask, source_mask_list[idx])
            histrory_mask = np.logical_and(histrory_mask, np.logical_not(source_mask_list[idx + 1]))
        binary_map = alpha_beta_swap(composite=np.array(composite_image),
                                     source=np.array(np.array(source_inputs[idx+1]['image'])),
                                     composite_mask=histrory_mask, source_mask=source_mask_list[idx+1])
        label_map = update_label_map(label_map, binary_map, idx+1)
        label_map_image = show_label_map(label_map)
        composite_image = create_composite(binary_map=binary_map, source=np.array(source_inputs[idx+1]['image']),
                                       target=np.array(composite_image))
        # 保存之间的过程图
        composite_image.save(f"test/multi/image_{idx}.jpg")
        label_map_image.save(f"test/multi/label_{idx}.jpg")
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

def run_single_p(source_input, cond_input):
    '''
    source_input: np.array, cond_input: PIL.Image
    '''
    source_mask = np.load('test/tmp/annotations.npy')
    composite_mask = np.logical_not(source_mask)
    cond_input = Image.open("test/tmp/annotations.png")
    binary_map = alpha_beta_swap(source_input, np.array(cond_input), composite_mask, source_mask)
    label_map = np.zeros((FIXED_HEIGHT, FIXED_WIDTH), dtype=np.int64)
    label_map = update_label_map(label_map, binary_map, 1)
    label_map_image = show_label_map(label_map)
    composite_image = create_composite(binary_map=binary_map, source=np.array(cond_input), target=source_input)

    return composite_image, label_map_image

examples = [
    ["./sam/assets/picture3.jpg"],
    ["./sam/assets/picture4.jpg"],
    ["./sam/assets/picture5.jpg"],
    ["./sam/assets/picture6.jpg"],
    ["./sam/assets/picture1.jpg"],
    ["./sam/assets/picture2.jpg"],
    ["./sam/assets/xiaohuangren.png"]
] + examples_o
cond_img_p = gr.Image(label="Input with points", shape=(FIXED_WIDTH, FIXED_HEIGHT), value=examples[0][0], type="pil")
segm_img_p = gr.Image(label="Segmented Image with points", shape=(FIXED_WIDTH, FIXED_HEIGHT), interactive=False)


with gr.Blocks(css=".block {padding: 10px;} .gr-button {margin: 5px;}", title="Interactive Digital Montage") as demo:
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

        with gr.TabItem("bonus 多图像笔刷"):
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
                label_map_canvas_multi = gr.Image(label="Label Map", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)
                composite_canvas_multi = gr.Image(label="Composite Image", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)

            with gr.Row(elem_classes=["block"]):
                run_button_multi = gr.Button("Run 🏃‍", elem_classes=["gr-button"])
                clean_all_canvas_button_multi = gr.Button("Clean 🧹", elem_classes=["gr-button"])
                use_default_button_multi = gr.Button("Use Default 🔄", elem_classes=["gr-button"])

            run_button_multi.click(run_multi,
                  inputs=[source_canvas_multi1, source_canvas_multi2, source_canvas_multi3, source_canvas_multi4],
                  outputs=[composite_canvas_multi, label_map_canvas_multi])
            use_default_button_multi.click(use_default_multi,
                                outputs=[source_canvas_multi1, source_canvas_multi2, source_canvas_multi3,
                                          source_canvas_multi4, composite_canvas, label_map_canvas_multi])
            clean_all_canvas_button_multi.click(clean_all_canvas_multi,
                                outputs=[source_canvas_multi1, source_canvas_multi2, source_canvas_multi3,
                                          source_canvas_multi4, composite_canvas, label_map_canvas_multi])
        with gr.TabItem("bonus 单一图像笔刷"):
            gr.Markdown("#### 指导")
            gr.Markdown("1. 在examples中选择或者自己上传图片")
            gr.Markdown("2. 在Input with points上标记")
            gr.Markdown("3. 点击 <Cut out objects ✂️>, 一定要确保annotations产生")
            gr.Markdown("4. 点击 <Run 🏃‍> 进行图像蒙太奇")
            gr.Markdown("注意，重新SAM的时候，需要点击 <Restart SAM 🔄> ; 你想测试Mobile SAM在seg everything上的能力，请点击 <Segmenting anything! 💥>")
            with gr.Tab("Point mode"):
                # Images
                with gr.Row(variant="panel"):
                    with gr.Column(scale=1):
                        source_canvas_p = gr.Image(label="Source Image", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                               value=SOURCE_PIL_IMGS[0])
                    with gr.Column(scale=1):
                        cond_img_p.render()
                    with gr.Column(scale=1):
                        segm_img_p.render()

                with gr.Row(variant="panel"):

                    label_map_canvas_p = gr.Image(label="Label Map", shape=(FIXED_WIDTH, FIXED_HEIGHT), container=True)
                    composite_canvas_p = gr.Image(label="Composite Image", shape=(FIXED_WIDTH, FIXED_HEIGHT),
                                                  container=True)

                # Submit & Clear
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("Try some of the examples below ⬇️")
                                gr.Examples(
                                    examples=examples,
                                    inputs=[cond_img_p],
                                    examples_per_page=5,
                                )

                    with gr.Column():
                        segment_btn_p = gr.Button("Cut out objects ✂️", variant="primary")
                        segment_any_p = gr.Button("Segmenting anything! 💥", variant="primary")
                        clear_btn_p = gr.Button("Restart SAM 🔄", variant="primary")
                        run_button_p = gr.Button("Run 🏃‍", variant="secondary")

            cond_img_p.select(get_points_with_draw, [cond_img_p], cond_img_p)
            segment_any_p.click(segment_everything, inputs=[cond_img_p], outputs=[segm_img_p])
            segment_btn_p.click(segment_with_points, inputs=[cond_img_p], outputs=[segm_img_p, cond_img_p])
            run_button_p.click(run_single_p, inputs=[source_canvas_p, cond_img_p],
                             outputs=[composite_canvas_p, label_map_canvas_p])

            def clear():
                return None, None

            clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])

demo.launch()
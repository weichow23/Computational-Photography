import gradio as gr
import numpy as np
import torch
from sam.mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw, Image
from sam.app.utils.tools import format_results, point_prompt
from sam.app.utils.tools_gradio import fast_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
# sam_checkpoint = "../weights/mobile_sam.pt"
sam_checkpoint = "./sam/weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)
global_points = []
global_point_label = []

@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    return fig


def segment_with_points(
    image,
    input_size=1024, # 这个应该是sam需要的输入大小
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label

    input_size = int(input_size)  # 这个是图片被crop之后的，由gr.Image决定
    w, h = image.size
    # zw: 为了保持形状一致
    scale = 1 # input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))  # [w, h]
    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    # print(scaled_points) # 有点奇怪
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image
    # 这里已经有points了

    nd_image = np.array(image) # [h, w, 3]
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
    ) # mask (3, 768, 1024)
    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations]) # (1, h, w)
    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []

    np.save('test/tmp/annotations.npy', annotations.squeeze(0)) # (h, w)
    return fig, image


def get_points_with_draw(image, evt: gr.SelectData):
    global global_points
    global global_point_label

    # 去掉黄点
    if global_point_label == []:
        image.save(f"test/tmp/annotations.png")

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0)
    global_points.append([x, y])
    global_point_label.append(1)
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image

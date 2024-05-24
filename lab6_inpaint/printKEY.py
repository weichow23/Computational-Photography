import torch
import network
from torchviz import make_dot

def print_model_info(path):
    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))['netG']
    for key, value in pretrained_dict.items():
        print(key)

def viz():
    model = network.Generator()
    img = torch.randn(1, 4, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    output = model(img, mask)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("/home1/zhouwei/Formulation/code/Lab6-DL/ImageCompletionPlus/images/viz.png", format="png")  # 保存为png文件
    # dot.view()  # 在默认图片查看器中打开图像

if __name__ == "__main__":
    viz()
    # path = 'model/places2/model.pth'
    # print_model_info(path)

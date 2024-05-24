<center>
  <font face="黑体" size = 5>
    深度学习图像补全
  </font>
   <center><font face="黑体" size = 5>
     lab 6
  </font>
  <center><font face="黑体" size = 4>
    姓名： 周炜
  </font>
  <center><font face="黑体" size = 4>
    学号： 32010103790
  </font>
</center> 




[TOC]

# 简单的深度学习图像补全

在 `Lab6-DL/ImageCompletion` 目录下执行 `python demo.py`, 就能看到图片补全的效果。

有2个bug需要修正一下

bug1：运行设备

```shell
RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cpu
```

需要在`main()`函数中增加`.to(device)`的代码

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mpv = mpv.to(device)
    mask = mask.to(device)
    x = x.to(device)
    # inpaint
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x, output, mask).to(device)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        show_img(imgs, nrow=3)
```

bug2:  `make_grid`中的`range`参数已经移除了，需要删除

```python
def show_img(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
```

运行结果如下：

![image-20240403132155209](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240403132155209.png)

# 稍微复杂的深度学习图像补全

### 代码补全

只需要照着ppt上的画法连接一遍即可，补全后的`class Generator(nn.Module)`的结果如下:
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        cnum = 32

        self.dw_conv01 = nn.Sequential(
            nn.Conv2d(4, cnum, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dw_conv02 = nn.Sequential(
            nn.Conv2d(cnum, cnum * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dw_conv03 = nn.Sequential(
            nn.Conv2d(cnum * 2, cnum * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dw_conv04 = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dw_conv05 = nn.Sequential(
            nn.Conv2d(cnum * 8, cnum * 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dw_conv06 = nn.Sequential(
            nn.Conv2d(cnum * 16, cnum * 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # attention module
        self.at_conv05 = AttentionConv(cnum * 16, cnum * 16, ksize=1, fuse=False)
        self.at_conv04 = AttentionConv(cnum * 8, cnum * 8)
        self.at_conv03 = AttentionConv(cnum * 4, cnum * 4)
        self.at_conv02 = AttentionConv(cnum * 2, cnum * 2)
        self.at_conv01 = AttentionConv(cnum, cnum)

        # decoder
        self.up_conv05 = nn.Sequential(
            nn.Conv2d(cnum * 16, cnum * 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_conv04 = nn.Sequential(
            nn.Conv2d(cnum * 32, cnum * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_conv03 = nn.Sequential(
            nn.Conv2d(cnum * 16, cnum * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_conv02 = nn.Sequential(
            nn.Conv2d(cnum * 8, cnum * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_conv01 = nn.Sequential(
            nn.Conv2d(cnum * 4, cnum, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(cnum * 2, cnum, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnum, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, img, mask):
        x = img
        # encoder
        x1 = self.dw_conv01(x)
        x2 = self.dw_conv02(x1)
        x3 = self.dw_conv03(x2)
        x4 = self.dw_conv04(x3)
        x5 = self.dw_conv05(x4)
        x6 = self.dw_conv06(x5)
        # attention
        x5 = self.at_conv05(x5, x6, mask)
        x4 = self.at_conv04(x4, x5, mask)
        x3 = self.at_conv03(x3, x4, mask)
        x2 = self.at_conv02(x2, x3, mask)
        x1 = self.at_conv01(x1, x2, mask)
        # decoder
        upx5 = self.up_conv05(
            F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=True))
        upx4 = self.up_conv04(
            F.interpolate(torch.cat([upx5, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        upx3 = self.up_conv03(
            F.interpolate(torch.cat([upx4, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        upx2 = self.up_conv02(
            F.interpolate(torch.cat([upx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        upx1 = self.up_conv01(
            F.interpolate(torch.cat([upx2, x2], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        # output
        output = self.decoder(
            F.interpolate(torch.cat([upx1, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=True))

        return output
```

### 模型可视化

使用`torchviz`库来可视化神经网络的结构。`torchviz`可以生成PyTorch模型的计算图. 但是由于图片太长，我放在了最后

这是一个典型的`U-NET`架构，是由[PEN-Net](https://github.com/researchmm/PEN-Net-for-Inpainting/blob/pytorch/model/pennet.py#L139) 更改过来的

### 结果

实验材料的图片中最优的是place2但是还是不够真实

![image-20240403145308637](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240403145308637.png)

采用我自己得图片结果如下：

![image-20240403180703282](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240403180703282.png)

这里效果最好的是facade但是可以发现效果还是很差

![image-20240403180800624](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240403180800624.png)

PEN-Net这样早期的GAN网络效果较差，而且训练的数据集也较小。目前`DiT`和其他的diffusion模型在inpaint上表现得已经更优了

# 附录




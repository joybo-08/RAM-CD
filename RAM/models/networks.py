import torch
import torchvision
import torch.nn as nn
from models.VMoE import VMoE
import models
import numpy as np
from models.help_funcs import *
from thop import profile
import time
# import matplotlib.pyplot as plt


def visualize_tensor(tensor, title, ax):
    """可视化张量"""
    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]  # 取第一个样本

    # 处理不同张量类型
    if tensor.shape[0] == 3:  # RGB图像
        img = np.transpose(tensor, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 归一化
        ax.imshow(img)
    else:  # 特征图
        if tensor.shape[0] > 1:
            tensor = tensor.mean(axis=0)  # 多通道取平均
        else:
            tensor = tensor.squeeze(0)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        ax.imshow(tensor, cmap='viridis')

    ax.set_title(title)
    ax.axis('off')

###############################################################################
# main Functions
###############################################################################

class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,dim=32):
        super(ResNet, self).__init__()

        resnet = models.resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.conv = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.dim = dim
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        expand = 2
        layers = 256 * expand

        self.conv1 = nn.Conv2d(layers//8, self.dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(layers//4, self.dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(layers//2, self.dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(layers, self.dim, kernel_size=3, padding=1)

        self.edge_detector = AdaptiveEdgeDetector(input_nc)


    def forward_layer(self, x):
        # 边缘增强
        x = self.edge_detector(x)
        # resnet layers
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)  # 1/4, in=64, out=64
        x_2 = self.layer2(x_1)  # 1/8, in=64, out=128
        x_3 = self.layer3(x_2)  # 1/8, in=128, out=256
        x_4 = self.layer4(x_3)  # 1/32, in=256, out=512

        x_1 = x_1
        x_2 = self.upsamplex2(x_2)
        x_3 = self.upsamplex2(x_3)
        x_4 = self.upsamplex2(x_4)

        # output layers
        x_1 = self.conv1(x_1)
        x_2 = self.conv2(x_2)
        x_3 = self.conv3(x_3)
        x_4 = self.conv4(x_4)

        return x_1,x_2,x_3,x_4


class RAM(ResNet):
    def __init__(self, input_nc, output_nc):
        super(RAM, self).__init__(input_nc, output_nc)
        #
        self.VMoE1 = VMoE(dim=self.dim)
        self.VMoE2 = VMoE(dim=self.dim)
        self.VMoE3 = VMoE(dim=self.dim)
        self.VMoE4 = VMoE(dim=self.dim)

        # self.deform_align1 = DeformableAlignment(self.dim)
        # self.deform_align2 = DeformableAlignment(self.dim)
        # self.deform_align3 = DeformableAlignment(self.dim)
        # self.deform_align34 = DeformableAlignment(self.dim)

        # self.rank_ai1 = RankAIModule(in_channels=self.dim)
        # self.rank_ai2 = RankAIModule(in_channels=self.dim)
        # self.rank_ai3 = RankAIModule(in_channels=self.dim)
        # self.rank_ai4 = RankAIModule(in_channels=self.dim)


    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1,x1_2,x1_3,x1_4 = self.forward_layer(x1)
        x2_10,x2_2,x2_3,x2_4 = self.forward_layer(x2)

        # x2_10 = self.deform_align1(x1_1, x2_1)
        # x2_2 = self.deform_align2(x1_2, x2_2)
        # x2_3 = self.deform_align3(x1_3, x2_3)
        # x2_4 = self.deform_align3(x1_4, x2_4)

        # # 创建可视化画布
        # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        #
        # # 可视化输入图像
        # visualize_tensor(x1, 'Input x1', axes[0, 0])
        # visualize_tensor(x2, 'Input x2', axes[0, 1])
        #
        # # 可视化特征图
        # visualize_tensor(x1_1, 'x1_1 Feature', axes[1, 2])
        # visualize_tensor(x2_1, 'x2_1 (Before Align)', axes[1, 0])
        # visualize_tensor(x2_10, 'x2_1 (After Align)', axes[1, 1])

        # x_1 = self.rank_ai1(x1_1, x2_10)
        x_1 = x1_1 - x2_10
        # x_2 = self.rank_ai2(x1_2, x2_2)
        x_2 = x1_2 - x2_2
        # x_3 = self.rank_ai3(x1_3, x2_3)
        x_3 = x1_3 - x2_3
        # x_4 = self.rank_ai4(x1_4, x2_4)
        x_4 = x1_4 - x2_4

        # visualize_tensor(x_1, 'x_1 (After RankAI)', axes[0, 2])
        # plt.tight_layout()
        # plt.show()

        x_4 = self.VMoE4(x_4)
        x_3 = self.VMoE3(x_4+x_3)
        x_2 = self.VMoE2(x_3+x_2)
        x = self.VMoE1(x_2+x_1)
        # x = x_1 + x_2 + x_3 + x_4
        x = self.upsamplex4(x)
        x = self.classifier(x)

        return x


# if __name__ == '__main__':
#     # 实例化模型对象
#     model = RAM(input_nc=3, output_nc=2).cuda()
#     # 生成随机输入张量
#     input = torch.randn(1, 3, 512, 512).cuda()
#     # 执行前向传播
#     output = model(input,input)
#     # 打印输入和输出的尺寸
#     print('input_size:', input.size())
#     print('output_size:', output.size())

if __name__ == '__main__':
    # 实例化模型对象
    model = RAM(input_nc=3, output_nc=2).cuda()

    # 生成随机输入张量
    input_tensor1 = torch.randn(1, 3, 256, 256).cuda()
    input_tensor2 = torch.randn(1, 3, 256, 256).cuda()  # 创建第二个输入张量

    # 计时器开始
    start_time = time.time()

    # 执行前向传播
    output = model(input_tensor1, input_tensor2)  # 使用两个输入进行前向传播

    # 计时器结束
    end_time = time.time()

    # 计算推理时间
    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"Inference Time: {inference_time:.2f} ms")

    # 打印模型的参数数量和 FLOPs，确保使用两个输入
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops, params = profile(model, inputs=(input_tensor1, input_tensor2))  # 传递两个输入
    # 计算延迟
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        start_time = time.time()  # 开始时间
        output = model(input_tensor1, input_tensor2)  # 前向传播
        end_time = time.time()  # 结束时间

    # 计算延迟（单位：毫秒）
    latency = (end_time - start_time) * 1000  # 转换为毫秒
    # 计算推理时间（单位：毫秒）
    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"Inference Time: {inference_time:.2f} ms")
    print(f"Latency: {latency:.2f} ms")
    print(f"FLOPs: {flops / 1e9:.2f} GFlops")  # 转换为 GFlops
    print(f"模型参数总量：{total_params}")
    print('input_size:', input_tensor1.size())
    print('output_size:', output.size())
    # 评估内存占用
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为 MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # 转换为 MB

    print(f"Allocated Memory: {allocated_memory:.2f} MB")
    print(f"Reserved Memory: {reserved_memory:.2f} MB")
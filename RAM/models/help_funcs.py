import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


# 可学习的自适应边缘检测模块
class AdaptiveEdgeDetector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        self.edge_conv = nn.Conv2d(32, in_dim, 3, padding=1)

        # 可学习的初始化参数
        self.sobel_x = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(32, 32, 3, padding=1, bias=False)

        # 初始化Sobel参数
        sobel_kernel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32).repeat(32, 32, 1, 1)
        sobel_kernel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32).repeat(32, 32, 1, 1)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=True)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=True)

    def forward(self, x):
        features = self.conv_layers(x)
        dx = self.sobel_x(features)
        dy = self.sobel_y(features)
        edge = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        edge = self.edge_conv(edge)
        return x + edge  # 残差连接


# 双向交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_conv = nn.Conv2d(dim, dim, 1)
        self.k_conv = nn.Conv2d(dim, dim, 1)
        self.v_conv = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        # x1: query分支
        # x2: key/value分支
        b, c, h, w = x1.shape
        q = self.q_conv(x1).view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        k = self.k_conv(x2).view(b, c, -1)  # [B, C, N]
        v = self.v_conv(x2).view(b, c, -1).permute(0, 2, 1)  # [B, N, C]

        attn = torch.bmm(q, k) / (c ** 0.5)
        attn = self.softmax(attn)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(b, c, h, w)
        return out + x1  # 残差连接

# 定义 Rank AI 模块（秩能智控模块）
class RankAIModule(nn.Module):
    '''秩驱动特征智能调控模块，通过对特征的秩驱动排序和动态融合，有效提升特征的重要性评估和自适应能力。'''
    def __init__(self, in_channels=256):
        super().__init__()
        mid_channels = in_channels * 2
        self.cross_attn = CrossAttention(in_channels)
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # 多尺度特征提取
        self.multi_scale = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, dilation=2, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, 1)
        )

        # 动态权重融合
        self.weight_gate = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 特征差异
        cross_feat1 = self.cross_attn(x1, x2)
        cross_feat2 = self.cross_attn(x2, x1)

        x = torch.abs(cross_feat1 - cross_feat2)

        # 通道权重 [1,32,1,1]
        c_weights = self.channel_att(x)

        # 空间权重 [1,1,64,64]
        s_weights = self.spatial_att(x)

        # 扩展通道权重至空间维度
        c_weights_expanded = c_weights.expand_as(x)  # [1,32,64,64]

        # 拼接特征生成动态权重
        weight_input = torch.cat([c_weights_expanded, s_weights], dim=1)  # [1,33,64,64]
        combined_weights = self.weight_gate(weight_input)  # [1,1,64,64]

        # 多尺度特征融合
        ms_features = self.multi_scale(x)  # [1,32,64,64]

        # 加权特征合成
        weighted_x = x * combined_weights  # [1,32,64,64]
        output = weighted_x + ms_features  # [1,32,64,64]

        # 残差连接
        output = output + x  # [1,32,64,64]

        return output

class DeformableAlignment(nn.Module):
    """可变形特征对齐模块"""

    def __init__(self, dim, stride=1):
        super().__init__()
        self.stride = stride
        self.conv_offset = nn.Conv2d(dim, 3 * 3 * 2, 3, padding=1)
        self.conv_mask = nn.Conv2d(dim, 3 * 3, 3, padding=1)
        self.conv_feat = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, feat1, feat2):
        # 生成偏移量和掩码
        offset = self.conv_offset(feat1)  # [b, 18, h, w]
        mask = torch.sigmoid(self.conv_mask(feat1))  # [b, 9, h, w]

        # 可变形卷积对齐
        aligned_feat = torchvision.ops.deform_conv2d(
            input=feat2,
            offset=offset,
            mask=mask,
            weight=self.conv_feat.weight,
            bias=self.conv_feat.bias,
            stride=self.stride,
            padding=1
        )
        return aligned_feat + feat1  # 残差连接

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_mb = total_params * 4 / (1024 ** 2)  # 从参数总数转换为MB
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Parameters Size (MB): {total_params_mb:.2f}")
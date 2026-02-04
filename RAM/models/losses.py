import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_loss(input, target, weight=None, ignore_index=255):
    """
    组合损失函数，包含：
    1. 交叉熵损失
    2. 边界聚焦损失
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    # 自适应尺寸对齐
    if input.shape[-2:] != target.shape[-2:]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    # 1. 交叉熵损失
    ce_loss = F.cross_entropy(input, target, weight=weight,
                              ignore_index=ignore_index, reduction='mean')

    # 2. 边界聚焦损失
    boundary_loss = edge_focal_loss(input, target, ignore_index)

    # 加权组合
    total_loss = ce_loss +  boundary_loss

    return total_loss

def edge_focal_loss(input, target, ignore_index=255, beta=0.7, gamma=2):
    # Sobel边缘检测核
    sobel_kernel = torch.tensor([
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],  # 水平梯度
        [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]  # 垂直梯度
    ], dtype=torch.float32, device=input.device)

    # 生成边缘掩码
    with torch.no_grad():
        # 创建有效掩码并提取变化区域
        valid_mask = (target != ignore_index).float()
        edge_target = target.clone().float() * valid_mask

        # 计算梯度幅值
        gradients = F.conv2d(edge_target.unsqueeze(1), sobel_kernel, padding=1)
        edge_magnitude = torch.sqrt(gradients[:, 0] ** 2 + gradients[:, 1] ** 2 + 1e-8)
        edge_mask = (edge_magnitude > 0.1 * edge_magnitude.max()).float()

    # 计算焦点权重
    prob = F.softmax(input, dim=1)
    pt = torch.gather(prob, 1, target.unsqueeze(1))  # 获取正确类别的概率
    focal_weight = (beta * pt + (1 - beta) * (1 - pt)).pow(gamma)

    # 边界区域加权
    weighted_loss = F.cross_entropy(input, target,
                                    weight=None,
                                    ignore_index=ignore_index,
                                    reduction='none') * focal_weight.squeeze(1)

    # 增强边界区域权重
    return (weighted_loss * (1 + 2 * edge_mask)).mean()


# 交叉熵函数
def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

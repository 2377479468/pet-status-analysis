"""
模型定义模块
路径：modules/emotion/model.py
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(model_name: str = "mobilenet_v2",
                 num_classes: int = 4,
                 pretrained: bool = True):
    """
    创建分类模型（迁移学习）

    参数:
        model_name: "mobilenet_v2" 或 "resnet18"
        num_classes: 分类类别数（默认4）
        pretrained: 是否使用预训练权重

    返回:
        model: PyTorch 模型
    """
    if model_name == "mobilenet_v2":
        # MobileNetV2：轻量级，适合快速训练和部署
        model = models.mobilenet_v2(weights="DEFAULT" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        # ResNet18：残差网络，特征提取能力强
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"不支持的模型: {model_name}，可选 mobilenet_v2 / resnet18")

    return model


def load_model(model_path: str, model_name: str = "mobilenet_v2",
               num_classes: int = 4, device: str = "cpu"):
    """
    加载训练好的模型权重

    参数:
        model_path: 模型权重文件路径 (.pth)
        model_name: 模型架构名称
        num_classes: 类别数
        device: 推理设备

    返回:
        model: 加载了权重的模型（eval模式）
    """
    model = create_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
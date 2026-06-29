"""
推理接口模块 — 供 A 的 main.py 调用
路径：modules/emotion/emotion_predictor.py

A 模块调用方式：
    from modules.emotion.emotion_predictor import predict_emotion
    result = predict_emotion("path/to/crop.jpg")
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============ 配置（根据实际情况修改） ============
# 模型权重文件路径（相对于本文件所在的 modules/emotion 目录）
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "emotion_model.pth"
# 模型架构
DEFAULT_MODEL_NAME = "mobilenet_v2"
# 类别标签
CLASSES = ["angry", "anxious", "happy", "relaxed"]
# 推理设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片预处理（与验证时保持一致）
IMG_SIZE = 224
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 全局模型缓存，避免每次调用都重新加载
_model = None


def _load_model(model_path=None, model_name=None):
    """加载模型（单例模式，只加载一次）"""
    global _model
    if _model is not None:
        return _model

    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    model_name = model_name or DEFAULT_MODEL_NAME

    # 导入同包内的 model 模块
    from modules.emotion.model import create_model

    _model = create_model(model_name, num_classes=len(CLASSES), pretrained=False)
    _model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()
    return _model


def predict_emotion(image_path: str,
                    model_path: str = None,
                    model_name: str = None) -> dict:
    """
    预测宠物图片的情绪/状态概率

    这是供 A 模块调用的核心函数！

    参数:
        image_path: 宠物裁剪图片路径（B模块输出的crop.jpg）
        model_path: 模型权重路径（可选，默认使用训练好的模型）
        model_name: 模型架构名称（可选）

    返回:
        dict: 四类状态概率，如 {"happy": 0.72, "angry": 0.08, ...}
    """
    # 1. 加载模型
    model = _load_model(model_path, model_name)

    # 2. 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    # 3. 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    # 4. 构建返回字典
    result = {}
    for i, class_name in enumerate(CLASSES):
        result[class_name] = round(float(probs[i]), 4)

    return result


# ============ 模型未训练时的占位函数 ============
def predict_emotion_fake(image_path: str = None) -> dict:
    """
    假数据占位函数 — 模型未训练好时使用
    返回固定概率，保证系统流程能跑通
    """
    return {
        "happy": 0.72,
        "angry": 0.08,
        "relaxed": 0.15,
        "anxious": 0.05
    }


if __name__ == "__main__":
    # 本地测试
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # 找一张测试图片
        train_dir = Path(__file__).parent.parent / "data" / "train"
        imgs = list(train_dir.rglob("*.jpg"))
        img_path = str(imgs[0]) if imgs else None

    if img_path:
        result = predict_emotion(img_path)
        print(f"图片: {img_path}")
        print(f"预测结果: {result}")
    else:
        print("未找到测试图片，使用假数据演示:")
        print(predict_emotion_fake())
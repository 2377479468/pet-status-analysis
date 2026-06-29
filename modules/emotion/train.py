"""
C 模块：宠物情绪/状态分类模型 —— 训练脚本（迁移学习）

从项目根目录运行：
    python -m modules.emotion.train

数据集目录结构（ImageFolder 格式，按类别分子文件夹）：
    modules/emotion/data/
    ├── train/
    │   ├── angry/    *.jpg
    │   ├── anxious/  *.jpg
    │   ├── happy/    *.jpg
    │   └── relaxed/  *.jpg
    └── val/
        ├── angry/ ...  (同上)

训练完成后产出：
    modules/emotion/models/emotion_model.pth   —— 供 emotion_predictor.py 加载
    modules/emotion/results/accuracy_curve.png —— 训练/验证准确率与 loss 曲线
    modules/emotion/results/confusion_matrix.png —— 验证集混淆矩阵
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面环境也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.emotion.model import create_model

# ============ 配置区（按需修改）============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"               # 数据集根目录（含 train/ 与 val/）
MODEL_OUT = BASE_DIR / "models" / "emotion_model.pth"
RESULTS_DIR = BASE_DIR / "results"

# 必须与 emotion_predictor.CLASSES 一致（ImageFolder 会按字母序排，正好对应）
CLASSES = ["angry", "anxious", "happy", "relaxed"]

MODEL_NAME = "mobilenet_v2"   # 或 "resnet18"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet 归一化（与 emotion_predictor.py 推理时保持一致）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# ============ 数据增强 / 预处理 ============
# 训练集做随机增强提升泛化；验证集只 resize + 归一化，保证评估稳定，
# 且验证 transform 必须与推理 transform 完全一致，否则训练/部署表现会脱节。
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def build_loaders():
    train_set = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_transform)
    val_set = datasets.ImageFolder(str(DATA_DIR / "val"), transform=eval_transform)

    # 关键一致性校验：训练标签顺序必须与推理 CLASSES 一致，否则概率会错位
    if train_set.classes != CLASSES:
        raise ValueError(
            f"数据集类别 {train_set.classes} 与期望 {CLASSES} 不一致！\n"
            f"请确保 train/ 下的子文件夹名恰好为 {CLASSES}（ImageFolder 按字母序映射索引）。"
        )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None):
    """跑一个 epoch。传 optimizer 则为训练模式，否则为评估模式。返回 (avg_loss, accuracy, 预测, 真值)。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_preds, all_labels


def plot_curves(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_acc"], label="train")
    ax1.plot(epochs, history["val_acc"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()

    ax2.plot(epochs, history["train_loss"], label="train")
    ax2.plot(epochs, history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=120)
    plt.close(fig)


def plot_confusion(preds, labels):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"未找到数据集目录：{DATA_DIR}\n"
            f"请按 README/本文件顶部说明放置 train/ 与 val/ 数据后再训练。"
        )

    train_loader, val_loader = build_loaders()
    model = create_model(MODEL_NAME, num_classes=len(CLASSES), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0
    best_preds, best_labels = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_acc={tr_acc:.3f} loss={tr_loss:.3f} | "
              f"val_acc={val_acc:.3f} loss={val_loss:.3f}")

        # 保存验证集表现最好的权重
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_preds, best_labels = val_preds, val_labels
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_OUT))
            print(f"    ↳ 保存最佳模型 (val_acc={val_acc:.3f}) -> {MODEL_OUT}")

    plot_curves(history)
    if best_preds:
        plot_confusion(np.array(best_preds), np.array(best_labels))

    print(f"\n训练完成！最佳验证准确率：{best_val_acc:.3f}")
    print(f"权重：{MODEL_OUT}")
    print(f"曲线/混淆矩阵：{RESULTS_DIR}")


if __name__ == "__main__":
    main()

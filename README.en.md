# Pet Status Analyzer · 宠物状态分析系统

![CI](https://github.com/2377479468/pet-status-analysis/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-detection%20%2B%20pose-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

📖 [简体中文](README.md) | **English**

Upload a short pet video or image → **YOLOv8** detects cats/dogs → auto-selects the best frame → **dog-pose** estimation & **MobileNetV2** emotion classification → decision-level fusion into a **six-dimensional status index** → an **ECharts** radar chart with playful captions. A complete computer-vision / pattern-recognition pipeline with a FastAPI backend, same-origin frontend, and **fully offline-capable demo**.

> ⚠️ The system outputs a **status index** (built from visual features, motion cues and classifier probabilities) for playful, visual analysis — it is **not an absolute judgment of a pet's real emotions**.

---

## ✨ Features

- 🎯 **Object detection** — YOLOv8 detects cat/dog, returning bounding box, class and confidence
- 🖼️ **Best-frame selection** — weighted scoring over confidence + sharpness (Laplacian variance) + centering + area ratio
- 🦴 **Pose & behavior** — dog-pose keypoint regression → static pose (sit/lie/stand) + dynamic behavior timeline from video
- 😀 **Emotion classification** — transfer-learned MobileNetV2 → happy / angry / relaxed / anxious probabilities
- 📊 **Six-dimensional status index** — decision-level fusion → happiness / activity / anger / confusion / alertness / good-dog
- 🌐 **Same-origin access + offline self-contained** — works over LAN from a phone; ECharts and demo data are bundled locally

## 🖼️ Showcase

| Detection · Best frame | Pose keypoints (val prediction) |
| --- | --- |
| ![detection](docs/showcase_detection.jpg) | ![pose](docs/showcase_pose.jpg) |

| Emotion training curves | Confusion matrix |
| --- | --- |
| ![accuracy](docs/showcase_accuracy.png) | ![confusion](docs/showcase_confusion.png) |

## 🏗️ Architecture

![architecture](docs/architecture.svg)

> Decoupled modules: the five algorithmic stages are orchestrated by the backend; the frontend talks to a single `/analyze` endpoint.

## 🧠 Pipeline

| Stage | Technique | Input → Output |
| --- | --- | --- |
| Detection | YOLOv8 (single-stage + NMS) | sampled frame → box + class + confidence |
| Best frame | multi-criteria weighted score | candidate frames → best frame + cropped subject |
| Pose / behavior | YOLOv8-pose keypoint regression | frame/video → keypoints + pose + behavior timeline |
| Emotion | MobileNetV2 transfer learning | cropped subject → four-class probabilities |
| Fusion | rule-based weighting (decision level) | probabilities + motion + quality + confidence → six-dim index |

## 📦 Project Layout

```
pet_project/
├── main.py                  # FastAPI backend, unified /analyze endpoint
├── requirements.txt
├── modules/
│   ├── detection/           # sampling / YOLO / best-frame / dog-pose
│   │   ├── pet_analyzer.py      #   entry point analyze_pet()
│   │   ├── video_process.py     #   sampling / detection / best-frame / crop / motion
│   │   └── dog_pose/            #   keypoint pose + multi-frame behavior
│   ├── emotion/             # transfer-learning classifier
│   │   ├── emotion_predictor.py / model.py / train.py
│   │   ├── models/              #   emotion_model.pth (see weights below)
│   │   └── results/            #   accuracy curve / confusion matrix
│   └── scoring/            # six-dim index + captions
│       ├── score_calculator.py
│       └── text_generator.py
├── static/                 # frontend: index.html / app.js / style.css (bundled offline ECharts + demo images)
├── weights/                # YOLO detection weights (auto-downloaded at runtime)
└── uploads/                # runtime uploads
```

## 🚀 Quick Start

```bash
conda create -n petdemo python=3.10
conda activate petdemo
pip install -r requirements.txt
python main.py            # if port is taken: PORT=8080 python main.py
```

- Local: `http://localhost:8000/static/index.html`
- Phone (same Wi-Fi): `http://<your-LAN-IP>:8000/static/index.html` (frontend uses `window.location.origin`; open the firewall port if needed)

> **WinError 10013 / port in use?** Windows sometimes reserves port 8000 (check with `netsh interface ipv4 show excludedportrange protocol=tcp`). Just switch ports with `PORT=8080 python main.py` — no code change needed.

## 🔌 `/analyze` API

`POST /analyze` with form field `file` (image or video) returns JSON:

```json
{
  "success": true,
  "animal": "dog",
  "animal_confidence": 0.93,
  "best_frame_url": "/static/results/xxx_best_frame.jpg",
  "emotion_probs": { "happy": 0.72, "angry": 0.08, "relaxed": 0.15, "anxious": 0.05 },
  "scores": {
    "开心指数": 82, "活跃指数": 72, "生气指数": 9,
    "疑惑指数": 31, "警觉指数": 27, "好狗指数": 88
  },
  "comment": "这是一只状态不错的狗狗，开心指数较高。"
}
```

Video input additionally returns `video_behavior` / `static_pose` / `detection_video_url`.

## 🏋️ Training & Weights

To keep the repo lean, large weights (`*.pt` / `*.pth`) are not tracked in Git — download them from [**Releases**](../../releases), or train your own:

| Weight | Location | How to obtain |
| --- | --- | --- |
| YOLOv8 detection `yolov8n.pt` | `weights/` | auto-downloaded by `ultralytics` on first run |
| dog-pose `best.pt` | `modules/detection/dog_pose/weights/` | from [Releases](../../releases), or train via `train_dog_pose.py` |
| Emotion `emotion_model.pth` | `modules/emotion/models/` | from [Releases](../../releases), or `python -m modules.emotion.train` |

> Without `emotion_model.pth` the system still runs — the backend falls back to placeholder probabilities so the full flow and demo keep working.
> dog-pose dataset reference: [Ultralytics dog-pose](https://docs.ultralytics.com/datasets/pose/dog-pose/).

## 🧩 Module Self-tests

```bash
python -m modules.scoring.score_calculator    # six-dim index
python -m modules.scoring.text_generator      # captions
python -m modules.emotion.test_c_e            # emotion→index chain (placeholder probs)
python -m modules.detection.pet_analyzer modules/detection/dog_pose/test_images/dog6.jpg
```

## 🗺️ Roadmap

- Temporal models (LSTM / TCN / ST-GCN) for more robust dynamic behavior recognition
- More emotion & keypoint data to improve robustness under occlusion / back views / blur
- Lightweight edge & mobile deployment

## 📄 License

[MIT](LICENSE)

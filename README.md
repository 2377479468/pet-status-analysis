# 宠物状态分析系统（模式识别课程大作业）

电脑/手机上传宠物短视频或图片 → 后端用 **YOLO** 检测猫狗 → 自动选最佳帧 →
**情绪/状态模型**预测概率 → 计算**六维状态指数** → 前端用 **ECharts 雷达图**展示结果与趣味文案。

> 说明：本系统输出的是**宠物状态指数**（基于视觉特征、运动信息和分类模型概率构建），
> 用于趣味化分析与可视化展示，**不是对宠物真实心理情绪的绝对判断**。

---

## 一、目录结构

```
pet_project/
├── main.py                  # A：FastAPI 后端，统一接口 /analyze，整合 B/C/E
├── requirements.txt         # 统一依赖
├── README.md                # 本文件
├── modules/
│   ├── detection/           # B：YOLO 猫狗检测 + 视频抽帧 + 最佳帧 + 狗姿态/行为
│   │   ├── pet_analyzer.py      #   B 的统一入口 analyze_pet()
│   │   ├── video_process.py     #   抽帧 / YOLO 检测 / 最佳帧 / 裁剪 / 运动分数
│   │   └── dog_pose/            #   狗静态姿态 + 多帧动态行为分析（含 weights/best.pt）
│   ├── emotion/             # C：情绪/状态分类模型
│   │   ├── emotion_predictor.py #   推理入口 predict_emotion()
│   │   ├── model.py             #   模型结构（迁移学习 mobilenet_v2 / resnet18）
│   │   ├── train.py             #   训练脚本
│   │   ├── models/              #   emotion_model.pth 权重
│   │   └── results/             #   准确率曲线 / 混淆矩阵
│   └── scoring/            # E：六维指数 + 文案
│       ├── score_calculator.py  #   calculate_scores()
│       └── text_generator.py    #   generate_comment()
├── static/                 # D：前端页面（index.html / app.js / style.css）
│   └── results/                 #   后端生成的最佳帧/检测图（运行时）
├── uploads/                # 运行时上传文件
└── weights/                # YOLO 检测权重（yolov8n.pt / yolov8m.pt）
```

## 二、环境搭建（统一 Python 3.10）

```bash
conda create -n petdemo python=3.10
conda activate petdemo
pip install -r requirements.txt
```

### 模型权重（仓库未包含，需自行获取）

为保持仓库轻量，体积较大的模型权重（`*.pt` / `*.pth`）已通过 `.gitignore` 排除，请按下表准备：

| 权重 | 位置 | 获取方式 |
| --- | --- | --- |
| YOLOv8 检测 `yolov8n.pt` | `weights/` | 首次运行由 `ultralytics` **自动联网下载**，无需手动准备 |
| dog-pose 姿态 `best.pt` | `modules/detection/dog_pose/weights/` | 运行 `modules/detection/dog_pose/train_dog_pose.py` 训练，或单独下载提供 |
| 情绪模型 `emotion_model.pth` | `modules/emotion/models/` | 运行 `python -m modules.emotion.train` 训练得到 |

> 缺少 `emotion_model.pth` 时系统不会崩溃：后端会自动回退为占位概率，仍可跑通完整流程与前端演示。
> dog-pose 数据集参考 [Ultralytics dog-pose](https://docs.ultralytics.com/datasets/pose/dog-pose/)。

## 三、启动后端

**必须在项目根目录（本 README 所在目录）启动**，因为代码使用相对路径访问 `static/`、`uploads/`：

```bash
python main.py
# 等价于：uvicorn main:app --host 0.0.0.0 --port 8000
```

启动后：
- 后端接口：`POST http://<电脑IP>:8000/analyze`
- 前端页面：`http://<电脑IP>:8000/static/index.html`

> **端口被占用 / WinError 10013？** Windows 有时把 8000 端口保留（可用
> `netsh interface ipv4 show excludedportrange protocol=tcp` 查看）。换个端口即可，
> 无需改代码（前端用 `window.location.origin` 自动适配）：
> ```powershell
> $env:PORT=8080; python main.py        # PowerShell
> set PORT=8080 && python main.py        # CMD
> ```
> 然后访问 `http://<电脑IP>:8080/static/index.html`。

## 四、电脑 / 手机访问

1. 电脑浏览器直接打开 `http://localhost:8000/static/index.html`。
2. 手机与电脑连同一 Wi-Fi，浏览器打开 `http://<电脑局域网IP>:8000/static/index.html`
   （电脑 IP 用 `ipconfig` 查看，如 `192.168.1.5`）。
   前端用 `window.location.origin` 自动指向同一台后端，无需改代码。
3. 若手机无法访问，检查 Windows 防火墙是否放行 8000 端口。

## 五、接口契约 `/analyze`

`POST /analyze`，表单字段 `file`（图片或视频），返回 JSON：

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
（视频输入还会额外返回 `video_behavior` / `static_pose` / `detection_video_url` 等字段供前端展示。）

## 六、备用演示方案（答辩用）

- 前端页面有「**使用演示数据**」按钮：不依赖后端即可展示雷达图/六维指数/文案，
  防止现场网络不通或模型太慢。点击后会从内置的 3 套数据（活泼狗 / 慵懒狗 / 猫咪）
  中随机选一套渲染。
- **完全离线自包含**：演示所需的全部资源都在本地，断网也能正常展示：
  - 图表库 ECharts → `static/echarts.min.js`（[index.html](static/index.html) 本地优先加载，
    本地缺失时才回退 CDN）。
  - 演示图片 → `static/demo/`（`dog_active*.jpg` / `dog_lazy*.jpg` / `cat.jpg`）。
  - 假数据定义在 [static/app.js](static/app.js) 顶部的 `MOCK_DOG_ACTIVE` / `MOCK_DOG_LAZY` /
    `MOCK_CAT`，图片路径均指向上面的本地文件（**不要再改回 placedog/placekitten 等外链**，
    那些服务已不可用）。
- 建议另备：1 个狗视频、1 个猫视频、1 张已跑好的结果截图、1 段录屏。

> 修改了 `static/` 下的 `app.js` / `index.html` 后，浏览器需 **强制刷新（Ctrl+F5）** 才会生效
> （否则会用缓存的旧脚本）。

## 七、训练情绪模型（C）

数据集按类别分文件夹放好后运行：

```bash
python -m modules.emotion.train
```

详见 `modules/emotion/train.py` 顶部的配置区（数据路径、类别、超参）。

## 八、各模块独立自测

```bash
python -m modules.scoring.score_calculator      # E：六维指数
python -m modules.scoring.text_generator        # E：文案
python -m modules.emotion.test_c_e              # C→E 链路（假概率）
python -m modules.detection.pet_analyzer modules/detection/dog_pose/test_images/dog6.jpg  # B
```

## 九、未包含项

- E 的**实验报告 / PPT**为文档类交付物，不在本仓库代码范围内。
- 情绪模型的进一步调优（更多数据/类别）由 C 持续迭代，当前权重保证流程可跑通。

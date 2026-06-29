# -*- coding: utf-8 -*-
"""
predict_dog_pose.py

狗狗姿态预测入口。

职责：
1. 对单张图片运行 dog-pose 模型。
2. 输出标准化狗狗静态姿态结果。
3. 支持图片、视频最佳帧、视频抽样帧复用。
4. 不输出视频动态行为：
   - moving
   - approaching
   - leaving
   - shaking
   - turning
   - stretching

视频动态行为后续由 video_behavior_analyzer.py 根据多帧结果分析。
"""

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .dog_pose_analyzer import (
    POSE_CN,
    calc_pose_confidence,
    classify_pose_image,
    draw_analysis,
    filter_image_actions,
)

# =========================
# 1. 配置
# =========================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "dog_pose"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOG_POSE_MODEL_PATHS = [
    BASE_DIR / "weights" / "best.pt",
    BASE_DIR / "models" / "dog_pose.pt",
    BASE_DIR / "model" / "dog_pose.pt",
    BASE_DIR / "dog_pose.pt",
    BASE_DIR / "runs" / "pose" / "train" / "weights" / "best.pt",
]

DOG_CLASS_NAMES = {
    "dog",
    "狗",
}

KPT_NAMES = [
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "left_ear_base",
    "right_ear_base",
    "left_ear_tip",
    "right_ear_tip",
    "nose",
    "chin",
]

_MODEL = None
_MODEL_LOAD_ERROR = None

# =========================
# 2. 基础工具
# =========================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default

def _round_float(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0

def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return []

def _image_stem(image_path: Any) -> str:
    try:
        return Path(str(image_path)).stem or "image"
    except Exception:
        return "image"

def _make_output_paths(image_path: Any, output_dir: Optional[Any] = None) -> Tuple[str, str]:
    output_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    stem = _image_stem(image_path)

    analysis_image_path = str(output_root / f"{stem}_dog_pose.jpg")
    json_path = str(output_root / f"{stem}_dog_pose.json")

    return analysis_image_path, json_path

def _normalize_keypoints_for_video(keypoints: Any) -> List[Dict[str, Any]]:
    """
    把 dog-pose 关键点统一转成视频绘制友好的列表格式。

    输出格式：
    [
        {
            "name": "nose",
            "x": 123.4,
            "y": 56.7,
            "confidence": 0.92
        }
    ]

    兼容格式：
    - {"nose": {"xy": [x, y], "conf": c}}
    - {"nose": {"x": x, "y": y, "confidence": c}}
    - {"nose": [x, y, c]}
    - [[x, y, c], ...]
    """
    if not keypoints:
        return []

    points = []

    if isinstance(keypoints, dict):
        for name, value in keypoints.items():
            x = None
            y = None
            confidence = 1.0

            if isinstance(value, dict):
                if isinstance(value.get("xy"), (list, tuple)) and len(value.get("xy")) >= 2:
                    x = value.get("xy")[0]
                    y = value.get("xy")[1]
                else:
                    x = value.get("x")
                    y = value.get("y")

                confidence = value.get(
                    "confidence",
                    value.get("conf", value.get("score", 1.0)),
                )

            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                x = value[0]
                y = value[1]
                confidence = value[2] if len(value) >= 3 else 1.0

            x = _safe_float(x, -1.0)
            y = _safe_float(y, -1.0)
            confidence = _safe_float(confidence, 1.0)

            if x < 0 or y < 0:
                continue

            points.append({
                "name": str(name),
                "x": _round_float(x, 3),
                "y": _round_float(y, 3),
                "confidence": _round_float(confidence, 4),
            })

        return points

    if isinstance(keypoints, (list, tuple)):
        for index, value in enumerate(keypoints):
            x = None
            y = None
            confidence = 1.0
            name = str(index)

            if isinstance(value, dict):
                name = str(value.get("name", index))

                if isinstance(value.get("xy"), (list, tuple)) and len(value.get("xy")) >= 2:
                    x = value.get("xy")[0]
                    y = value.get("xy")[1]
                else:
                    x = value.get("x")
                    y = value.get("y")

                confidence = value.get(
                    "confidence",
                    value.get("conf", value.get("score", 1.0)),
                )

            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                x = value[0]
                y = value[1]
                confidence = value[2] if len(value) >= 3 else 1.0

            x = _safe_float(x, -1.0)
            y = _safe_float(y, -1.0)
            confidence = _safe_float(confidence, 1.0)

            if x < 0 or y < 0:
                continue

            points.append({
                "name": name,
                "x": _round_float(x, 3),
                "y": _round_float(y, 3),
                "confidence": _round_float(confidence, 4),
            })

    return points

def _empty_result(
    image_path: Any,
    message: str,
    enabled: bool = True,
    pose_label: str = "unknown",
    raw_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pose_label_cn = POSE_CN.get(pose_label, "未知")

    return {
        "enabled": enabled,
        "animal": "dog",
        "pose_label": pose_label,
        "pose_label_cn": pose_label_cn,
        "pose_actions": [],
        "pose_confidence": 0.0,
        "keypoints": {},
        "keypoints_list": [],
        "points": [],
        "keypoint_count": 0,
        "box": [],
        "box_int": [],
        "dog_count": 0,
        "analysis_image_path": None,
        "json_path": None,
        "message": message,
        "raw_result": raw_result or {},
        "source_image_path": str(image_path) if image_path is not None else None,
    }

def _save_json(data: Dict[str, Any], json_path: str) -> None:
    save_dir = os.path.dirname(json_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# =========================
# 3. 模型加载
# =========================

def _find_model_path() -> Optional[str]:
    for path in DOG_POSE_MODEL_PATHS:
        if path.exists():
            return str(path)

    env_path = os.environ.get("DOG_POSE_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    return None

def load_dog_pose_model():
    """
    加载 dog-pose 模型。

    优先查找：
    - weights/best.pt
    - models/dog_pose.pt
    - model/dog_pose.pt
    - dog_pose.pt
    - runs/pose/train/weights/best.pt
    - 环境变量 DOG_POSE_MODEL_PATH
    """
    global _MODEL
    global _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL, None

    if _MODEL_LOAD_ERROR:
        return None, _MODEL_LOAD_ERROR

    model_path = _find_model_path()

    if not model_path:
        _MODEL_LOAD_ERROR = (
            "未找到 dog-pose 模型文件。请将模型放到 "
            "weights/best.pt、models/dog_pose.pt，或设置环境变量 DOG_POSE_MODEL_PATH。"
        )
        return None, _MODEL_LOAD_ERROR

    try:
        from ultralytics import YOLO

        _MODEL = YOLO(model_path)
        return _MODEL, None

    except Exception as exc:
        _MODEL_LOAD_ERROR = f"加载 dog-pose 模型失败：{exc}"
        return None, _MODEL_LOAD_ERROR

# =========================
# 4. YOLO 输出解析
# =========================

def _get_class_name(result: Any, class_id: int) -> str:
    names = getattr(result, "names", None)

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)

def _extract_boxes(result: Any) -> List[Dict[str, Any]]:
    boxes_obj = getattr(result, "boxes", None)

    if boxes_obj is None:
        return []

    xyxy_list = _to_list(getattr(boxes_obj, "xyxy", None))
    conf_list = _to_list(getattr(boxes_obj, "conf", None))
    cls_list = _to_list(getattr(boxes_obj, "cls", None))

    boxes = []

    for index, xyxy in enumerate(xyxy_list):
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            continue

        box = [
            _safe_float(xyxy[0]),
            _safe_float(xyxy[1]),
            _safe_float(xyxy[2]),
            _safe_float(xyxy[3]),
        ]

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        confidence = _safe_float(conf_list[index], 0.0) if index < len(conf_list) else 0.0
        class_id = _safe_int(cls_list[index], 0) if index < len(cls_list) else 0

        boxes.append({
            "index": index,
            "box": box,
            "box_int": [
                _safe_int(round(box[0])),
                _safe_int(round(box[1])),
                _safe_int(round(box[2])),
                _safe_int(round(box[3])),
            ],
            "det_confidence": _round_float(confidence, 4),
            "class_id": class_id,
            "class_name": _get_class_name(result, class_id),
        })

    return boxes

def _extract_keypoints_for_index(result: Any, index: int) -> Dict[str, Dict[str, Any]]:
    keypoints_obj = getattr(result, "keypoints", None)

    if keypoints_obj is None:
        return {}

    xy_all = _to_list(getattr(keypoints_obj, "xy", None))
    conf_all = _to_list(getattr(keypoints_obj, "conf", None))

    if not xy_all or index >= len(xy_all):
        return {}

    xy_items = xy_all[index] or []

    if conf_all and index < len(conf_all):
        conf_items = conf_all[index] or []
    else:
        conf_items = []

    keypoints = {}

    for kpt_index, xy in enumerate(xy_items):
        if kpt_index >= len(KPT_NAMES):
            name = f"kpt_{kpt_index}"
        else:
            name = KPT_NAMES[kpt_index]

        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue

        x = _safe_float(xy[0], 0.0)
        y = _safe_float(xy[1], 0.0)

        if x <= 0 and y <= 0:
            continue

        confidence = 1.0
        if kpt_index < len(conf_items):
            confidence = _safe_float(conf_items[kpt_index], 1.0)

        keypoints[name] = {
            "xy": [_round_float(x, 3), _round_float(y, 3)],
            "conf": _round_float(confidence, 4),
            "x": _round_float(x, 3),
            "y": _round_float(y, 3),
            "confidence": _round_float(confidence, 4),
        }

    return keypoints

def _is_dog_detection(item: Dict[str, Any]) -> bool:
    class_name = str(item.get("class_name", "")).lower()

    if class_name in DOG_CLASS_NAMES:
        return True

    class_id = item.get("class_id")

    if class_id == 0:
        return True

    return False

def _parse_yolo_result(result: Any) -> List[Dict[str, Any]]:
    box_items = _extract_boxes(result)
    dogs = []

    for item in box_items:
        if not _is_dog_detection(item):
            continue

        index = item.get("index", 0)
        keypoints = _extract_keypoints_for_index(result, index)
        keypoints_list = _normalize_keypoints_for_video(keypoints)

        pose_debug = classify_pose_image(
            keypoints,
            confs=None,
            box=item.get("box"),
            return_debug=True,
        )

        pose_label = pose_debug.get("pose", "unknown")
        pose_label_cn = pose_debug.get("pose_cn", POSE_CN.get(pose_label, "未知"))
        pose_actions = filter_image_actions(pose_debug.get("actions", []))
        pose_confidence = calc_pose_confidence(keypoints)

        dog = {
            "animal": "dog",
            "box": item.get("box", []),
            "box_int": item.get("box_int", []),
            "det_confidence": item.get("det_confidence", 0.0),
            "class_id": item.get("class_id", 0),
            "class_name": item.get("class_name", "dog"),

            "keypoints": keypoints,
            "keypoints_list": keypoints_list,
            "points": keypoints_list,
            "keypoint_count": len(keypoints_list),

            "pose": pose_label,
            "pose_cn": pose_label_cn,
            "pose_label": pose_label,
            "pose_label_cn": pose_label_cn,
            "pose_actions": pose_actions,
            "pose_confidence": pose_confidence,
            "pose_debug": pose_debug,
        }

        dogs.append(dog)

    dogs.sort(
        key=lambda dog: (
            _safe_float(dog.get("det_confidence"), 0.0),
            _safe_float(dog.get("pose_confidence"), 0.0),
        ),
        reverse=True,
    )

    return dogs

# =========================
# 5. 核心执行函数
# =========================

def run_dog_pose(
    image_path: Any,
    output_dir: Optional[Any] = None,
    save_visualization: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    运行 dog-pose 模型，返回所有狗狗检测结果。

    返回结构：
    {
        "enabled": True,
        "image_path": "...",
        "dogs": [...],
        "dog_count": 1,
        "save_path": "...",
        "json_path": "...",
        "message": "..."
    }
    """
    image_path = str(image_path)
    image_path_obj = Path(image_path)

    if not image_path_obj.exists():
        return {
            "enabled": False,
            "image_path": image_path,
            "dogs": [],
            "dog_count": 0,
            "save_path": None,
            "json_path": None,
            "message": f"图片不存在：{image_path}",
        }

    image = cv2.imread(image_path)

    if image is None:
        return {
            "enabled": False,
            "image_path": image_path,
            "dogs": [],
            "dog_count": 0,
            "save_path": None,
            "json_path": None,
            "message": f"图片读取失败：{image_path}",
        }

    model, error = load_dog_pose_model()

    if error:
        return {
            "enabled": False,
            "image_path": image_path,
            "dogs": [],
            "dog_count": 0,
            "save_path": None,
            "json_path": None,
            "message": error,
        }

    analysis_image_path, json_path = _make_output_paths(image_path, output_dir)

    try:
        results = model(image_path, verbose=False)

        if not results:
            dogs = []
        else:
            dogs = _parse_yolo_result(results[0])

        save_path = None

        if save_visualization:
            save_path = draw_analysis(image, dogs, analysis_image_path)

        result = {
            "enabled": True,
            "image_path": image_path,
            "dogs": dogs,
            "dog_count": len(dogs),
            "save_path": save_path,
            "json_path": json_path if save_json else None,
            "message": "ok" if dogs else "dog-pose 未检测到有效狗姿态",
        }

        if save_json:
            _save_json(result, json_path)

        return result

    except Exception as exc:
        return {
            "enabled": False,
            "image_path": image_path,
            "dogs": [],
            "dog_count": 0,
            "save_path": None,
            "json_path": None,
            "message": f"dog-pose 推理失败：{exc}",
            "traceback": traceback.format_exc(),
        }

# =========================
# 6. 对 pet_analyzer 暴露的函数
# =========================
# =========================
# 5. 对 pet_analyzer / video_process 暴露的函数
# =========================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default

def normalize_box(box):
    if not box or not isinstance(box, (list, tuple)) or len(box) < 4:
        return []

    return [
        safe_int(round(box[0])),
        safe_int(round(box[1])),
        safe_int(round(box[2])),
        safe_int(round(box[3])),
    ]

def normalize_keypoints(keypoints):
    """
    统一 dog-pose 关键点格式。

    输出：
    - keypoints: dict，按 name 索引
    - keypoints_list: list，方便画点和连线
    - points: list，兼容 video_process.py
    """
    keypoints_dict = {}
    keypoints_list = []

    if not keypoints:
        return {}, [], []

    if isinstance(keypoints, dict):
        for name, value in keypoints.items():
            x = None
            y = None
            confidence = 1.0

            if isinstance(value, dict):
                if isinstance(value.get("xy"), (list, tuple)) and len(value.get("xy")) >= 2:
                    x = value.get("xy")[0]
                    y = value.get("xy")[1]
                else:
                    x = value.get("x")
                    y = value.get("y")

                confidence = value.get(
                    "confidence",
                    value.get("conf", value.get("score", 1.0)),
                )

            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                x = value[0]
                y = value[1]
                confidence = value[2] if len(value) >= 3 else 1.0

            if x is None or y is None:
                continue

            point = {
                "name": str(name),
                "x": safe_float(x),
                "y": safe_float(y),
                "confidence": safe_float(confidence, 1.0),
            }

            keypoints_dict[str(name)] = {
                "x": point["x"],
                "y": point["y"],
                "confidence": point["confidence"],
            }
            keypoints_list.append(point)

        return keypoints_dict, keypoints_list, keypoints_list

    if isinstance(keypoints, (list, tuple)):
        for index, value in enumerate(keypoints):
            x = None
            y = None
            confidence = 1.0
            name = str(index)

            if isinstance(value, dict):
                name = str(value.get("name", index))

                if isinstance(value.get("xy"), (list, tuple)) and len(value.get("xy")) >= 2:
                    x = value.get("xy")[0]
                    y = value.get("xy")[1]
                else:
                    x = value.get("x")
                    y = value.get("y")

                confidence = value.get(
                    "confidence",
                    value.get("conf", value.get("score", 1.0)),
                )

            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                x = value[0]
                y = value[1]
                confidence = value[2] if len(value) >= 3 else 1.0

            if x is None or y is None:
                continue

            point = {
                "name": name,
                "x": safe_float(x),
                "y": safe_float(y),
                "confidence": safe_float(confidence, 1.0),
            }

            keypoints_dict[name] = {
                "x": point["x"],
                "y": point["y"],
                "confidence": point["confidence"],
            }
            keypoints_list.append(point)

    return keypoints_dict, keypoints_list, keypoints_list

def get_first_dog(result):
    dogs = result.get("dogs", [])

    if not dogs:
        return None

    if not isinstance(dogs, list):
        return None

    if len(dogs) <= 0:
        return None

    return dogs[0]

def build_empty_pose_result(
    image_path=None,
    frame_index=None,
    timestamp=None,
    raw_result=None,
    message="dog-pose 未检测到有效狗姿态",
):
    raw_result = raw_result or {}

    return {
        "enabled": True,
        "animal": "dog",

        "pose_label": "unknown",
        "pose_label_cn": "未知",
        "pose_actions": [],
        "pose_confidence": 0.0,

        "keypoints": {},
        "keypoints_list": [],
        "points": [],
        "keypoint_count": 0,

        "box": [],
        "box_int": [],

        "dog_count": 0,
        "frame_index": frame_index,
        "time": timestamp,
        "timestamp": timestamp,

        "source_image_path": str(image_path) if image_path else None,
        "analysis_image_path": raw_result.get("save_path"),
        "json_path": raw_result.get("json_path"),

        "message": message,
        "raw_result": raw_result,
    }

def build_pose_result(
    image_path,
    result,
    first_dog,
    frame_index=None,
    timestamp=None,
):
    raw_keypoints = (
        first_dog.get("keypoints")
        or first_dog.get("points")
        or first_dog.get("kpts")
        or first_dog.get("landmarks")
        or {}
    )

    keypoints, keypoints_list, points = normalize_keypoints(raw_keypoints)

    raw_box = (
        first_dog.get("box")
        or first_dog.get("bbox")
        or first_dog.get("xyxy")
        or []
    )

    box_int = normalize_box(raw_box)

    pose_label = (
        first_dog.get("pose_label")
        or first_dog.get("pose")
        or first_dog.get("label")
        or "unknown"
    )

    pose_label_cn = (
        first_dog.get("pose_label_cn")
        or first_dog.get("pose_cn")
        or first_dog.get("label_cn")
        or "未知"
    )

    pose_actions = (
        first_dog.get("pose_actions")
        or first_dog.get("actions")
        or []
    )

    pose_confidence = safe_float(
        first_dog.get(
            "pose_confidence",
            first_dog.get("confidence", first_dog.get("score", 0.0)),
        ),
        0.0,
    )

    dogs = result.get("dogs", [])

    return {
        "enabled": True,
        "animal": "dog",

        "pose_label": pose_label,
        "pose_label_cn": pose_label_cn,
        "pose_actions": pose_actions,
        "pose_confidence": pose_confidence,

        "keypoints": keypoints,
        "keypoints_list": keypoints_list,
        "points": points,
        "keypoint_count": len(keypoints_list),

        "box": box_int,
        "box_int": box_int,

        "dog_count": len(dogs) if isinstance(dogs, list) else 1,
        "frame_index": frame_index,
        "time": timestamp,
        "timestamp": timestamp,

        "source_image_path": str(image_path) if image_path else None,
        "analysis_image_path": result.get("save_path"),
        "json_path": result.get("json_path"),

        "message": result.get("message", "ok"),
        "raw_result": result,
    }

def predict_dog_pose(image_path):
    """
    pet_analyzer.py 会动态加载这个函数。

    返回标准字段：
    - pose_label
    - pose_label_cn
    - pose_actions
    - pose_confidence
    - keypoints
    - keypoints_list
    - points
    - keypoint_count
    - box / box_int
    - dog_count
    """
    try:
        result = run_dog_pose(image_path)
        first_dog = get_first_dog(result)

        if not first_dog:
            return build_empty_pose_result(
                image_path=image_path,
                raw_result=result,
                message=result.get("message", "dog-pose 未检测到有效狗姿态"),
            )

        return build_pose_result(
            image_path=image_path,
            result=result,
            first_dog=first_dog,
        )

    except Exception as exc:
        return {
            "enabled": False,
            "animal": "dog",

            "pose_label": "error",
            "pose_label_cn": "错误",
            "pose_actions": [],
            "pose_confidence": 0.0,

            "keypoints": {},
            "keypoints_list": [],
            "points": [],
            "keypoint_count": 0,

            "box": [],
            "box_int": [],

            "dog_count": 0,
            "frame_index": None,
            "time": None,
            "timestamp": None,

            "source_image_path": str(image_path) if image_path else None,
            "analysis_image_path": None,
            "json_path": None,

            "message": f"dog-pose 预测失败：{exc}",
            "raw_result": None,
        }

def predict_dog_pose_for_video_frame(
    image_path,
    frame_index=None,
    timestamp=None,
    save_visualization=False,
    save_json=False,
):
    """
    video_process.py 调用这个函数。

    注意：
    - image_path 是视频抽出来的临时帧。
    - frame_index / timestamp 必须原样写回结果。
    - save_visualization=False 时，尽量不要额外生成可视化图片，避免标注视频抽帧时太慢。
    """
    try:
        result = run_dog_pose(
            image_path,
            save_visualization=save_visualization,
            save_json=save_json,
        )
        first_dog = get_first_dog(result)

        if not first_dog:
            return build_empty_pose_result(
                image_path=image_path,
                frame_index=frame_index,
                timestamp=timestamp,
                raw_result=result,
                message=result.get("message", "dog-pose 未检测到有效狗姿态"),
            )

        return build_pose_result(
            image_path=image_path,
            result=result,
            first_dog=first_dog,
            frame_index=frame_index,
            timestamp=timestamp,
        )

    except TypeError:
        try:
            result = run_dog_pose(image_path)
            first_dog = get_first_dog(result)

            if not first_dog:
                return build_empty_pose_result(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    raw_result=result,
                    message=result.get("message", "dog-pose 未检测到有效狗姿态"),
                )

            return build_pose_result(
                image_path=image_path,
                result=result,
                first_dog=first_dog,
                frame_index=frame_index,
                timestamp=timestamp,
            )

        except Exception as exc:
            return {
                "enabled": False,
                "animal": "dog",

                "pose_label": "error",
                "pose_label_cn": "错误",
                "pose_actions": [],
                "pose_confidence": 0.0,

                "keypoints": {},
                "keypoints_list": [],
                "points": [],
                "keypoint_count": 0,

                "box": [],
                "box_int": [],

                "dog_count": 0,
                "frame_index": frame_index,
                "time": timestamp,
                "timestamp": timestamp,

                "source_image_path": str(image_path) if image_path else None,
                "analysis_image_path": None,
                "json_path": None,

                "message": f"dog-pose 视频帧预测失败：{exc}",
                "raw_result": None,
            }

    except Exception as exc:
        return {
            "enabled": False,
            "animal": "dog",

            "pose_label": "error",
            "pose_label_cn": "错误",
            "pose_actions": [],
            "pose_confidence": 0.0,

            "keypoints": {},
            "keypoints_list": [],
            "points": [],
            "keypoint_count": 0,

            "box": [],
            "box_int": [],

            "dog_count": 0,
            "frame_index": frame_index,
            "time": timestamp,
            "timestamp": timestamp,

            "source_image_path": str(image_path) if image_path else None,
            "analysis_image_path": None,
            "json_path": None,

            "message": f"dog-pose 视频帧预测失败：{exc}",
            "raw_result": None,
        }
# =========================
# 7. 视频抽样帧专用入口
# =========================

def predict_dog_pose_for_video_frame(
    image_path: Any,
    frame_index: Optional[int] = None,
    timestamp: Optional[float] = None,
    save_visualization: bool = False,
    save_json: bool = False,
) -> Dict[str, Any]:
    """
    给 video_process.py 抽样帧使用。

    和 predict_dog_pose 的区别：
    - 默认不保存可视化图；
    - 默认不保存 json；
    - 返回 frame_index / time；
    - 保留 box + keypoints；
    - 额外返回 keypoints_list / points，方便标注视频绘制关键点。
    """
    try:
        result = run_dog_pose(
            image_path,
            output_dir=DEFAULT_OUTPUT_DIR,
            save_visualization=save_visualization,
            save_json=save_json,
        )

        dogs = result.get("dogs", [])

        if not dogs:
            return {
                "enabled": result.get("enabled", True),
                "frame_index": frame_index,
                "time": timestamp,
                "animal": "dog",
                "pose_label": "unknown",
                "pose_label_cn": "未知",
                "pose_actions": [],
                "pose_confidence": 0.0,
                "keypoints": {},
                "keypoints_list": [],
                "points": [],
                "keypoint_count": 0,
                "box": [],
                "box_int": [],
                "dog_count": 0,
                "message": result.get("message", "dog-pose 未检测到有效狗姿态"),
                "source_image_path": str(image_path),
            }

        first_dog = dogs[0]

        pose_label = first_dog.get("pose_label") or first_dog.get("pose") or "unknown"
        pose_label_cn = (
            first_dog.get("pose_label_cn")
            or first_dog.get("pose_cn")
            or POSE_CN.get(pose_label, "未知")
        )

        pose_actions = filter_image_actions(first_dog.get("pose_actions", []))
        keypoints = first_dog.get("keypoints", {})
        keypoints_list = first_dog.get("keypoints_list") or _normalize_keypoints_for_video(keypoints)

        return {
            "enabled": True,
            "frame_index": frame_index,
            "time": timestamp,

            "animal": "dog",
            "pose_label": pose_label,
            "pose_label_cn": pose_label_cn,
            "pose_actions": pose_actions,
            "pose_confidence": _round_float(first_dog.get("pose_confidence"), 4),

            "keypoints": keypoints,
            "keypoints_list": keypoints_list,
            "points": keypoints_list,
            "keypoint_count": len(keypoints_list),

            "box": first_dog.get("box", []),
            "box_int": first_dog.get("box_int", []),
            "det_confidence": _round_float(first_dog.get("det_confidence"), 4),

            "dog_count": result.get("dog_count", len(dogs)),
            "dogs": dogs,

            "analysis_image_path": result.get("save_path"),
            "json_path": result.get("json_path"),
            "message": result.get("message", "ok"),
            "source_image_path": str(image_path),
        }

    except Exception as exc:
        return {
            "enabled": False,
            "frame_index": frame_index,
            "time": timestamp,
            "animal": "dog",
            "pose_label": "error",
            "pose_label_cn": "错误",
            "pose_actions": [],
            "pose_confidence": 0.0,
            "keypoints": {},
            "keypoints_list": [],
            "points": [],
            "keypoint_count": 0,
            "box": [],
            "box_int": [],
            "dog_count": 0,
            "message": f"predict_dog_pose_for_video_frame 执行失败：{exc}",
            "traceback": traceback.format_exc(),
            "source_image_path": str(image_path) if image_path is not None else None,
        }

# =========================
# 8. 命令行测试
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dog pose prediction on one image.")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化结果")
    parser.add_argument("--no-json", action="store_true", help="不保存 JSON 结果")

    args = parser.parse_args()

    output = run_dog_pose(
        args.image_path,
        save_visualization=not args.no_vis,
        save_json=not args.no_json,
    )

    print(json.dumps(output, ensure_ascii=False, indent=2))
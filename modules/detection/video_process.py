# -*- coding: utf-8 -*-
"""
video_process.py

视频处理模块。

职责：
1. 识别视频中的猫/狗。
2. 抽取最佳帧。
3. 输出裁剪图。
4. 计算基础运动分和画质分。
5. 对狗视频调用 dog-pose 进行动态行为分析。
6. 生成带关键点和骨架线的标注视频。
7. 将 OpenCV 输出视频转码为浏览器兼容 MP4。

说明：
- 动作分析关键点帧 pose_frames 和标注视频关键点帧 annotated_pose_frames 分离。
- 动作分析默认约 5 FPS。
- 标注视频关键点默认约 10 FPS。
- 输出标注视频最高 20 FPS。
"""

import os
import math
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# =========================
# 1. 配置
# =========================

RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)

DETECT_SAMPLE_PER_SECOND = 5

BEHAVIOR_POSE_SAMPLE_PER_SECOND = 5
ANNOTATED_POSE_SAMPLE_PER_SECOND = 10

MAX_BEHAVIOR_POSE_FRAMES = 100
MAX_ANNOTATED_POSE_FRAMES = 120
MIN_BEHAVIOR_POSE_FRAMES = 12

OUTPUT_VIDEO_MAX_FPS = 20

ENABLE_BROWSER_MP4_CONVERT = True
BROWSER_MP4_SUFFIX = "_web"

YOLO_MODEL_PATHS = [
    "yolov8n.pt",
    "weights/yolov8n.pt",
    "models/yolov8n.pt",
    "model/yolov8n.pt",
]

DOG_POSE_EDGES = [
    ("nose", "chin"),
    ("nose", "left_ear_base"),
    ("nose", "right_ear_base"),
    ("left_ear_base", "left_ear_tip"),
    ("right_ear_base", "right_ear_tip"),

    ("front_left_paw", "front_left_knee"),
    ("front_left_knee", "front_left_elbow"),
    ("front_right_paw", "front_right_knee"),
    ("front_right_knee", "front_right_elbow"),

    ("rear_left_paw", "rear_left_knee"),
    ("rear_left_knee", "rear_left_elbow"),
    ("rear_right_paw", "rear_right_knee"),
    ("rear_right_knee", "rear_right_elbow"),
]

ANIMAL_CN = {
    "cat": "猫",
    "dog": "狗",
    "unknown": "未知",
}

_MODEL = None
_MODEL_LOAD_ERROR = None

predict_dog_pose_for_video_frame = None
pose_load_error = None

analyze_video_behavior = None
behavior_load_error = None

try:
    from modules.detection.dog_pose.predict_dog_pose import predict_dog_pose_for_video_frame as _predict_dog_pose_for_video_frame

    predict_dog_pose_for_video_frame = _predict_dog_pose_for_video_frame
except Exception as exc:
    pose_load_error = f"dog-pose 模块加载失败：{exc}"

try:
    from modules.detection.dog_pose.video_behavior_analyzer import analyze_video_behavior as _analyze_video_behavior

    analyze_video_behavior = _analyze_video_behavior
except Exception as exc:
    behavior_load_error = f"视频动态分析模块加载失败：{exc}"

# =========================
# 2. 通用工具
# =========================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default

def normalize_path(path: Any) -> Optional[str]:
    if not path:
        return None

    return str(path).replace("\\", "/")

def to_public_url(path: Any) -> Optional[str]:
    normalized = normalize_path(path)

    if not normalized:
        return None

    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized

    normalized = normalized.lstrip("./")

    if normalized.startswith("/"):
        return normalized

    return "/" + normalized

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def get_video_duration(frame_count: int, fps: float) -> float:
    if fps <= 0:
        return 0.0

    return frame_count / fps

def make_result_paths(video_path: str) -> Dict[str, str]:
    stem = Path(video_path).stem or "video"

    return {
        "best_frame_path": os.path.join(RESULT_DIR, f"{stem}_best_frame.jpg"),
        "crop_path": os.path.join(RESULT_DIR, f"{stem}_crop.jpg"),
        "annotated_video_path": os.path.join(RESULT_DIR, f"{stem}_annotated.mp4"),
    }

def convert_to_browser_mp4(input_path: Any) -> Tuple[Optional[str], bool, str]:
    input_path = normalize_path(input_path)

    if not input_path:
        return None, False, "输入视频路径为空。"

    if not os.path.exists(input_path):
        return input_path, False, "原始标注视频不存在，无法转码。"

    if not ENABLE_BROWSER_MP4_CONVERT:
        return input_path, False, "未启用浏览器兼容 MP4 转码。"

    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        return input_path, False, "未找到 ffmpeg，返回 OpenCV 原始视频。"

    input_file = Path(input_path)
    output_path = str(input_file.with_name(f"{input_file.stem}{BROWSER_MP4_SUFFIX}.mp4"))

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            message = completed.stderr.strip() or "ffmpeg 转码失败。"
            return input_path, False, message[-500:]

        if not os.path.exists(output_path):
            return input_path, False, "ffmpeg 执行完成，但未生成浏览器兼容视频。"

        return output_path, True, "ok"

    except Exception as exc:
        return input_path, False, f"ffmpeg 转码异常：{exc}"

def format_time_label(seconds: Any) -> str:
    seconds = max(0.0, safe_float(seconds))
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    remain_seconds = total_seconds % 60
    return f"{minutes:02d}:{remain_seconds:02d}"

def fallback_behavior_timeline(video_result: Dict[str, Any], duration: float) -> List[Dict[str, Any]]:
    behavior = str(video_result.get("main_behavior") or "unknown")
    behavior_cn = str(video_result.get("main_behavior_cn") or "未知")
    end_time = round(max(0.0, safe_float(duration)), 4)

    if end_time <= 0:
        return []

    return [
        {
            "start_time": 0.0,
            "end_time": end_time,
            "start_label": format_time_label(0.0),
            "end_label": format_time_label(end_time),
            "behavior": behavior,
            "behavior_cn": behavior_cn,
            "confidence": safe_float(video_result.get("confidence"), 0.0),
            "duration": end_time,
        }
    ]

# =========================
# 3. YOLO 加载和检测
# =========================

def find_yolo_model_path() -> str:
    env_path = os.environ.get("YOLO_MODEL_PATH")

    if env_path and Path(env_path).exists():
        return env_path

    for path in YOLO_MODEL_PATHS:
        if Path(path).exists():
            return path

    return "yolov8n.pt"

def load_yolo_model():
    global _MODEL
    global _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL, None

    if _MODEL_LOAD_ERROR:
        return None, _MODEL_LOAD_ERROR

    try:
        from ultralytics import YOLO

        model_path = find_yolo_model_path()
        _MODEL = YOLO(model_path)
        return _MODEL, None

    except Exception as exc:
        _MODEL_LOAD_ERROR = f"YOLO 模型加载失败：{exc}"
        return None, _MODEL_LOAD_ERROR

def empty_detection(message: str = "未检测到猫或狗。") -> Dict[str, Any]:
    return {
        "animal": "unknown",
        "animal_cn": "未知",
        "animal_confidence": 0.0,
        "box": [],
        "score": 0.0,
        "message": message,
    }

def detect_cat_dog(frame: np.ndarray) -> Dict[str, Any]:
    model, error = load_yolo_model()

    if error:
        return {
            "animal": "unknown",
            "animal_cn": "未知",
            "animal_confidence": 0.0,
            "box": [],
            "score": 0.0,
            "message": error,
        }

    try:
        results = model(frame, verbose=False)

        if not results:
            return empty_detection("未检测到猫或狗。")

        result = results[0]
        boxes_obj = getattr(result, "boxes", None)

        if boxes_obj is None or len(boxes_obj) == 0:
            return empty_detection("未检测到猫或狗。")

        names = getattr(result, "names", {}) or {}
        best_item = None

        for box_obj in boxes_obj:
            cls_id = safe_int(box_obj.cls[0], -1)
            confidence = safe_float(box_obj.conf[0], 0.0)
            class_name = str(names.get(cls_id, cls_id)).lower()

            if class_name not in ("cat", "dog"):
                continue

            xyxy = box_obj.xyxy[0].detach().cpu().numpy().tolist()

            if len(xyxy) < 4:
                continue

            box = [
                safe_int(round(xyxy[0])),
                safe_int(round(xyxy[1])),
                safe_int(round(xyxy[2])),
                safe_int(round(xyxy[3])),
            ]

            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            frame_area = max(frame.shape[0] * frame.shape[1], 1)
            score = confidence * 0.8 + min(area / frame_area, 1.0) * 0.2

            item = {
                "animal": class_name,
                "animal_cn": ANIMAL_CN.get(class_name, "未知"),
                "animal_confidence": round(confidence, 4),
                "box": box,
                "score": score,
                "message": "ok",
            }

            if best_item is None or item["score"] > best_item["score"]:
                best_item = item

        if best_item is None:
            return empty_detection("未检测到猫或狗。")

        return best_item

    except Exception as exc:
        return empty_detection(f"YOLO 检测失败：{exc}")

# =========================
# 4. 图像质量和裁剪
# =========================

def calculate_image_quality(frame: np.ndarray) -> int:
    if frame is None or frame.size == 0:
        return 0

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        brightness_score = 100 - abs(brightness - 127.5) / 127.5 * 45
        sharpness_score = min(sharpness / 120.0 * 100, 100)

        score = brightness_score * 0.45 + sharpness_score * 0.55
        return int(round(clamp(score, 0, 100)))

    except Exception:
        return 0

def expand_box(box: List[int], width: int, height: int, ratio: float = 0.12) -> List[int]:
    if not box or len(box) < 4:
        return [0, 0, width, height]

    x1, y1, x2, y2 = box[:4]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    pad_x = int(box_width * ratio)
    pad_y = int(box_height * ratio)

    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]

def save_crop(frame: np.ndarray, box: List[int], output_path: str) -> Optional[str]:
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(box, width, height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, crop)
    return output_path

# =========================
# 5. 视频基础分析
# =========================

def calculate_frame_difference(prev_frame: np.ndarray, current_frame: np.ndarray) -> float:
    if prev_frame is None or current_frame is None:
        return 0.0

    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        prev_gray = cv2.resize(prev_gray, (160, 90))
        curr_gray = cv2.resize(curr_gray, (160, 90))

        diff = cv2.absdiff(prev_gray, curr_gray)
        return float(np.mean(diff))

    except Exception:
        return 0.0

def motion_score_from_diffs(diffs: List[float]) -> int:
    if not diffs:
        return 0

    avg_diff = float(np.mean(diffs))
    score = min(avg_diff / 35.0 * 100, 100)
    return int(round(clamp(score, 0, 100)))

def trend_from_motion_score(score: int) -> Tuple[str, str]:
    if score >= 75:
        return "active", "宠物在视频中活动明显，整体状态较活跃。"

    if score >= 45:
        return "moving", "宠物在视频中有一定活动，状态较自然。"

    if score >= 18:
        return "slight", "宠物在视频中有轻微活动，整体较稳定。"

    return "stable", "宠物在视频中基本静止，状态较稳定。"

def get_sample_indices(frame_count: int, fps: float, sample_per_second: int, max_frames: int) -> List[int]:
    if frame_count <= 0:
        return []

    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(sample_per_second, 1))))
    indices = list(range(0, frame_count, step))

    if max_frames > 0 and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames)
        indices = [indices[int(round(pos))] for pos in positions]

    return sorted(set([safe_int(index) for index in indices if 0 <= index < frame_count]))

def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success or frame is None:
            return None

        return frame

    except Exception:
        return None

def save_temp_frame(frame: np.ndarray, stem: str, frame_index: int) -> str:
    temp_dir = os.path.join(RESULT_DIR, "_tmp_pose_frames")
    os.makedirs(temp_dir, exist_ok=True)

    path = os.path.join(temp_dir, f"{stem}_{frame_index}.jpg")
    cv2.imwrite(path, frame)
    return path

# =========================
# 6. dog-pose 结果处理
# =========================

def normalize_pose_points(pose_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    if not isinstance(pose_result, dict):
        return {}

    raw_points = (
        pose_result.get("keypoints")
        or pose_result.get("points")
        or pose_result.get("keypoints_list")
        or pose_result.get("kpts")
        or pose_result.get("landmarks")
        or {}
    )

    result = {}

    if isinstance(raw_points, dict):
        for name, value in raw_points.items():
            point = normalize_single_point(value)

            if point:
                result[str(name)] = point

        return result

    if isinstance(raw_points, (list, tuple)):
        for index, value in enumerate(raw_points):
            point_name = str(index)

            if isinstance(value, dict):
                point_name = str(value.get("name", index))

            point = normalize_single_point(value)

            if point:
                result[point_name] = point

    return result

def normalize_single_point(value: Any) -> Optional[Dict[str, float]]:
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

        confidence = value.get("confidence", value.get("conf", value.get("score", 1.0)))

    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = value[0]
        y = value[1]
        confidence = value[2] if len(value) >= 3 else 1.0

    if x is None or y is None:
        return None

    confidence = safe_float(confidence, 1.0)

    if confidence < 0.1:
        return None

    return {
        "x": safe_float(x),
        "y": safe_float(y),
        "confidence": confidence,
    }


def normalize_pose_label(pose_result: Dict[str, Any]) -> Tuple[str, str, float]:
    if not isinstance(pose_result, dict):
        return "unknown", "未知", 0.0

    pose_label = (
        pose_result.get("pose_label")
        or pose_result.get("main_pose")
        or pose_result.get("pose")
        or pose_result.get("behavior")
        or pose_result.get("label")
        or "unknown"
    )

    pose_label = str(pose_label or "unknown").strip().lower()

    label_mapping = {
        "lying": "lie",
        "lie_down": "lie",
        "down": "lie",
        "sitting": "sit",
        "sit_down": "sit",
        "standing": "stand",
        "stand_up": "stand",
        "error": "unknown",
    }

    pose_label = label_mapping.get(pose_label, pose_label)

    pose_cn_map = {
        "unknown": "未知",
        "sit": "坐着",
        "stand": "站着",
        "lie": "趴着",
        "alert": "警觉",
        "relaxed": "放松",
    }

    pose_label_cn = (
        pose_result.get("pose_label_cn")
        or pose_result.get("pose_cn")
        or pose_result.get("main_pose_cn")
        or pose_cn_map.get(pose_label, "未知")
    )

    pose_confidence = safe_float(
        pose_result.get("pose_confidence", pose_result.get("confidence", 0.0)),
        0.0,
    )

    return pose_label, str(pose_label_cn or "未知"), pose_confidence

def normalize_pose_box(pose_result: Dict[str, Any]) -> List[int]:
    if not isinstance(pose_result, dict):
        return []

    box = pose_result.get("box") or pose_result.get("box_int") or pose_result.get("bbox") or []

    if isinstance(box, dict):
        x1 = safe_int(box.get("x1", box.get("left", 0)))
        y1 = safe_int(box.get("y1", box.get("top", 0)))
        x2 = safe_int(box.get("x2", box.get("right", 0)))
        y2 = safe_int(box.get("y2", box.get("bottom", 0)))

        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]

    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x1 = safe_int(round(safe_float(box[0])))
        y1 = safe_int(round(safe_float(box[1])))
        x2 = safe_int(round(safe_float(box[2])))
        y2 = safe_int(round(safe_float(box[3])))

        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]

    return []

def run_dog_pose_on_frame(
    frame: np.ndarray,
    video_stem: str,
    frame_index: int,
    timestamp: float,
    save_visualization: bool = False,
) -> Dict[str, Any]:
    if predict_dog_pose_for_video_frame is None:
        return {
            "success": False,
            "frame_index": frame_index,
            "time": timestamp,
            "timestamp": timestamp,
            "animal": "dog",
            "pose_label": "unknown",
            "pose_label_cn": "未知",
            "pose_confidence": 0.0,
            "keypoints": {},
            "points": {},
            "keypoints_list": [],
            "box": [],
            "box_int": [],
            "message": pose_load_error or "dog-pose 模块未加载。",
        }

    temp_path = None

    try:
        temp_path = save_temp_frame(frame, video_stem, frame_index)

        pose_result = predict_dog_pose_for_video_frame(
            temp_path,
            frame_index=frame_index,
            timestamp=timestamp,
            save_visualization=save_visualization,
            save_json=False,
        )

        if not isinstance(pose_result, dict):
            pose_result = {}

        keypoints = normalize_pose_points(pose_result)
        box = normalize_pose_box(pose_result)
        pose_label, pose_label_cn, pose_confidence = normalize_pose_label(pose_result)

        pose_result["success"] = bool(pose_result.get("success", True))
        pose_result["frame_index"] = frame_index
        pose_result["time"] = timestamp
        pose_result["timestamp"] = timestamp
        pose_result["animal"] = "dog"

        pose_result["pose_label"] = pose_label
        pose_result["pose_label_cn"] = pose_label_cn
        pose_result["pose_confidence"] = pose_confidence

        pose_result["keypoints"] = keypoints
        pose_result["points"] = keypoints
        pose_result["keypoint_count"] = len(keypoints)

        pose_result["box"] = box
        pose_result["box_int"] = box

        pose_result["source_frame_path"] = normalize_path(temp_path)

        return pose_result

    except Exception as exc:
        return {
            "success": False,
            "frame_index": frame_index,
            "time": timestamp,
            "timestamp": timestamp,
            "animal": "dog",
            "pose_label": "unknown",
            "pose_label_cn": "未知",
            "pose_confidence": 0.0,
            "keypoints": {},
            "points": {},
            "keypoints_list": [],
            "keypoint_count": 0,
            "box": [],
            "box_int": [],
            "message": f"dog-pose 预测失败：{exc}",
            "traceback": traceback.format_exc(),
            "source_frame_path": normalize_path(temp_path),
        }

def collect_pose_frames(
    video_path: str,
    fps: float,
    frame_count: int,
    sample_per_second: int,
    max_frames: int,
    save_visualization: bool = False,
    min_frames: int = 0,
) -> List[Dict[str, Any]]:
    if predict_dog_pose_for_video_frame is None:
        return []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    video_stem = Path(video_path).stem or "video"
    indices = get_sample_indices(frame_count, fps, sample_per_second, max_frames)

    if min_frames > 0 and frame_count > 0 and len(indices) < min_frames:
        expanded_count = min(frame_count, min_frames, max_frames if max_frames > 0 else min_frames)
        positions = np.linspace(0, frame_count - 1, expanded_count)
        expanded_indices = [safe_int(round(pos)) for pos in positions]
        indices = sorted(set(indices + expanded_indices))

    pose_frames = []

    try:
        for frame_index in indices:
            frame = read_frame_at(cap, frame_index)

            if frame is None:
                continue

            timestamp = frame_index / max(fps, 1.0)
            pose_result = run_dog_pose_on_frame(
                frame,
                video_stem=video_stem,
                frame_index=frame_index,
                timestamp=timestamp,
                save_visualization=save_visualization,
            )
            pose_frames.append(pose_result)

    finally:
        cap.release()

    return pose_frames
# =========================
# 7. 关键点标注视频
# =========================

def draw_pose_on_frame(frame: np.ndarray, pose_frame: Dict[str, Any]) -> np.ndarray:
    if frame is None:
        return frame

    output = frame.copy()
    points = normalize_pose_points(pose_frame)

    for name_a, name_b in DOG_POSE_EDGES:
        point_a = points.get(name_a)
        point_b = points.get(name_b)

        if not point_a or not point_b:
            continue

        x1 = int(round(point_a["x"]))
        y1 = int(round(point_a["y"]))
        x2 = int(round(point_b["x"]))
        y2 = int(round(point_b["y"]))

        cv2.line(output, (x1, y1), (x2, y2), (70, 220, 70), 2)

    for name, point in points.items():
        x = int(round(point["x"]))
        y = int(round(point["y"]))

        cv2.circle(output, (x, y), 4, (0, 128, 255), -1)
        cv2.circle(output, (x, y), 6, (255, 255, 255), 1)

        if name and not name.isdigit():
            cv2.putText(
                output,
                str(name),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    timestamp = pose_frame.get("time", pose_frame.get("timestamp"))
    if timestamp is not None:
        cv2.putText(
            output,
            f"{safe_float(timestamp):.2f}s",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output

def build_pose_lookup(pose_frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup = {}

    for pose_frame in pose_frames:
        if not isinstance(pose_frame, dict):
            continue

        frame_index = safe_int(pose_frame.get("frame_index"), -1)

        if frame_index >= 0:
            lookup[frame_index] = pose_frame

    return lookup

def find_nearest_pose_frame(
    frame_index: int,
    sorted_pose_indices: List[int],
    pose_lookup: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not sorted_pose_indices:
        return None

    nearest_index = min(sorted_pose_indices, key=lambda index: abs(index - frame_index))
    return pose_lookup.get(nearest_index)

def create_annotated_video(
    video_path: str,
    pose_frames: List[Dict[str, Any]],
    output_path: str,
    output_max_fps: int = OUTPUT_VIDEO_MAX_FPS,
    use_interpolation: bool = True,
) -> Optional[str]:
    if not pose_frames:
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = None

    try:
        source_fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
        frame_count = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
        height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)

        if width <= 0 or height <= 0:
            return None

        output_fps = min(max(source_fps, 1.0), max(output_max_fps, 1))
        step = max(1, int(round(source_fps / output_fps))) if source_fps > output_fps else 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        if not writer.isOpened():
            return None

        pose_lookup = build_pose_lookup(pose_frames)
        sorted_pose_indices = sorted(pose_lookup.keys())

        current_frame_index = 0

        while True:
            success, frame = cap.read()

            if not success or frame is None:
                break

            if current_frame_index % step != 0:
                current_frame_index += 1
                continue

            pose_frame = pose_lookup.get(current_frame_index)

            if pose_frame is None and use_interpolation:
                pose_frame = find_nearest_pose_frame(
                    current_frame_index,
                    sorted_pose_indices,
                    pose_lookup,
                )

            if pose_frame is not None:
                frame = draw_pose_on_frame(frame, pose_frame)

            writer.write(frame)
            current_frame_index += 1

        if frame_count == 0 and current_frame_index == 0:
            return None

        return output_path

    except Exception:
        return None

    finally:
        cap.release()

        if writer is not None:
            writer.release()

# =========================
# 8. 图片分析
# =========================

def analyze_image(image_path: str) -> Dict[str, Any]:
    try:
        frame = cv2.imread(image_path)

        if frame is None:
            return {
                "animal": "unknown",
                "animal_cn": "未知",
                "animal_confidence": 0.0,
                "best_frame_path": normalize_path(image_path),
                "best_frame_url": to_public_url(image_path),
                "crop_path": None,
                "crop_url": None,
                "motion_score": 0,
                "image_quality": 0,
                "trend": "unknown",
                "summary": "图片读取失败。",
                "message": "图片读取失败。",
            }

        paths = make_result_paths(image_path)
        detection = detect_cat_dog(frame)

        os.makedirs(os.path.dirname(paths["best_frame_path"]), exist_ok=True)
        cv2.imwrite(paths["best_frame_path"], frame)

        crop_saved_path = None
        if detection.get("box"):
            crop_saved_path = save_crop(frame, detection.get("box"), paths["crop_path"])

        image_quality = calculate_image_quality(frame)

        return {
            "animal": detection.get("animal", "unknown"),
            "animal_cn": detection.get("animal_cn", "未知"),
            "animal_confidence": round(safe_float(detection.get("animal_confidence")), 4),
            "best_frame_path": normalize_path(paths["best_frame_path"]),
            "best_frame_url": to_public_url(paths["best_frame_path"]),
            "crop_path": normalize_path(crop_saved_path),
            "crop_url": to_public_url(crop_saved_path),
            "motion_score": 0,
            "image_quality": image_quality,
            "trend": "image",
            "summary": "图片分析完成。",
            "detection": detection,
            "message": "ok",
        }

    except Exception as exc:
        return {
            "animal": "unknown",
            "animal_cn": "未知",
            "animal_confidence": 0.0,
            "best_frame_path": normalize_path(image_path),
            "best_frame_url": to_public_url(image_path),
            "crop_path": None,
            "crop_url": None,
            "motion_score": 0,
            "image_quality": 0,
            "trend": "unknown",
            "summary": f"图片分析失败：{exc}",
            "message": f"图片分析失败：{exc}",
            "traceback": traceback.format_exc(),
        }

# =========================
# 9. 视频分析主函数
# =========================

def analyze_video(video_path: str) -> Dict[str, Any]:
    paths = make_result_paths(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "animal": "unknown",
            "animal_cn": "未知",
            "animal_confidence": 0.0,
            "best_frame_path": None,
            "best_frame_url": None,
            "crop_path": None,
            "crop_url": None,
            "motion_score": 0,
            "image_quality": 0,
            "trend": "unknown",
            "summary": "视频打开失败。",
            "video_behavior": {
                "enabled": False,
                "main_behavior": "unknown",
                "main_behavior_cn": "未知",
                "behavior_timeline": [],
                "message": "视频打开失败。",
            },
            "behavior_timeline": [],
            "raw_annotated_video_path": None,
            "raw_annotated_video_url": None,
            "annotated_video_path": None,
            "annotated_video_url": None,
            "browser_video_converted": False,
            "browser_video_message": "视频打开失败。",
            "duration": 0.0,
            "frame_count": 0,
            "message": "视频打开失败。",
        }

    best_detection = None
    best_frame = None
    best_frame_index = 0
    previous_sample_frame = None
    motion_diffs = []
    sampled_quality_scores = []

    try:
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
        frame_count = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        duration = get_video_duration(frame_count, fps)

        detect_indices = get_sample_indices(
            frame_count=frame_count,
            fps=fps,
            sample_per_second=DETECT_SAMPLE_PER_SECOND,
            max_frames=max(20, int(duration * DETECT_SAMPLE_PER_SECOND)) if duration > 0 else 60,
        )

        if not detect_indices and frame_count > 0:
            detect_indices = [0]

        for frame_index in detect_indices:
            frame = read_frame_at(cap, frame_index)

            if frame is None:
                continue

            quality = calculate_image_quality(frame)
            sampled_quality_scores.append(quality)

            detection = detect_cat_dog(frame)

            if previous_sample_frame is not None:
                motion_diffs.append(calculate_frame_difference(previous_sample_frame, frame))

            previous_sample_frame = frame

            if best_detection is None or safe_float(detection.get("score")) > safe_float(best_detection.get("score")):
                best_detection = detection
                best_frame = frame.copy()
                best_frame_index = frame_index

        if best_detection is None:
            best_detection = empty_detection("未检测到有效视频帧。")

        if best_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, first_frame = cap.read()

            if success and first_frame is not None:
                best_frame = first_frame.copy()
            else:
                best_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        os.makedirs(os.path.dirname(paths["best_frame_path"]), exist_ok=True)
        cv2.imwrite(paths["best_frame_path"], best_frame)

        crop_saved_path = None
        if best_detection.get("box"):
            crop_saved_path = save_crop(best_frame, best_detection.get("box"), paths["crop_path"])

        legacy_motion_score = motion_score_from_diffs(motion_diffs)
        image_quality = int(round(np.mean(sampled_quality_scores))) if sampled_quality_scores else calculate_image_quality(best_frame)
        trend, trend_comment = trend_from_motion_score(legacy_motion_score)

        best_animal = best_detection.get("animal", "unknown")
        best_confidence = safe_float(best_detection.get("animal_confidence"), 0.0)

        pose_frames = []
        annotated_pose_frames = []
        video_result = None

        if best_animal == "dog" and predict_dog_pose_for_video_frame is not None:
            pose_frames = collect_pose_frames(
                video_path=video_path,
                fps=fps,
                frame_count=frame_count,
                sample_per_second=BEHAVIOR_POSE_SAMPLE_PER_SECOND,
                max_frames=MAX_BEHAVIOR_POSE_FRAMES,
                save_visualization=False,
                min_frames=MIN_BEHAVIOR_POSE_FRAMES,
            )

            annotated_pose_frames = collect_pose_frames(
                video_path=video_path,
                fps=fps,
                frame_count=frame_count,
                sample_per_second=ANNOTATED_POSE_SAMPLE_PER_SECOND,
                max_frames=MAX_ANNOTATED_POSE_FRAMES,
                save_visualization=False,
            )

            if analyze_video_behavior is not None:
                try:
                    try:
                        video_result = analyze_video_behavior(
                            pose_frames,
                            fps=BEHAVIOR_POSE_SAMPLE_PER_SECOND,
                            return_debug=False,
                        )
                    except TypeError:
                        video_result = analyze_video_behavior(
                            pose_frames,
                            fps=BEHAVIOR_POSE_SAMPLE_PER_SECOND,
                            return_debug=True,
                        )

                    if not isinstance(video_result, dict):
                        video_result = None

                except Exception as exc:
                    video_result = {
                        "enabled": False,
                        "main_behavior": "unknown",
                        "main_behavior_cn": "未知",
                        "motion_state": "unknown",
                        "motion_state_cn": "未知",
                        "trend": trend,
                        "summary": f"视频动态行为分析失败：{exc}",
                        "motion_score": legacy_motion_score,
                        "behavior_timeline": [],
                        "message": f"视频动态行为分析失败：{exc}",
                        "traceback": traceback.format_exc(),
                    }

        if video_result is None:
            video_result = {
                "enabled": False,
                "main_behavior": "unknown",
                "main_behavior_cn": "未知",
                "motion_state": "unknown",
                "motion_state_cn": "未知",
                "trend": trend,
                "summary": trend_comment,
                "motion_score": legacy_motion_score,
                "behavior_timeline": [],
                "message": "未启用 dog-pose 动态分析。" if best_animal == "dog" else "非狗视频不进行 dog-pose 动态分析。",
            }

        behavior_timeline = video_result.get("behavior_timeline", [])

        if not isinstance(behavior_timeline, list):
            behavior_timeline = []

        if not behavior_timeline and video_result.get("main_behavior"):
            behavior_timeline = fallback_behavior_timeline(video_result, duration)

        for segment in behavior_timeline:
            if isinstance(segment, dict):
                segment["start_label"] = segment.get("start_label") or format_time_label(segment.get("start_time", 0.0))
                segment["end_label"] = segment.get("end_label") or format_time_label(segment.get("end_time", 0.0))

        video_result["behavior_timeline"] = behavior_timeline

        final_motion_score = safe_int(video_result.get("motion_score"), legacy_motion_score)
        final_trend = str(video_result.get("trend") or trend)
        final_trend_comment = str(video_result.get("summary") or trend_comment)

        annotated_video_path = None

        if annotated_pose_frames:
            annotated_video_path = create_annotated_video(
                video_path,
                annotated_pose_frames,
                paths["annotated_video_path"],
                output_max_fps=OUTPUT_VIDEO_MAX_FPS,
                use_interpolation=True,
            )

        browser_video_path, browser_video_converted, browser_video_message = convert_to_browser_mp4(
            annotated_video_path
        )

        if not browser_video_path:
            browser_video_path = annotated_video_path

        return {
            "animal": best_animal,
            "animal_cn": ANIMAL_CN.get(best_animal, "未知"),
            "animal_confidence": round(safe_float(best_confidence), 4),

            "best_frame_path": normalize_path(paths["best_frame_path"]),
            "best_frame_url": to_public_url(paths["best_frame_path"]),
            "crop_path": normalize_path(crop_saved_path),
            "crop_url": to_public_url(crop_saved_path),

            "motion_score": final_motion_score,
            "image_quality": image_quality,
            "trend": final_trend,
            "trend_cn": video_result.get("trend_cn", ""),
            "summary": final_trend_comment,

            "main_behavior": video_result.get("main_behavior", "unknown"),
            "main_behavior_cn": video_result.get("main_behavior_cn", "未知"),
            "motion_state": video_result.get("motion_state", "unknown"),
            "motion_state_cn": video_result.get("motion_state_cn", "未知"),

            "video_behavior": video_result,
            "video_result": video_result,
            "behavior_timeline": behavior_timeline,

            "raw_annotated_video_path": normalize_path(annotated_video_path),
            "raw_annotated_video_url": to_public_url(annotated_video_path),

            "annotated_video_path": normalize_path(browser_video_path),
            "annotated_video_url": to_public_url(browser_video_path),
            "browser_video_converted": browser_video_converted,
            "browser_video_message": browser_video_message,

            "pose_frame_count": len(pose_frames),
            "valid_pose_frame_count": safe_int(video_result.get("valid_frame_count"), 0),
            "annotated_pose_frame_count": len(annotated_pose_frames),

            "behavior_pose_fps": BEHAVIOR_POSE_SAMPLE_PER_SECOND,
            "behavior_pose_sample_per_second": BEHAVIOR_POSE_SAMPLE_PER_SECOND,
            "max_behavior_pose_frames": MAX_BEHAVIOR_POSE_FRAMES,
            "min_behavior_pose_frames": MIN_BEHAVIOR_POSE_FRAMES,

            "annotated_pose_sample_per_second": ANNOTATED_POSE_SAMPLE_PER_SECOND,
            "max_annotated_pose_frames": MAX_ANNOTATED_POSE_FRAMES,
            "output_video_max_fps": OUTPUT_VIDEO_MAX_FPS,

            "duration": round(safe_float(duration), 4),
            "frame_count": frame_count,
            "fps": round(safe_float(fps), 4),

            "pose_load_error": pose_load_error,
            "behavior_load_error": behavior_load_error,

            "message": "ok",
        }

    except Exception as exc:
        return {
            "animal": "unknown",
            "animal_cn": "未知",
            "animal_confidence": 0.0,

            "best_frame_path": None,
            "best_frame_url": None,
            "crop_path": None,
            "crop_url": None,

            "motion_score": 0,
            "image_quality": 0,
            "trend": "unknown",
            "summary": f"视频分析失败：{exc}",

            "video_behavior": {
                "enabled": False,
                "main_behavior": "unknown",
                "main_behavior_cn": "未知",
                "behavior_timeline": [],
                "message": f"视频分析失败：{exc}",
            },
            "video_result": {
                "enabled": False,
                "main_behavior": "unknown",
                "main_behavior_cn": "未知",
                "behavior_timeline": [],
                "message": f"视频分析失败：{exc}",
            },
            "behavior_timeline": [],

            "raw_annotated_video_path": None,
            "raw_annotated_video_url": None,
            "annotated_video_path": None,
            "annotated_video_url": None,
            "browser_video_converted": False,
            "browser_video_message": f"视频分析失败：{exc}",

            "duration": 0.0,
            "frame_count": 0,
            "fps": 0.0,

            "pose_load_error": pose_load_error,
            "behavior_load_error": behavior_load_error,

            "message": f"视频分析失败：{exc}",
            "traceback": traceback.format_exc(),
        }

    finally:
        cap.release()

# =========================
# 10. 命令行测试
# =========================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze pet image/video.")
    parser.add_argument("input_path", help="输入图片或视频路径")
    args = parser.parse_args()

    input_path = args.input_path
    suffix = Path(input_path).suffix.lower()

    if suffix in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        output = analyze_video(input_path)
    else:
        output = analyze_image(input_path)

    print(json.dumps(output, ensure_ascii=False, indent=2))
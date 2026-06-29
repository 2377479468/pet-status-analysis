# -*- coding: utf-8 -*-
"""
video_behavior_analyzer.py

狗狗视频动态行为分析模块。

设计目标：
- dog_pose_analyzer / predict_dog_pose.py 继续负责单帧图片姿态分析
- 本模块只负责视频关键点序列的时序行为分析
- 从“单帧阈值判断”升级为：
  关键点序列 -> 帧级特征 -> 滑动窗口特征 -> 行为评分 -> 时间一致性分段

输入：
- pose_frames: predict_dog_pose_for_video_frame 输出的连续关键点结果
- fps: 关键点采样 FPS，不是原视频 FPS

输出：
- main_behavior
- motion_state
- trend
- motion_score
- behavior_timeline
- summary
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 1. 中文映射
# =========================

BEHAVIOR_CN = {
    "unknown": "未知",
    "still": "静止",
    "stable": "稳定",
    "slight": "轻微活动",
    "moving": "移动",
    "active": "活跃",
    "walk": "走动",
    "run": "跑动",
    "approaching": "靠近镜头",
    "leaving": "远离镜头",
    "turning": "转身",
    "shaking": "抖动",
    "stretching": "伸懒腰",
    "head_moving": "头部活动",
    "body_moving": "身体活动",
    "sit": "坐着",
    "stand": "站着",
    "lie": "趴着",
    "lie_down": "趴着",
    "alert": "警觉",
    "relaxed": "放松",
}

MOTION_STATE_CN = {
    "unknown": "未知",
    "still": "静止",
    "slight": "轻微活动",
    "moving": "移动",
    "active": "活跃",
}

TREND_CN = {
    "unknown": "未知",
    "stable": "整体稳定",
    "approaching": "逐渐靠近",
    "leaving": "逐渐远离",
    "moving": "持续活动",
    "mixed": "行为变化较多",
}

# =========================
# 2. 基础工具
# =========================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def distance(p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> float:
    if not p1 or not p2:
        return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def mean(values: List[float], default: float = 0.0) -> float:
    values = [safe_float(v) for v in values if v is not None]
    if not values:
        return default
    return sum(values) / len(values)

def median(values: List[float], default: float = 0.0) -> float:
    values = sorted([safe_float(v) for v in values if v is not None])
    if not values:
        return default
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0

def linear_slope(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """
    返回 slope, r2。
    """
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0, 0.0

    x_mean = mean(xs)
    y_mean = mean(ys)

    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx <= 1e-9:
        return 0.0, 0.0

    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))

    if ss_tot <= 1e-9:
        r2 = 0.0
    else:
        r2 = clamp(1.0 - ss_res / ss_tot, 0.0, 1.0)

    return slope, r2

def normalize_label(label: Any) -> str:
    label = str(label or "").strip().lower()

    mapping = {
        "lying": "lie",
        "lie_down": "lie",
        "down": "lie",
        "sit_down": "sit",
        "sitting": "sit",
        "standing": "stand",
        "stand_up": "stand",
        "unknown_pose": "unknown",
        "error": "unknown",
    }

    return mapping.get(label, label or "unknown")

# =========================
# 3. 关键点解析
# =========================

def parse_point(value: Any) -> Optional[Tuple[float, float]]:
    """
    支持：
    - [x, y]
    - [x, y, conf]
    - {"x": x, "y": y}
    - {"point": [x, y]}
    """
    if value is None:
        return None

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return safe_float(value.get("x")), safe_float(value.get("y"))

        if "point" in value:
            return parse_point(value.get("point"))

        if "position" in value:
            return parse_point(value.get("position"))

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return safe_float(value[0]), safe_float(value[1])

    return None

def extract_keypoints(frame: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    尽量兼容 predict_dog_pose.py 的多种输出：
    - keypoints: dict
    - keypoints_list: list
    - points: list
    """
    keypoints: Dict[str, Tuple[float, float]] = {}

    raw_keypoints = frame.get("keypoints")
    if isinstance(raw_keypoints, dict):
        for name, value in raw_keypoints.items():
            point = parse_point(value)
            if point:
                keypoints[str(name)] = point

    keypoints_list = frame.get("keypoints_list")
    if isinstance(keypoints_list, list):
        for index, value in enumerate(keypoints_list):
            point = parse_point(value)
            if point:
                keypoints.setdefault(f"kp_{index}", point)

    points = frame.get("points")
    if isinstance(points, list):
        for index, value in enumerate(points):
            point = parse_point(value)
            if point:
                keypoints.setdefault(f"pt_{index}", point)

    return keypoints

def extract_bbox(frame: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    box = frame.get("box") or frame.get("box_int") or frame.get("bbox")

    if isinstance(box, dict):
        x1 = safe_float(box.get("x1", box.get("left", 0)))
        y1 = safe_float(box.get("y1", box.get("top", 0)))
        x2 = safe_float(box.get("x2", box.get("right", 0)))
        y2 = safe_float(box.get("y2", box.get("bottom", 0)))
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2

    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x1 = safe_float(box[0])
        y1 = safe_float(box[1])
        x2 = safe_float(box[2])
        y2 = safe_float(box[3])
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2

    return None

def bbox_center(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float]]:
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> float:
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def keypoint_center(keypoints: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not keypoints:
        return None

    xs = [p[0] for p in keypoints.values()]
    ys = [p[1] for p in keypoints.values()]

    return mean(xs), mean(ys)

def keypoint_scale(keypoints: Dict[str, Tuple[float, float]], bbox: Optional[Tuple[float, float, float, float]] = None) -> float:
    """
    优先用关键点分布尺度；关键点太少时退化到 bbox 对角线。
    """
    if len(keypoints) >= 3:
        xs = [p[0] for p in keypoints.values()]
        ys = [p[1] for p in keypoints.values()]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        scale = math.hypot(width, height)
        if scale > 1e-6:
            return scale

    if bbox:
        x1, y1, x2, y2 = bbox
        return math.hypot(x2 - x1, y2 - y1)

    return 0.0

def estimate_body_angle(keypoints: Dict[str, Tuple[float, float]]) -> float:
    """
    尽量估计身体朝向角。
    如果有头/臀/尾/身体点，优先使用；否则用关键点云主方向近似。
    """
    lower_keys = {k.lower(): v for k, v in keypoints.items()}

    head = None
    rear = None

    for name in ("nose", "head", "kp_0", "pt_0"):
        if name in lower_keys:
            head = lower_keys[name]
            break

    for name in ("tail_base", "tail", "hip", "rear", "kp_6", "pt_6"):
        if name in lower_keys:
            rear = lower_keys[name]
            break

    if head and rear:
        return math.atan2(head[1] - rear[1], head[0] - rear[0])

    if len(keypoints) < 2:
        return 0.0

    points = list(keypoints.values())
    cx, cy = keypoint_center(keypoints) or (0.0, 0.0)

    sxx = sum((x - cx) ** 2 for x, _ in points)
    syy = sum((y - cy) ** 2 for _, y in points)
    sxy = sum((x - cx) * (y - cy) for x, y in points)

    if abs(sxx - syy) < 1e-6 and abs(sxy) < 1e-6:
        return 0.0

    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)

def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b)
    while diff > math.pi:
        diff -= 2.0 * math.pi
    return abs(diff)

def named_points_motion(
    current: Dict[str, Tuple[float, float]],
    previous: Dict[str, Tuple[float, float]],
    keywords: Tuple[str, ...],
) -> float:
    distances = []

    for name, point in current.items():
        lower_name = name.lower()
        if not any(keyword in lower_name for keyword in keywords):
            continue

        previous_point = previous.get(name)
        if previous_point:
            distances.append(distance(point, previous_point))

    return mean(distances)

# =========================
# 4. 帧级特征
# =========================

def build_frame_features(pose_frames: List[Dict[str, Any]], fps: float = 5.0) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []

    previous_keypoints: Dict[str, Tuple[float, float]] = {}
    previous_center: Optional[Tuple[float, float]] = None
    previous_time: Optional[float] = None
    previous_angle = 0.0

    for index, frame in enumerate(pose_frames):
        timestamp = safe_float(
            frame.get("time", frame.get("timestamp", index / max(fps, 1e-6))),
            index / max(fps, 1e-6),
        )

        keypoints = extract_keypoints(frame)
        bbox = extract_bbox(frame)

        center = keypoint_center(keypoints) or bbox_center(bbox)
        scale = keypoint_scale(keypoints, bbox)
        area = bbox_area(bbox)
        angle = estimate_body_angle(keypoints)

        dt = 0.0
        if previous_time is not None:
            dt = max(1e-6, timestamp - previous_time)

        center_motion = 0.0
        center_speed = 0.0
        if center and previous_center and dt > 0:
            center_motion = distance(center, previous_center)
            center_speed = center_motion / dt

        all_motion_values = []
        if previous_keypoints:
            for name, point in keypoints.items():
                if name in previous_keypoints:
                    all_motion_values.append(distance(point, previous_keypoints[name]))

        all_motion = mean(all_motion_values)
        head_motion = named_points_motion(keypoints, previous_keypoints, ("head", "nose", "ear", "eye", "kp_0", "pt_0"))
        paw_motion = named_points_motion(keypoints, previous_keypoints, ("paw", "leg", "foot"))
        tail_motion = named_points_motion(keypoints, previous_keypoints, ("tail",))
        angle_change = angle_diff(angle, previous_angle) if previous_time is not None else 0.0

        pose_label = normalize_label(
            frame.get("pose_label")
            or frame.get("main_pose")
            or frame.get("behavior")
            or frame.get("label")
        )

        pose_confidence = safe_float(frame.get("pose_confidence", frame.get("confidence", 0.0)))

        feature = {
            "frame_index": frame.get("frame_index", index),
            "time": timestamp,
            "timestamp": timestamp,
            "valid": bool(keypoints or bbox),
            "pose_label": pose_label,
            "pose_label_cn": frame.get("pose_label_cn") or BEHAVIOR_CN.get(pose_label, "未知"),
            "pose_confidence": pose_confidence,

            "keypoint_count": len(keypoints),
            "center": center,
            "scale": scale,
            "bbox_area": area,
            "body_angle": angle,

            "center_motion": center_motion,
            "center_speed": center_speed,
            "all_motion": all_motion,
            "head_motion": head_motion,
            "paw_motion": paw_motion,
            "tail_motion": tail_motion,
            "angle_change": angle_change,
        }

        features.append(feature)

        previous_keypoints = keypoints
        previous_center = center
        previous_time = timestamp
        previous_angle = angle

    return features

# =========================
# 5. 滑动窗口特征
# =========================

def build_window_features(
    frame_features: List[Dict[str, Any]],
    fps: float = 5.0,
    window_seconds: float = 1.2,
    step_seconds: float = 0.4,
) -> List[Dict[str, Any]]:
    valid_features = [f for f in frame_features if f.get("valid")]

    if not valid_features:
        return []

    window_size = max(3, int(round(window_seconds * fps)))
    step_size = max(1, int(round(step_seconds * fps)))

    if len(valid_features) < window_size:
        window_size = len(valid_features)

    windows: List[Dict[str, Any]] = []

    for start in range(0, len(valid_features) - window_size + 1, step_size):
        items = valid_features[start:start + window_size]
        times = [safe_float(f["time"]) for f in items]
        start_time = times[0]
        end_time = times[-1]
        duration = max(1e-6, end_time - start_time)

        scales = [safe_float(f.get("scale")) for f in items]
        areas = [safe_float(f.get("bbox_area")) for f in items]
        center_speeds = [safe_float(f.get("center_speed")) for f in items[1:]]
        all_motions = [safe_float(f.get("all_motion")) for f in items[1:]]
        head_motions = [safe_float(f.get("head_motion")) for f in items[1:]]
        paw_motions = [safe_float(f.get("paw_motion")) for f in items[1:]]
        tail_motions = [safe_float(f.get("tail_motion")) for f in items[1:]]
        angle_changes = [safe_float(f.get("angle_change")) for f in items[1:]]

        scale_base = max(median(scales), 1e-6)
        area_base = max(median(areas), 1e-6)

        scale_slope, scale_r2 = linear_slope(times, scales)
        area_slope, area_r2 = linear_slope(times, areas)

        normalized_scale_slope = scale_slope / scale_base
        normalized_area_slope = area_slope / area_base

        pose_labels = [f.get("pose_label", "unknown") for f in items]
        pose_counter = Counter(pose_labels)
        dominant_pose, dominant_pose_count = pose_counter.most_common(1)[0]
        dominant_pose_ratio = dominant_pose_count / max(1, len(items))

        centers = [f.get("center") for f in items if f.get("center")]
        center_displacement = 0.0
        if len(centers) >= 2:
            center_displacement = distance(centers[0], centers[-1])

        motion_energy = mean(all_motions)
        center_speed = mean(center_speeds)
        head_motion = mean(head_motions)
        paw_motion = mean(paw_motions)
        tail_motion = mean(tail_motions)
        angle_change_total = sum(angle_changes)

        window = {
            "start_index": start,
            "end_index": start + window_size - 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,

            "frame_count": len(items),
            "dominant_pose": dominant_pose,
            "dominant_pose_cn": BEHAVIOR_CN.get(dominant_pose, "未知"),
            "dominant_pose_ratio": dominant_pose_ratio,
            "pose_distribution": dict(pose_counter),

            "scale_slope": normalized_scale_slope,
            "scale_r2": scale_r2,
            "area_slope": normalized_area_slope,
            "area_r2": area_r2,

            "center_speed": center_speed,
            "center_displacement": center_displacement,
            "motion_energy": motion_energy,
            "head_motion": head_motion,
            "paw_motion": paw_motion,
            "tail_motion": tail_motion,
            "angle_change_total": angle_change_total,
        }

        windows.append(window)

    if not windows and valid_features:
        return build_window_features(valid_features, fps=fps, window_seconds=max(0.6, len(valid_features) / max(fps, 1.0)), step_seconds=0.5)

    return windows

# =========================
# 6. 行为评分
# =========================

def score_window_behavior(window: Dict[str, Any]) -> Dict[str, Any]:
    scores = {
        "still": 0.0,
        "moving": 0.0,
        "walk": 0.0,
        "run": 0.0,
        "approaching": 0.0,
        "leaving": 0.0,
        "turning": 0.0,
        "stretching": 0.0,
        "sit": 0.0,
        "stand": 0.0,
        "lie": 0.0,
        "head_moving": 0.0,
    }

    motion = safe_float(window.get("motion_energy"))
    speed = safe_float(window.get("center_speed"))
    scale_slope = safe_float(window.get("scale_slope"))
    scale_r2 = safe_float(window.get("scale_r2"))
    area_slope = safe_float(window.get("area_slope"))
    area_r2 = safe_float(window.get("area_r2"))
    angle_change_total = safe_float(window.get("angle_change_total"))
    head_motion = safe_float(window.get("head_motion"))
    paw_motion = safe_float(window.get("paw_motion"))

    dominant_pose = normalize_label(window.get("dominant_pose"))
    dominant_pose_ratio = safe_float(window.get("dominant_pose_ratio"))

    low_motion = motion < 2.0 and speed < 8.0
    medium_motion = motion >= 2.0 or speed >= 8.0
    high_motion = motion >= 8.0 or speed >= 35.0

    if low_motion:
        scores["still"] += 0.75

    if medium_motion:
        scores["moving"] += clamp(motion / 8.0, 0.2, 0.8)
        scores["walk"] += clamp((speed - 6.0) / 35.0, 0.0, 0.8)

    if high_motion:
        scores["run"] += clamp((speed - 28.0) / 45.0, 0.2, 0.95)

    if scale_slope > 0.06 and scale_r2 > 0.35:
        scores["approaching"] += clamp(scale_slope * 5.0 + scale_r2 * 0.4, 0.0, 0.95)

    if area_slope > 0.10 and area_r2 > 0.35:
        scores["approaching"] += clamp(area_slope * 2.0 + area_r2 * 0.25, 0.0, 0.6)

    if scale_slope < -0.06 and scale_r2 > 0.35:
        scores["leaving"] += clamp(abs(scale_slope) * 5.0 + scale_r2 * 0.4, 0.0, 0.95)

    if area_slope < -0.10 and area_r2 > 0.35:
        scores["leaving"] += clamp(abs(area_slope) * 2.0 + area_r2 * 0.25, 0.0, 0.6)

    if angle_change_total > 0.75 and motion >= 1.5:
        scores["turning"] += clamp(angle_change_total / 2.5, 0.2, 0.9)

    if dominant_pose in ("sit", "stand", "lie"):
        scores[dominant_pose] += clamp(dominant_pose_ratio, 0.0, 0.85)
        if low_motion:
            scores[dominant_pose] += 0.15

    if head_motion > motion * 1.4 and head_motion > 2.5 and speed < 15.0:
        scores["head_moving"] += clamp(head_motion / 12.0, 0.2, 0.8)

    # 伸懒腰：姿态变化 + 身体尺度增大 + 整体位移不太大。
    if scale_slope > 0.04 and motion >= 1.0 and speed < 25.0 and dominant_pose in ("stand", "lie", "unknown"):
        scores["stretching"] += clamp(scale_slope * 4.0 + motion / 20.0, 0.0, 0.75)

    # 爪部运动明显时更偏向走动。
    if paw_motion > 2.0 and speed > 6.0:
        scores["walk"] += clamp(paw_motion / 12.0, 0.1, 0.5)

    # 防止靠近/远离压过明显静止姿态。
    if low_motion and dominant_pose in ("sit", "stand", "lie"):
        scores["approaching"] *= 0.4
        scores["leaving"] *= 0.4
        scores["moving"] *= 0.5
        scores["walk"] *= 0.4
        scores["run"] *= 0.3

    behavior, confidence = max(scores.items(), key=lambda item: item[1])

    if confidence < 0.25:
        behavior = dominant_pose if dominant_pose in ("sit", "stand", "lie") else "unknown"
        confidence = max(confidence, dominant_pose_ratio * 0.5)

    return {
        "behavior": behavior,
        "behavior_cn": BEHAVIOR_CN.get(behavior, "未知"),
        "confidence": round(clamp(confidence, 0.0, 1.0), 3),
        "scores": {key: round(clamp(value, 0.0, 1.0), 3) for key, value in scores.items()},
    }

def classify_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    predictions = []

    for window in windows:
        prediction = score_window_behavior(window)
        predictions.append({
            **window,
            **prediction,
        })

    return predictions

# =========================
# 7. 时间一致性和平滑
# =========================

def smooth_predictions(predictions: List[Dict[str, Any]], min_repeat: int = 2) -> List[Dict[str, Any]]:
    if len(predictions) < 3:
        return predictions

    smoothed = [dict(p) for p in predictions]

    for index in range(1, len(predictions) - 1):
        previous_behavior = predictions[index - 1].get("behavior")
        current_behavior = predictions[index].get("behavior")
        next_behavior = predictions[index + 1].get("behavior")

        current_confidence = safe_float(predictions[index].get("confidence"))

        if previous_behavior == next_behavior and current_behavior != previous_behavior and current_confidence < 0.65:
            smoothed[index]["behavior"] = previous_behavior
            smoothed[index]["behavior_cn"] = BEHAVIOR_CN.get(previous_behavior, "未知")
            smoothed[index]["confidence"] = round(
                mean([
                    safe_float(predictions[index - 1].get("confidence")),
                    current_confidence,
                    safe_float(predictions[index + 1].get("confidence")),
                ]),
                3,
            )

    if min_repeat <= 1 or len(smoothed) < min_repeat + 1:
        return smoothed

    final_predictions = [dict(p) for p in smoothed]

    run_start = 0
    while run_start < len(smoothed):
        run_behavior = smoothed[run_start].get("behavior")
        run_end = run_start

        while run_end + 1 < len(smoothed) and smoothed[run_end + 1].get("behavior") == run_behavior:
            run_end += 1

        run_length = run_end - run_start + 1

        if run_length < min_repeat:
            left_behavior = smoothed[run_start - 1].get("behavior") if run_start > 0 else None
            right_behavior = smoothed[run_end + 1].get("behavior") if run_end + 1 < len(smoothed) else None

            replacement = None
            if left_behavior == right_behavior and left_behavior:
                replacement = left_behavior
            elif left_behavior:
                replacement = left_behavior
            elif right_behavior:
                replacement = right_behavior

            if replacement:
                for index in range(run_start, run_end + 1):
                    final_predictions[index]["behavior"] = replacement
                    final_predictions[index]["behavior_cn"] = BEHAVIOR_CN.get(replacement, "未知")

        run_start = run_end + 1

    return final_predictions

def build_behavior_timeline(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not predictions:
        return []

    timeline: List[Dict[str, Any]] = []

    current_behavior = predictions[0].get("behavior", "unknown")
    current_start = safe_float(predictions[0].get("start_time"))
    current_end = safe_float(predictions[0].get("end_time"))
    confidences = [safe_float(predictions[0].get("confidence"))]

    for prediction in predictions[1:]:
        behavior = prediction.get("behavior", "unknown")
        start_time = safe_float(prediction.get("start_time"))
        end_time = safe_float(prediction.get("end_time"))

        if behavior == current_behavior:
            current_end = max(current_end, end_time)
            confidences.append(safe_float(prediction.get("confidence")))
            continue

        timeline.append({
            "start_time": round(current_start, 2),
            "end_time": round(current_end, 2),
            "duration": round(max(0.0, current_end - current_start), 2),
            "behavior": current_behavior,
            "behavior_cn": BEHAVIOR_CN.get(current_behavior, "未知"),
            "confidence": round(mean(confidences), 3),
        })

        current_behavior = behavior
        current_start = start_time
        current_end = end_time
        confidences = [safe_float(prediction.get("confidence"))]

    timeline.append({
        "start_time": round(current_start, 2),
        "end_time": round(current_end, 2),
        "duration": round(max(0.0, current_end - current_start), 2),
        "behavior": current_behavior,
        "behavior_cn": BEHAVIOR_CN.get(current_behavior, "未知"),
        "confidence": round(mean(confidences), 3),
    })

    # 合并过短片段，避免时间轴碎片化。
    if len(timeline) <= 1:
        return timeline

    merged: List[Dict[str, Any]] = []

    for segment in timeline:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]

        if (
            segment["behavior"] == previous["behavior"]
            or segment["duration"] < 0.35
        ):
            total_duration = previous["duration"] + segment["duration"]
            previous_confidence = previous["confidence"]
            segment_confidence = segment["confidence"]

            previous["end_time"] = segment["end_time"]
            previous["duration"] = round(max(0.0, previous["end_time"] - previous["start_time"]), 2)

            if total_duration > 0:
                previous["confidence"] = round(
                    (previous_confidence * max(previous["duration"], 0.01) + segment_confidence * max(segment["duration"], 0.01))
                    / max(previous["duration"] + segment["duration"], 0.01),
                    3,
                )
        else:
            merged.append(segment)

    return merged

# =========================
# 8. 汇总
# =========================

def infer_motion_state(frame_features: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> str:
    if not frame_features:
        return "unknown"

    motions = [safe_float(f.get("all_motion")) for f in frame_features if f.get("valid")]
    speeds = [safe_float(f.get("center_speed")) for f in frame_features if f.get("valid")]

    avg_motion = mean(motions)
    avg_speed = mean(speeds)

    behavior_counter = Counter(p.get("behavior") for p in predictions)
    active_count = sum(behavior_counter.get(name, 0) for name in ("moving", "walk", "run", "turning", "approaching", "leaving"))

    if avg_motion < 1.2 and avg_speed < 5.0:
        return "still"

    if avg_motion < 3.0 and avg_speed < 12.0:
        return "slight"

    if active_count >= max(1, len(predictions) * 0.45):
        return "active" if avg_motion > 7.0 or avg_speed > 30.0 else "moving"

    return "moving"

def infer_trend(predictions: List[Dict[str, Any]]) -> str:
    if not predictions:
        return "unknown"

    counter = Counter(p.get("behavior") for p in predictions)

    if counter.get("approaching", 0) >= max(1, len(predictions) * 0.35):
        return "approaching"

    if counter.get("leaving", 0) >= max(1, len(predictions) * 0.35):
        return "leaving"

    active_count = sum(counter.get(name, 0) for name in ("moving", "walk", "run", "turning"))
    stable_count = sum(counter.get(name, 0) for name in ("still", "sit", "stand", "lie"))

    if active_count >= max(1, len(predictions) * 0.45):
        return "moving"

    if stable_count >= max(1, len(predictions) * 0.55):
        return "stable"

    return "mixed"

def calculate_motion_score(frame_features: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> int:
    valid_features = [f for f in frame_features if f.get("valid")]

    if not valid_features:
        return 0

    avg_motion = mean([safe_float(f.get("all_motion")) for f in valid_features])
    avg_speed = mean([safe_float(f.get("center_speed")) for f in valid_features])

    behavior_counter = Counter(p.get("behavior") for p in predictions)
    run_ratio = behavior_counter.get("run", 0) / max(1, len(predictions))
    walk_ratio = behavior_counter.get("walk", 0) / max(1, len(predictions))
    moving_ratio = sum(behavior_counter.get(name, 0) for name in ("moving", "approaching", "leaving", "turning")) / max(1, len(predictions))
    still_ratio = sum(behavior_counter.get(name, 0) for name in ("still", "sit", "stand", "lie")) / max(1, len(predictions))

    score = 0.0
    score += clamp(avg_motion / 10.0, 0.0, 1.0) * 35.0
    score += clamp(avg_speed / 45.0, 0.0, 1.0) * 30.0
    score += run_ratio * 25.0
    score += walk_ratio * 15.0
    score += moving_ratio * 15.0
    score -= still_ratio * 12.0

    return int(round(clamp(score, 0.0, 100.0)))

def pick_main_behavior(predictions: List[Dict[str, Any]], frame_features: List[Dict[str, Any]]) -> str:
    if predictions:
        weighted_scores: Dict[str, float] = {}

        for prediction in predictions:
            behavior = prediction.get("behavior", "unknown")
            confidence = safe_float(prediction.get("confidence"), 0.0)
            duration = max(0.1, safe_float(prediction.get("duration"), 0.4))
            weighted_scores[behavior] = weighted_scores.get(behavior, 0.0) + confidence * duration

        if weighted_scores:
            behavior = max(weighted_scores.items(), key=lambda item: item[1])[0]
            if behavior != "unknown":
                return behavior

    pose_labels = [normalize_label(f.get("pose_label")) for f in frame_features if f.get("pose_label")]
    pose_counter = Counter(label for label in pose_labels if label != "unknown")

    if pose_counter:
        return pose_counter.most_common(1)[0][0]

    return "unknown"

def build_summary(
    main_behavior: str,
    motion_state: str,
    trend: str,
    motion_score: int,
    timeline: List[Dict[str, Any]],
) -> str:
    main_cn = BEHAVIOR_CN.get(main_behavior, "未知")
    motion_cn = MOTION_STATE_CN.get(motion_state, "未知")

    if not timeline:
        return "视频中可用的狗狗关键点序列较少，暂未形成稳定的行为判断。"

    if main_behavior == "approaching":
        return f"狗狗整体呈靠近镜头趋势，当前活跃指数约为 {motion_score}。"

    if main_behavior == "leaving":
        return f"狗狗整体呈远离镜头趋势，当前活跃指数约为 {motion_score}。"

    if main_behavior == "run":
        return f"狗狗运动幅度较大，疑似正在跑动，活跃指数约为 {motion_score}。"

    if main_behavior in ("walk", "moving", "turning"):
        return f"狗狗正在活动，主要行为为{main_cn}，活跃指数约为 {motion_score}。"

    if main_behavior in ("sit", "stand", "lie"):
        return f"狗狗主要姿态是{main_cn}，整体状态为{motion_cn}，活跃指数约为 {motion_score}。"

    if motion_state == "still":
        return f"狗狗整体比较稳定，画面中没有明显大幅动作，活跃指数约为 {motion_score}。"

    if trend == "mixed":
        return f"狗狗在视频中行为变化较多，当前活跃指数约为 {motion_score}。"

    return f"狗狗当前主要状态为{main_cn}，活跃指数约为 {motion_score}。"

# =========================
# 9. 主入口
# =========================

def analyze_video_behavior(
    pose_frames: List[Dict[str, Any]],
    fps: float = 5.0,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """
    分析狗狗视频关键点序列。

    参数：
    - pose_frames: predict_dog_pose_for_video_frame 输出列表
    - fps: 关键点采样 FPS，不是原视频 FPS
    - return_debug: 是否返回调试信息
    """
    fps = safe_float(fps, 5.0)
    if fps <= 0:
        fps = 5.0

    if not pose_frames:
        return {
            "enabled": False,
            "main_behavior": "unknown",
            "main_behavior_cn": "未知",
            "motion_state": "unknown",
            "motion_state_cn": "未知",
            "trend": "unknown",
            "trend_cn": "未知",
            "motion_score": 0,
            "avg_motion": 0.0,
            "avg_speed": 0.0,
            "frame_count": 0,
            "valid_frame_count": 0,
            "behavior_timeline": [],
            "behaviors": [],
            "summary": "视频中没有可用于行为分析的狗狗关键点序列。",
            "message": "empty pose_frames",
        }

    frame_features = build_frame_features(pose_frames, fps=fps)
    valid_features = [f for f in frame_features if f.get("valid")]

    if len(valid_features) < 2:
        pose_label = normalize_label(valid_features[0].get("pose_label")) if valid_features else "unknown"
        main_behavior = pose_label if pose_label != "unknown" else "unknown"

        result = {
            "enabled": False,
            "main_behavior": main_behavior,
            "main_behavior_cn": BEHAVIOR_CN.get(main_behavior, "未知"),
            "motion_state": "unknown",
            "motion_state_cn": "未知",
            "trend": "unknown",
            "trend_cn": "未知",
            "motion_score": 0,
            "avg_motion": 0.0,
            "avg_speed": 0.0,
            "frame_count": len(pose_frames),
            "valid_frame_count": len(valid_features),
            "behavior_timeline": [],
            "behaviors": [],
            "summary": "可用关键点帧数不足，无法进行稳定的视频行为分析。",
            "message": "not enough valid pose frames",
        }

        if return_debug:
            result["frame_features"] = frame_features

        return result

    window_seconds = 1.2
    step_seconds = 0.4

    if len(valid_features) < int(round(fps * window_seconds)):
        window_seconds = max(0.6, len(valid_features) / fps)
        step_seconds = max(0.2, window_seconds / 2.0)

    windows = build_window_features(
        frame_features,
        fps=fps,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )

    raw_predictions = classify_windows(windows)
    predictions = smooth_predictions(raw_predictions, min_repeat=2)
    timeline = build_behavior_timeline(predictions)

    main_behavior = pick_main_behavior(predictions, frame_features)
    motion_state = infer_motion_state(frame_features, predictions)
    trend = infer_trend(predictions)
    motion_score = calculate_motion_score(frame_features, predictions)

    avg_motion = mean([safe_float(f.get("all_motion")) for f in valid_features])
    avg_speed = mean([safe_float(f.get("center_speed")) for f in valid_features])

    behaviors = []
    for segment in timeline:
        behaviors.append({
            "behavior": segment["behavior"],
            "behavior_cn": segment["behavior_cn"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "duration": segment["duration"],
            "confidence": segment["confidence"],
        })

    result = {
        "enabled": True,
        "main_behavior": main_behavior,
        "main_behavior_cn": BEHAVIOR_CN.get(main_behavior, "未知"),
        "motion_state": motion_state,
        "motion_state_cn": MOTION_STATE_CN.get(motion_state, "未知"),
        "trend": trend,
        "trend_cn": TREND_CN.get(trend, "未知"),
        "motion_score": motion_score,
        "avg_motion": round(avg_motion, 3),
        "avg_speed": round(avg_speed, 3),
        "frame_count": len(pose_frames),
        "valid_frame_count": len(valid_features),
        "behavior_timeline": timeline,
        "behaviors": behaviors,
        "summary": build_summary(main_behavior, motion_state, trend, motion_score, timeline),
        "message": "ok",
    }

    if return_debug:
        result["frame_features"] = frame_features
        result["window_features"] = windows
        result["window_predictions"] = predictions
        result["raw_window_predictions"] = raw_predictions

    return result
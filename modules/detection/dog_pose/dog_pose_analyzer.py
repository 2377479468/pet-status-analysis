# -*- coding: utf-8 -*-
"""
dog_pose_analyzer.py

狗狗静态姿态分析模块。

职责：
1. 用于单张图片、视频最佳帧、视频抽样帧的静态姿态识别。
2. 只输出静态姿态，不输出视频动态行为。
3. 主姿态：
   - stand
   - sit
   - lie_down
   - unknown
4. 附加静态状态：
   - look_up
   - head_down
   - alert
   - relaxed
5. 视频动态行为：
   - moving
   - approaching
   - leaving
   - shaking
   - turning
   - stretching
   后续交给 video_behavior_analyzer.py 处理。
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2

# =========================
# 1. 标签映射
# =========================

POSE_CN = {
    # 图片/单帧静态主姿态
    "stand": "站立",
    "sit": "坐下",
    "lie_down": "趴下",
    "unknown": "未知",
    "error": "错误",

    # 图片/单帧静态附加状态
    "look_up": "抬头",
    "head_down": "低头",
    "alert": "警觉",
    "relaxed": "放松",

    # 视频动态标签，仅预留给 video_behavior_analyzer.py，不在本文件输出
    "still": "静止",
    "moving": "移动",
    "approaching": "靠近镜头",
    "leaving": "远离镜头",
    "shaking": "摇晃",
    "turning": "转身",
    "stretching": "伸懒腰",
}

IMAGE_MAIN_POSES = {
    "stand",
    "sit",
    "lie_down",
    "unknown",
}

IMAGE_EXTRA_POSES = {
    "look_up",
    "head_down",
    "alert",
    "relaxed",
}

IMAGE_ALLOWED_ACTIONS = IMAGE_MAIN_POSES | IMAGE_EXTRA_POSES

# =========================
# 2. 基础工具函数
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

def _round_float(value: Any, digits: int = 3) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except Exception:
        return None

def _is_valid_point(point: Any) -> bool:
    if point is None:
        return False

    if isinstance(point, dict):
        point = point.get("xy")

    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False

    x = safe_float(point[0], None)
    y = safe_float(point[1], None)

    if x is None or y is None:
        return False

    if math.isnan(x) or math.isnan(y):
        return False

    return True

def _xy(point: Any) -> Optional[Tuple[float, float]]:
    if point is None:
        return None

    if isinstance(point, dict):
        point = point.get("xy")

    if not _is_valid_point(point):
        return None

    return safe_float(point[0]), safe_float(point[1])

def _point_from_kpts(kpts: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
    return _xy(kpts.get(name))

def _conf_from_kpts_or_confs(
    kpts: Dict[str, Any],
    confs: Optional[Dict[str, Any]],
    name: str,
    default: float = 1.0,
) -> float:
    if isinstance(confs, dict) and name in confs:
        return safe_float(confs.get(name), default)

    value = kpts.get(name)
    if isinstance(value, dict):
        return safe_float(value.get("conf"), default)

    return default

def _distance(p1: Any, p2: Any) -> Optional[float]:
    p1 = _xy(p1)
    p2 = _xy(p2)

    if p1 is None or p2 is None:
        return None

    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def _mean(values: List[Any]) -> Optional[float]:
    valid_values = []

    for value in values:
        value = safe_float(value, None)
        if value is not None:
            valid_values.append(value)

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)

def _angle_deg(p1: Any, p2: Any, p3: Any) -> Optional[float]:
    p1 = _xy(p1)
    p2 = _xy(p2)
    p3 = _xy(p3)

    if p1 is None or p2 is None or p3 is None:
        return None

    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    len1 = math.hypot(v1[0], v1[1])
    len2 = math.hypot(v2[0], v2[1])

    if len1 < 1e-6 or len2 < 1e-6:
        return None

    cos_value = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
    cos_value = max(-1.0, min(1.0, cos_value))

    return math.degrees(math.acos(cos_value))

def _segment_angle_abs(p1: Any, p2: Any) -> Optional[float]:
    p1 = _xy(p1)
    p2 = _xy(p2)

    if p1 is None or p2 is None:
        return None

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    return abs(math.degrees(math.atan2(dy, dx))) % 180

def _is_vertical_segment(
    p1: Any,
    p2: Any,
    min_angle: float = 65.0,
    max_angle: float = 115.0,
) -> bool:
    angle = _segment_angle_abs(p1, p2)
    if angle is None:
        return False
    return min_angle <= angle <= max_angle

def _is_horizontal_segment(
    p1: Any,
    p2: Any,
    max_angle: float = 30.0,
    min_reverse_angle: float = 150.0,
) -> bool:
    angle = _segment_angle_abs(p1, p2)
    if angle is None:
        return False
    return angle <= max_angle or angle >= min_reverse_angle

def _box_size(box: Any) -> Tuple[float, float]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 1.0, 1.0

    x1, y1, x2, y2 = [safe_float(v, 0.0) for v in box[:4]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    return width, height

def _normalize_y(point: Optional[Tuple[float, float]], box: Any) -> Optional[float]:
    if point is None:
        return None

    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None

    y1 = safe_float(box[1], 0.0)
    y2 = safe_float(box[3], y1 + 1.0)
    height = max(1.0, y2 - y1)

    return (point[1] - y1) / height

def _normalize_x(point: Optional[Tuple[float, float]], box: Any) -> Optional[float]:
    if point is None:
        return None

    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None

    x1 = safe_float(box[0], 0.0)
    x2 = safe_float(box[2], x1 + 1.0)
    width = max(1.0, x2 - x1)

    return (point[0] - x1) / width

# =========================
# 3. 关键点兼容
# =========================

def _get_body_points(kpts: Dict[str, Any]) -> Dict[str, Optional[Tuple[float, float]]]:
    names = [
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
        "withers",
        "tail_start",
        "tail_base",
        "hip",
        "shoulder",
    ]

    return {name: _point_from_kpts(kpts, name) for name in names}

def _collect_confs(kpts: Dict[str, Any], confs: Optional[Dict[str, Any]]) -> Dict[str, float]:
    names = set(kpts.keys())

    if isinstance(confs, dict):
        names |= set(confs.keys())

    return {
        name: _conf_from_kpts_or_confs(kpts, confs, name, 1.0)
        for name in names
    }

def _has_reliable_limb(
    conf_map: Dict[str, float],
    names: List[str],
    min_conf: float = 0.45,
    min_count: int = 2,
) -> bool:
    valid_count = 0

    for name in names:
        if safe_float(conf_map.get(name), 0.0) >= min_conf:
            valid_count += 1

    return valid_count >= min_count

def _leg_angle(elbow: Any, knee: Any, paw: Any) -> Optional[float]:
    return _angle_deg(elbow, knee, paw)

def _leg_extended(elbow: Any, knee: Any, paw: Any, threshold: float = 145.0) -> bool:
    angle = _leg_angle(elbow, knee, paw)
    if angle is None:
        return False
    return angle >= threshold

def _upper_leg_downward(elbow: Any, knee: Any) -> bool:
    return _is_vertical_segment(elbow, knee)

def _lower_leg_downward(paw: Any, knee: Any) -> bool:
    return _is_vertical_segment(knee, paw)

def _limb_horizontal(elbow: Any, knee: Any, paw: Any) -> bool:
    upper_horizontal = _is_horizontal_segment(elbow, knee)
    lower_horizontal = _is_horizontal_segment(knee, paw)

    angle = _leg_angle(elbow, knee, paw)
    extended = angle is not None and angle >= 130.0

    return upper_horizontal or lower_horizontal or (
        extended and (upper_horizontal or lower_horizontal)
    )

# =========================
# 4. 特征提取
# =========================

def extract_static_features(
    kpts: Dict[str, Any],
    confs: Optional[Dict[str, Any]] = None,
    box: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if not isinstance(kpts, dict):
        kpts = {}

    points = _get_body_points(kpts)
    conf_map = _collect_confs(kpts, confs)

    width, height = _box_size(box)
    box_ratio = width / max(1.0, height)

    front_left_paw = points["front_left_paw"]
    front_left_knee = points["front_left_knee"]
    front_left_elbow = points["front_left_elbow"]

    front_right_paw = points["front_right_paw"]
    front_right_knee = points["front_right_knee"]
    front_right_elbow = points["front_right_elbow"]

    rear_left_paw = points["rear_left_paw"]
    rear_left_knee = points["rear_left_knee"]
    rear_left_elbow = points["rear_left_elbow"]

    rear_right_paw = points["rear_right_paw"]
    rear_right_knee = points["rear_right_knee"]
    rear_right_elbow = points["rear_right_elbow"]

    nose = points["nose"]
    chin = points["chin"]
    left_ear_base = points["left_ear_base"]
    right_ear_base = points["right_ear_base"]
    left_ear_tip = points["left_ear_tip"]
    right_ear_tip = points["right_ear_tip"]

    withers = points["withers"] or points["shoulder"]
    tail_start = points["tail_start"] or points["tail_base"] or points["hip"]

    front_left_reliable = _has_reliable_limb(
        conf_map,
        ["front_left_elbow", "front_left_knee", "front_left_paw"],
    )
    front_right_reliable = _has_reliable_limb(
        conf_map,
        ["front_right_elbow", "front_right_knee", "front_right_paw"],
    )
    rear_left_reliable = _has_reliable_limb(
        conf_map,
        ["rear_left_elbow", "rear_left_knee", "rear_left_paw"],
    )
    rear_right_reliable = _has_reliable_limb(
        conf_map,
        ["rear_right_elbow", "rear_right_knee", "rear_right_paw"],
    )

    front_limb_reliable = front_left_reliable or front_right_reliable
    rear_limb_reliable = rear_left_reliable or rear_right_reliable

    front_left_angle = _leg_angle(front_left_elbow, front_left_knee, front_left_paw)
    front_right_angle = _leg_angle(front_right_elbow, front_right_knee, front_right_paw)
    rear_left_angle = _leg_angle(rear_left_elbow, rear_left_knee, rear_left_paw)
    rear_right_angle = _leg_angle(rear_right_elbow, rear_right_knee, rear_right_paw)

    front_angles = [v for v in [front_left_angle, front_right_angle] if v is not None]
    rear_angles = [v for v in [rear_left_angle, rear_right_angle] if v is not None]

    front_max_angle = max(front_angles) if front_angles else None
    rear_max_angle = max(rear_angles) if rear_angles else None

    front_extended = (
        (_leg_extended(front_left_elbow, front_left_knee, front_left_paw) and front_left_reliable)
        or (_leg_extended(front_right_elbow, front_right_knee, front_right_paw) and front_right_reliable)
    )

    rear_extended = (
        (_leg_extended(rear_left_elbow, rear_left_knee, rear_left_paw) and rear_left_reliable)
        or (_leg_extended(rear_right_elbow, rear_right_knee, rear_right_paw) and rear_right_reliable)
    )

    front_down = (
        (_upper_leg_downward(front_left_elbow, front_left_knee) and front_left_reliable)
        or (_upper_leg_downward(front_right_elbow, front_right_knee) and front_right_reliable)
    )

    rear_down = (
        (_upper_leg_downward(rear_left_elbow, rear_left_knee) and rear_left_reliable)
        or (_upper_leg_downward(rear_right_elbow, rear_right_knee) and rear_right_reliable)
    )

    front_lower_down = (
        (_lower_leg_downward(front_left_paw, front_left_knee) and front_left_reliable)
        or (_lower_leg_downward(front_right_paw, front_right_knee) and front_right_reliable)
    )

    rear_lower_down = (
        (_lower_leg_downward(rear_left_paw, rear_left_knee) and rear_left_reliable)
        or (_lower_leg_downward(rear_right_paw, rear_right_knee) and rear_right_reliable)
    )

    front_limb_horizontal = (
        (_limb_horizontal(front_left_elbow, front_left_knee, front_left_paw) and front_left_reliable)
        or (_limb_horizontal(front_right_elbow, front_right_knee, front_right_paw) and front_right_reliable)
    )

    rear_limb_horizontal = (
        (_limb_horizontal(rear_left_elbow, rear_left_knee, rear_left_paw) and rear_left_reliable)
        or (_limb_horizontal(rear_right_elbow, rear_right_knee, rear_right_paw) and rear_right_reliable)
    )

    front_paws = [p for p in [front_left_paw, front_right_paw] if p is not None]
    rear_paws = [p for p in [rear_left_paw, rear_right_paw] if p is not None]

    front_paw_y_values = [
        _normalize_y(p, box)
        for p in front_paws
        if _normalize_y(p, box) is not None
    ]
    rear_paw_y_values = [
        _normalize_y(p, box)
        for p in rear_paws
        if _normalize_y(p, box) is not None
    ]

    front_paw_y = max(front_paw_y_values) if front_paw_y_values else None
    rear_paw_y = max(rear_paw_y_values) if rear_paw_y_values else None

    head_points = [
        p for p in [
            nose,
            chin,
            left_ear_base,
            right_ear_base,
            left_ear_tip,
            right_ear_tip,
        ]
        if p is not None
    ]

    head_y_values = [
        _normalize_y(p, box)
        for p in head_points
        if _normalize_y(p, box) is not None
    ]
    head_y = _mean(head_y_values) if head_y_values else None

    nose_y = _normalize_y(nose, box)
    chin_y = _normalize_y(chin, box)

    body_angle = None
    body_horiz = None
    body_vert = None

    if withers is not None and tail_start is not None:
        body_angle = _segment_angle_abs(withers, tail_start)

        if body_angle is not None:
            body_horiz = body_angle <= 30.0 or body_angle >= 150.0
            body_vert = 60.0 <= body_angle <= 120.0
    else:
        if box_ratio > 1.25:
            body_horiz = True
            body_vert = False
        elif box_ratio < 0.85:
            body_horiz = False
            body_vert = True

    body_y_candidates = []

    for p in [
        front_left_elbow,
        front_right_elbow,
        rear_left_elbow,
        rear_right_elbow,
        withers,
        tail_start,
    ]:
        normalized_y = _normalize_y(p, box)
        if normalized_y is not None:
            body_y_candidates.append(normalized_y)

    body_y = _mean(body_y_candidates) if body_y_candidates else None

    head_to_body_y = None
    if head_y is not None and body_y is not None:
        head_to_body_y = head_y - body_y

    visible_points = 0
    visible_confs = []

    for name, point in points.items():
        if point is not None:
            visible_points += 1
            visible_confs.append(safe_float(conf_map.get(name), 1.0))

    avg_conf = _mean(visible_confs) if visible_confs else 0.0

    return {
        "box_ratio": box_ratio,
        "box_width": width,
        "box_height": height,

        "body_angle": body_angle,
        "body_horiz": body_horiz,
        "body_vert": body_vert,

        "front_left_angle": front_left_angle,
        "front_right_angle": front_right_angle,
        "rear_left_angle": rear_left_angle,
        "rear_right_angle": rear_right_angle,
        "front_max_angle": front_max_angle,
        "rear_max_angle": rear_max_angle,

        "front_extended": front_extended,
        "rear_extended": rear_extended,

        "front_down": front_down,
        "rear_down": rear_down,
        "front_lower_down": front_lower_down,
        "rear_lower_down": rear_lower_down,

        "front_limb_horizontal": front_limb_horizontal,
        "rear_limb_horizontal": rear_limb_horizontal,

        "front_limb_reliable": front_limb_reliable,
        "rear_limb_reliable": rear_limb_reliable,
        "front_left_reliable": front_left_reliable,
        "front_right_reliable": front_right_reliable,
        "rear_left_reliable": rear_left_reliable,
        "rear_right_reliable": rear_right_reliable,

        "front_paw_y": front_paw_y,
        "rear_paw_y": rear_paw_y,
        "head_y": head_y,
        "nose_y": nose_y,
        "chin_y": chin_y,
        "head_to_body_y": head_to_body_y,

        "has_front_paw": bool(front_paws),
        "has_rear_paw": bool(rear_paws),
        "has_head": bool(head_points),
        "has_tail_start": tail_start is not None,
        "has_withers": withers is not None,

        "visible_points": visible_points,
        "avg_conf": avg_conf,

        "points": points,
        "conf_map": conf_map,
    }

def _features_for_debug(feats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "box_ratio": _round_float(feats.get("box_ratio"), 3),
        "box_width": _round_float(feats.get("box_width"), 2),
        "box_height": _round_float(feats.get("box_height"), 2),

        "body_angle": _round_float(feats.get("body_angle"), 2),
        "body_horiz": feats.get("body_horiz"),
        "body_vert": feats.get("body_vert"),

        "front_left_angle": _round_float(feats.get("front_left_angle"), 2),
        "front_right_angle": _round_float(feats.get("front_right_angle"), 2),
        "rear_left_angle": _round_float(feats.get("rear_left_angle"), 2),
        "rear_right_angle": _round_float(feats.get("rear_right_angle"), 2),
        "front_max_angle": _round_float(feats.get("front_max_angle"), 2),
        "rear_max_angle": _round_float(feats.get("rear_max_angle"), 2),

        "front_extended": feats.get("front_extended"),
        "rear_extended": feats.get("rear_extended"),

        "front_down": feats.get("front_down"),
        "rear_down": feats.get("rear_down"),
        "front_lower_down": feats.get("front_lower_down"),
        "rear_lower_down": feats.get("rear_lower_down"),

        "front_limb_horizontal": feats.get("front_limb_horizontal"),
        "rear_limb_horizontal": feats.get("rear_limb_horizontal"),

        "front_limb_reliable": feats.get("front_limb_reliable"),
        "rear_limb_reliable": feats.get("rear_limb_reliable"),

        "front_paw_y": _round_float(feats.get("front_paw_y"), 3),
        "rear_paw_y": _round_float(feats.get("rear_paw_y"), 3),
        "head_y": _round_float(feats.get("head_y"), 3),
        "nose_y": _round_float(feats.get("nose_y"), 3),
        "chin_y": _round_float(feats.get("chin_y"), 3),
        "head_to_body_y": _round_float(feats.get("head_to_body_y"), 3),

        "has_front_paw": feats.get("has_front_paw"),
        "has_rear_paw": feats.get("has_rear_paw"),
        "has_head": feats.get("has_head"),

        "visible_points": feats.get("visible_points"),
        "avg_conf": _round_float(feats.get("avg_conf"), 3),
    }

# =========================
# 5. 静态姿态分类
# =========================

def classify_pose_image(
    kpts: Dict[str, Any],
    confs: Optional[Dict[str, Any]] = None,
    box: Optional[List[float]] = None,
    return_debug: bool = False,
) -> Any:
    try:
        feats = extract_static_features(kpts, confs, box)

        box_ratio = feats.get("box_ratio", 1.0)
        body_horiz = feats.get("body_horiz")
        body_vert = feats.get("body_vert")

        front_down = feats.get("front_down", False)
        rear_down = feats.get("rear_down", False)
        front_lower_down = feats.get("front_lower_down", False)
        rear_lower_down = feats.get("rear_lower_down", False)

        front_paw_y = feats.get("front_paw_y")
        rear_paw_y = feats.get("rear_paw_y")
        head_y = feats.get("head_y")
        nose_y = feats.get("nose_y")
        chin_y = feats.get("chin_y")
        head_to_body_y = feats.get("head_to_body_y")

        has_front_paw = feats.get("has_front_paw", False)
        has_rear_paw = feats.get("has_rear_paw", False)
        has_head = feats.get("has_head", False)

        front_limb_horizontal = feats.get("front_limb_horizontal", False)
        rear_limb_horizontal = feats.get("rear_limb_horizontal", False)

        front_limb_reliable = feats.get("front_limb_reliable", True)
        rear_limb_reliable = feats.get("rear_limb_reliable", True)

        visible_points = safe_int(feats.get("visible_points"), 0)
        avg_conf = safe_float(feats.get("avg_conf"), 0.0)

        scores = {
            "lie_down": 0.0,
            "stand": 0.0,
            "sit": 0.0,
        }
        reasons = {key: [] for key in scores}

        extra_scores = {
            "look_up": 0.0,
            "head_down": 0.0,
            "alert": 0.0,
            "relaxed": 0.0,
        }
        extra_reasons = {key: [] for key in extra_scores}

        def add(pose: str, value: float, reason: str) -> None:
            scores[pose] += float(value)
            reasons[pose].append(reason)

        def add_extra(pose: str, value: float, reason: str) -> None:
            extra_scores[pose] += float(value)
            extra_reasons[pose].append(reason)

        # lie_down
        if box_ratio > 1.65:
            add("lie_down", 4.5, "检测框明显横向，符合趴下")
        elif box_ratio > 1.35:
            add("lie_down", 3.0, "检测框偏横向，可能趴下")
        elif box_ratio > 1.15:
            add("lie_down", 1.2, "检测框略横向")

        if body_horiz is True:
            add("lie_down", 2.0, "身体方向偏水平")

        if front_limb_horizontal and box_ratio > 1.05:
            add("lie_down", 2.5, "前腿横向伸展，符合趴下")

        if rear_limb_horizontal and box_ratio > 1.1:
            add("lie_down", 1.5, "后腿横向伸展，符合趴下")

        if has_front_paw and not has_rear_paw and box_ratio > 1.15:
            add("lie_down", 1.0, "前爪可见后爪不可见且身体横向")

        if front_paw_y is not None and front_paw_y < 0.72 and box_ratio > 1.1:
            add("lie_down", 1.0, "前爪不在底部且身体偏横向")

        # stand
        front_full_stand = front_down and front_lower_down
        rear_full_stand = rear_down and rear_lower_down

        if front_full_stand and rear_full_stand:
            add("stand", 4.0, "前后腿上下段均竖直，符合站立承重")
        elif front_full_stand or rear_full_stand:
            add("stand", 2.0, "至少一组腿上下段竖直")
        elif front_down and rear_down:
            add("stand", 1.2, "仅上段腿竖直，弱站立特征")
        elif front_down or rear_down:
            add("stand", 0.8, "存在竖直腿部，弱站立特征")

        if box_ratio < 1.2:
            add("stand", 0.8, "检测框不横向")

        if front_full_stand and front_paw_y is not None and front_paw_y > 0.75:
            add("stand", 1.0, "前腿承重且前爪靠底")

        if rear_full_stand and rear_paw_y is not None and rear_paw_y > 0.75:
            add("stand", 1.0, "后腿承重且后爪靠底")

        if not front_limb_reliable and not rear_limb_reliable:
            scores["stand"] -= 1.0
            reasons["stand"].append("腿部关键点整体不可靠，降低站立分")

        # sit
        if front_down and not rear_lower_down:
            add("sit", 3.5, "前腿竖直，后腿下段收折，典型坐姿")

        if front_full_stand and not rear_full_stand:
            add("sit", 2.0, "前腿承重，后腿不承重，偏坐下")

        if box_ratio < 0.95 and front_paw_y is not None and front_paw_y > 0.72:
            add("sit", 2.0, "竖向框且前爪靠底，符合坐姿")

        if box_ratio < 0.95 and not rear_limb_reliable:
            add("sit", 1.5, "竖向框且后腿关键点不可靠，偏坐姿")

        if box_ratio < 0.9 and front_down:
            add("sit", 1.5, "竖向框且前腿竖直，符合坐姿支撑")

        if body_vert is True:
            add("sit", 1.5, "身体方向偏竖直")

        if front_down and not has_rear_paw:
            add("sit", 1.2, "前腿竖直且后爪不可见，可能坐下")

        if rear_paw_y is not None and rear_paw_y < 0.7:
            add("sit", 1.5, "后爪位置较高，可能后腿折叠")

        # 附加静态状态
        if has_head:
            if nose_y is not None and chin_y is not None:
                if nose_y < chin_y - 0.03:
                    add_extra("look_up", 0.8, "鼻子高于下巴，头部上扬")
                elif nose_y > chin_y + 0.04:
                    add_extra("head_down", 0.8, "鼻子低于下巴，头部下垂")

            if head_y is not None:
                if head_y < 0.38:
                    add_extra("look_up", 0.4, "头部位于检测框上方")
                elif head_y > 0.55:
                    add_extra("head_down", 0.4, "头部位置偏低")

            if head_to_body_y is not None:
                if head_to_body_y < -0.18:
                    add_extra("look_up", 0.4, "头部明显高于身体")
                elif head_to_body_y > -0.02:
                    add_extra("head_down", 0.4, "头部接近或低于身体")

        if avg_conf >= 0.65 and visible_points >= 8:
            add_extra("alert", 1.0, "关键点清晰且头部/身体可见，偏警觉")
        elif avg_conf >= 0.45 and visible_points >= 6:
            add_extra("alert", 0.5, "关键点较完整，可能警觉")

        if scores["lie_down"] >= 3.0 and extra_scores["alert"] < 1.0:
            add_extra("relaxed", 1.0, "趴下且警觉特征不强，偏放松")

        # 消歧：stand vs sit
        if not rear_lower_down and scores["stand"] >= 2.0:
            scores["stand"] -= 2.5
            reasons["stand"].append("后腿下段不竖直，不符合站立承重，降低站立分")

        if front_down and not rear_lower_down and scores["sit"] >= 2.0:
            scores["sit"] += 1.0
            reasons["sit"].append("前腿支撑后腿收折，补强坐下")

        if box_ratio < 1.0 and front_down and not rear_full_stand:
            scores["sit"] += 1.0
            reasons["sit"].append("竖向框+前腿支撑+后腿非完整承重，补强坐下")

        # 消歧：stand vs lie_down
        if box_ratio > 1.35 and (front_limb_horizontal or rear_limb_horizontal):
            scores["stand"] -= 3.0
            reasons["stand"].append("检测框横向且腿横向伸展，排除站立")

        if box_ratio > 1.35 and not (front_full_stand and rear_full_stand):
            scores["stand"] -= 2.0
            reasons["stand"].append("横向框且缺少完整承重腿，降低站立分")

        # 消歧：sit vs lie_down
        if scores["sit"] >= 2.0 and scores["lie_down"] >= 2.0:
            if box_ratio > 1.25:
                scores["sit"] -= 1.5
                reasons["sit"].append("检测框偏横向，降低坐下分")
            elif box_ratio < 1.0:
                scores["lie_down"] -= 1.0
                reasons["lie_down"].append("检测框偏竖向，降低趴下分")

        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score < 2.0:
            best = "unknown"
            best_score = 0.0

        actions = []

        if best != "unknown":
            actions.append({
                "pose": best,
                "pose_cn": POSE_CN.get(best, "未知"),
                "score": _round_float(best_score, 3),
                "type": "main",
            })

        for pose, score in extra_scores.items():
            threshold = 0.7

            if pose in {"alert", "relaxed"}:
                threshold = 0.8

            if score >= threshold:
                actions.append({
                    "pose": pose,
                    "pose_cn": POSE_CN.get(pose, pose),
                    "score": _round_float(score, 3),
                    "type": "extra",
                })

        actions = filter_image_actions(actions)

        result = {
            "pose": best,
            "pose_cn": POSE_CN.get(best, "未知"),
            "actions": actions,
            "scores": {
                key: _round_float(value, 3)
                for key, value in scores.items()
            },
            "extra_scores": {
                key: _round_float(value, 3)
                for key, value in extra_scores.items()
            },
            "reasons": reasons.get(best, []),
            "extra_reasons": extra_reasons,
            "features": _features_for_debug(feats),
        }

        return result if return_debug else best

    except Exception as exc:
        if return_debug:
            return {
                "pose": "error",
                "pose_cn": POSE_CN.get("error", "错误"),
                "actions": [],
                "scores": {},
                "extra_scores": {},
                "reasons": [str(exc)],
                "extra_reasons": {},
                "features": {},
            }

        return "error"

# =========================
# 6. 兼容函数
# =========================

def classify_pose(
    kpts: Dict[str, Any],
    confs: Optional[Dict[str, Any]] = None,
    box: Optional[List[float]] = None,
    return_debug: bool = False,
) -> Any:
    return classify_pose_image(
        kpts,
        confs=confs,
        box=box,
        return_debug=return_debug,
    )

def filter_image_actions(actions: Any) -> List[Dict[str, Any]]:
    if not isinstance(actions, list):
        return []

    filtered = []

    for action in actions:
        if not isinstance(action, dict):
            continue

        pose = action.get("pose")
        if pose not in IMAGE_ALLOWED_ACTIONS:
            continue

        filtered.append(action)

    return filtered

def calc_pose_confidence(keypoints: Dict[str, Any]) -> float:
    if not isinstance(keypoints, dict) or not keypoints:
        return 0.0

    important_names = [
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
        "nose",
        "chin",
    ]

    confs = []

    for name in important_names:
        value = keypoints.get(name)
        if isinstance(value, dict):
            confs.append(safe_float(value.get("conf"), 0.0))

    if not confs:
        for value in keypoints.values():
            if isinstance(value, dict) and "conf" in value:
                confs.append(safe_float(value.get("conf"), 0.0))

    if not confs:
        return 0.0

    avg_conf = sum(confs) / len(confs)
    visible_ratio = sum(1 for conf in confs if conf >= 0.35) / max(1, len(confs))

    confidence = 0.65 * avg_conf + 0.35 * visible_ratio
    confidence = max(0.0, min(1.0, confidence))

    return round(confidence, 3)

# =========================
# 7. 绘图
# =========================

_KEYPOINT_COLORS = {
    "front_left_paw": (0, 255, 255),
    "front_left_knee": (0, 255, 255),
    "front_left_elbow": (0, 255, 255),

    "front_right_paw": (0, 220, 255),
    "front_right_knee": (0, 220, 255),
    "front_right_elbow": (0, 220, 255),

    "rear_left_paw": (255, 180, 0),
    "rear_left_knee": (255, 180, 0),
    "rear_left_elbow": (255, 180, 0),

    "rear_right_paw": (255, 120, 0),
    "rear_right_knee": (255, 120, 0),
    "rear_right_elbow": (255, 120, 0),

    "nose": (0, 255, 0),
    "chin": (0, 200, 0),
    "left_ear_base": (255, 0, 255),
    "right_ear_base": (255, 0, 255),
    "left_ear_tip": (255, 80, 255),
    "right_ear_tip": (255, 80, 255),
}

_SKELETON = [
    ("front_left_elbow", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),

    ("front_right_elbow", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),

    ("rear_left_elbow", "rear_left_knee"),
    ("rear_left_knee", "rear_left_paw"),

    ("rear_right_elbow", "rear_right_knee"),
    ("rear_right_knee", "rear_right_paw"),

    ("left_ear_tip", "left_ear_base"),
    ("right_ear_tip", "right_ear_base"),
    ("left_ear_base", "nose"),
    ("right_ear_base", "nose"),
    ("nose", "chin"),
]

def _draw_text(
    image,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = org

    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def _extract_xy_for_draw(value: Any) -> Optional[Tuple[int, int]]:
    point = _xy(value)

    if point is None:
        return None

    return int(round(point[0])), int(round(point[1]))

def draw_analysis(image, dogs: List[Dict[str, Any]], save_path: str) -> str:
    """
    绘制静态姿态分析图。

    推荐调用方式：
    draw_analysis(image, dogs_result, save_path)

    dogs_result 每个元素建议包含：
    {
        "box": [...],
        "box_int": [...],
        "keypoints": {...},
        "pose": "sit",
        "pose_cn": "坐下",
        "pose_actions": [...],
        "pose_confidence": 0.8
    }
    """
    if image is None:
        return save_path

    canvas = image.copy()

    if not isinstance(dogs, list):
        dogs = []

    for idx, dog in enumerate(dogs):
        if not isinstance(dog, dict):
            continue

        box = dog.get("box_int") or dog.get("box") or []
        keypoints = dog.get("keypoints", {}) or {}

        pose = dog.get("pose") or dog.get("pose_label") or "unknown"
        pose_cn = dog.get("pose_cn") or dog.get("pose_label_cn") or POSE_CN.get(pose, "未知")
        pose_confidence = safe_float(dog.get("pose_confidence"), 0.0)

        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = [safe_int(v, 0) for v in box[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"Dog {idx + 1}: {pose_cn}({pose}) {pose_confidence:.2f}"
            _draw_text(
                canvas,
                label,
                (x1, max(18, y1 - 8)),
                (255, 255, 255),
                0.52,
                1,
            )

        for p1_name, p2_name in _SKELETON:
            p1 = _extract_xy_for_draw(keypoints.get(p1_name))
            p2 = _extract_xy_for_draw(keypoints.get(p2_name))

            if p1 is None or p2 is None:
                continue

            color = _KEYPOINT_COLORS.get(p1_name, (0, 255, 255))
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

        for name, value in keypoints.items():
            point = _extract_xy_for_draw(value)

            if point is None:
                continue

            conf = 1.0
            if isinstance(value, dict):
                conf = safe_float(value.get("conf"), 1.0)

            if conf < 0.2:
                continue

            color = _KEYPOINT_COLORS.get(name, (255, 255, 0))
            radius = 4 if conf >= 0.5 else 3

            cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)

        actions = filter_image_actions(dog.get("pose_actions", []))

        if actions and isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = [safe_int(v, 0) for v in box[:4]]

            action_text = " + ".join([
                str(action.get("pose_cn", action.get("pose", "未知")))
                for action in actions
                if isinstance(action, dict)
            ])

            _draw_text(
                canvas,
                action_text,
                (x1, min(canvas.shape[0] - 8, y2 + 18)),
                (0, 255, 255),
                0.5,
                1,
            )

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(save_path, canvas)

    return save_path

# =========================
# 8. 测试入口
# =========================

if __name__ == "__main__":
    print("dog_pose_analyzer.py loaded.")
    print("This module only handles static dog pose analysis.")
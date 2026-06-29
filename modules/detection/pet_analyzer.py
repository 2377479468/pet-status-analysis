# -*- coding: utf-8 -*-
"""
pet_analyzer.py

宠物分析统一入口。

职责：
1. 判断输入是图片还是视频。
2. 调用 video_process.py 完成基础猫狗检测、最佳帧、裁剪图、视频动态分析。
3. 如果识别为狗，则调用 predict_dog_pose.py 对图片/最佳帧做静态狗姿态分析。
4. 视频结果中同时返回：
   - pose_result：最佳帧静态姿态
   - video_result：多帧动态行为
5. 猫暂时不跑 dog-pose。
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# =========================
# 1. 文件类型配置
# =========================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
}

# =========================
# 2. 基础工具
# =========================

def normalize_path(path: Any) -> Optional[str]:
    if path is None:
        return None

    if path == "":
        return ""

    try:
        return Path(str(path)).as_posix()
    except Exception:
        return str(path)

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

def get_input_type(input_path: Any) -> str:
    suffix = Path(str(input_path)).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return "image"

    if suffix in VIDEO_EXTENSIONS:
        return "video"

    return "unknown"

def empty_pose_result(
    animal: str = "unknown",
    message: str = "未进行姿态分析。",
) -> Dict[str, Any]:
    return {
        "enabled": False,
        "animal": animal,
        "pose_label": "unknown",
        "pose_label_cn": "未知",
        "pose_actions": [],
        "pose_confidence": 0.0,
        "keypoints": {},
        "keypoints_list": [],
        "points": [],
        "box": [],
        "box_int": [],
        "dog_count": 0,
        "analysis_image_path": None,
        "source_pose_image_path": None,
        "json_path": None,
        "message": message,
        "raw_result": {},
    }

def empty_video_result(message: str = "非视频输入，未进行视频动态分析。") -> Dict[str, Any]:
    return {
        "enabled": False,
        "main_behavior": "unknown",
        "main_behavior_cn": "未知",
        "motion_state": "unknown",
        "motion_state_cn": "未知",
        "trend": "unknown",
        "trend_cn": "未知",
        "behaviors": [],
        "motion_score": 0,
        "direction_score": 0,
        "shake_score": 0,
        "turning_score": 0,
        "stretch_score": 0,
        "summary": message,

        "pose_label": "unknown",
        "pose_label_cn": "未知",
        "frame_count": 0,
        "valid_frame_count": 0,
        "fps": 0.0,
        "avg_motion": 0.0,
        "avg_speed": 0.0,
    }

# =========================
# 3. 动态加载模块
# =========================

def load_video_process():
    try:
        from modules.detection.video_process import analyze_image, analyze_video
        return analyze_image, analyze_video, None
    except Exception as exc:
        return None, None, f"video_process.py 加载失败: {exc}"

def load_predict_dog_pose():
    try:
        from modules.detection.dog_pose.predict_dog_pose import predict_dog_pose
        return predict_dog_pose, None
    except Exception as exc:
        return None, f"predict_dog_pose.py 加载失败: {exc}"

# =========================
# 4. 姿态分析
# =========================

def should_run_dog_pose(animal: str) -> bool:
    return animal == "dog"

def choose_pose_image_path(
    input_type: str,
    input_path: str,
    base_result: Dict[str, Any],
) -> Optional[str]:
    if input_type == "image":
        if input_path and Path(input_path).exists():
            return input_path

        best_frame_path = base_result.get("best_frame_path")
        if best_frame_path and Path(str(best_frame_path)).exists():
            return str(best_frame_path)

        return None

    if input_type == "video":
        best_frame_path = base_result.get("best_frame_path")
        if best_frame_path and Path(str(best_frame_path)).exists():
            return str(best_frame_path)

        return None

    return None

def normalize_pose_result(
    pose_result: Dict[str, Any],
    source_pose_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    pose_result["source_pose_image_path"] = normalize_path(source_pose_image_path)

    pose_result.setdefault("enabled", True)
    pose_result.setdefault("animal", "dog")
    pose_result.setdefault("pose_label", "unknown")
    pose_result.setdefault("pose_label_cn", "未知")
    pose_result.setdefault("pose_actions", [])
    pose_result.setdefault("pose_confidence", 0.0)
    pose_result.setdefault("keypoints", {})
    pose_result.setdefault("keypoints_list", [])
    pose_result.setdefault("points", [])
    pose_result.setdefault("box", [])
    pose_result.setdefault("box_int", [])
    pose_result.setdefault("dog_count", 0)
    pose_result.setdefault("analysis_image_path", None)
    pose_result.setdefault("json_path", None)
    pose_result.setdefault("message", "ok")
    pose_result.setdefault("raw_result", {})

    pose_result["analysis_image_path"] = normalize_path(pose_result.get("analysis_image_path"))
    pose_result["json_path"] = normalize_path(pose_result.get("json_path"))

    return pose_result

def run_pose_analysis(
    input_type: str,
    input_path: str,
    animal: str,
    base_result: Dict[str, Any],
) -> Dict[str, Any]:
    if not should_run_dog_pose(animal):
        return empty_pose_result(
            animal=animal,
            message="当前不是狗，跳过 dog-pose 静态姿态分析。",
        )

    predict_dog_pose, load_error = load_predict_dog_pose()

    if load_error:
        return empty_pose_result(
            animal="dog",
            message=load_error,
        )

    pose_image_path = choose_pose_image_path(
        input_type=input_type,
        input_path=input_path,
        base_result=base_result,
    )

    if not pose_image_path:
        return empty_pose_result(
            animal="dog",
            message="未找到可用于 dog-pose 的图片帧。",
        )

    try:
        pose_result = predict_dog_pose(pose_image_path)

        if not isinstance(pose_result, dict):
            return empty_pose_result(
                animal="dog",
                message="predict_dog_pose 返回结果格式错误。",
            )

        return normalize_pose_result(
            pose_result=pose_result,
            source_pose_image_path=pose_image_path,
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
            "box": [],
            "box_int": [],
            "dog_count": 0,
            "analysis_image_path": None,
            "json_path": None,
            "message": f"dog-pose 预测失败: {exc}",
            "traceback": traceback.format_exc(),
            "raw_result": {},
            "source_pose_image_path": normalize_path(pose_image_path),
        }

# =========================
# 5. 结果合并
# =========================

def normalize_video_result(
    input_type: str,
    base_result: Dict[str, Any],
) -> Dict[str, Any]:
    video_result = base_result.get("video_result")

    if input_type != "video":
        return empty_video_result()

    if not isinstance(video_result, dict):
        return empty_video_result("视频动态结果格式异常。")

    default_result = empty_video_result("视频动态分析未返回摘要。")
    default_result.update(video_result)

    default_result["motion_score"] = safe_int(default_result.get("motion_score"), 0)
    default_result["direction_score"] = safe_int(default_result.get("direction_score"), 0)
    default_result["shake_score"] = safe_int(default_result.get("shake_score"), 0)
    default_result["turning_score"] = safe_int(default_result.get("turning_score"), 0)
    default_result["stretch_score"] = safe_int(default_result.get("stretch_score"), 0)
    default_result["frame_count"] = safe_int(default_result.get("frame_count"), 0)
    default_result["valid_frame_count"] = safe_int(default_result.get("valid_frame_count"), 0)
    default_result["fps"] = safe_float(default_result.get("fps"), 0.0)
    default_result["avg_motion"] = safe_float(default_result.get("avg_motion"), 0.0)
    default_result["avg_speed"] = safe_float(default_result.get("avg_speed"), 0.0)

    return default_result

def build_final_result(
    input_type: str,
    input_path: str,
    base_result: Dict[str, Any],
    pose_result: Dict[str, Any],
) -> Dict[str, Any]:
    animal = base_result.get("animal", "unknown")
    animal_cn = base_result.get("animal_cn", "未知")
    video_result = normalize_video_result(input_type, base_result)

    result = {
        "success": bool(base_result.get("success", True)),
        "input_type": input_type,
        "input_path": normalize_path(input_path),

        "animal": animal,
        "animal_cn": animal_cn,
        "animal_confidence": safe_float(base_result.get("animal_confidence"), 0.0),

        "best_frame_path": normalize_path(base_result.get("best_frame_path")),
        "crop_path": normalize_path(base_result.get("crop_path")),
        "annotated_video_path": normalize_path(base_result.get("annotated_video_path")),
        "box": base_result.get("box", []),

        "motion_score": safe_int(
            base_result.get("motion_score"),
            safe_int(video_result.get("motion_score"), 0),
        ),
        "legacy_motion_score": safe_int(
            base_result.get("legacy_motion_score"),
            safe_int(base_result.get("motion_score"), 0),
        ),

        "image_quality": safe_int(base_result.get("image_quality"), 0),

        "trend": base_result.get(
            "trend",
            video_result.get("trend", "unknown"),
        ),
        "legacy_trend": base_result.get(
            "legacy_trend",
            base_result.get("trend", video_result.get("trend", "unknown")),
        ),
        "trend_comment": base_result.get("trend_comment", ""),
        "legacy_trend_comment": base_result.get(
            "legacy_trend_comment",
            base_result.get("trend_comment", ""),
        ),

        "pose_result": pose_result,
        "video_result": video_result,

        "sampled_frames": base_result.get("sampled_frames", []),
        "pose_frames": base_result.get("pose_frames", []),

        "pose_frame_count": safe_int(base_result.get("pose_frame_count"), 0),
        "annotated_pose_frame_count": safe_int(base_result.get("annotated_pose_frame_count"), 0),
        "pose_analysis_fps": safe_float(base_result.get("pose_analysis_fps"), 0.0),
        "annotated_pose_fps": safe_float(base_result.get("annotated_pose_fps"), 0.0),
        "output_video_fps": safe_float(base_result.get("output_video_fps"), 0.0),

        "fps": safe_float(base_result.get("fps"), 0.0),
        "total_frames": safe_int(base_result.get("total_frames"), 0),
        "sampled_count": safe_int(base_result.get("sampled_count"), 0),

        "message": base_result.get("message", "ok"),
    }

    return result

def save_full_result_json(result: Dict[str, Any]) -> Optional[str]:
    try:
        output_dir = Path("static/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        input_type = result.get("input_type", "unknown")
        animal = result.get("animal", "unknown")
        output_path = output_dir / f"pet_result_{input_type}_{animal}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return output_path.as_posix()
    except Exception:
        return None

def build_display_result(result: Dict[str, Any]) -> Dict[str, Any]:
    pose_result = result.get("pose_result") or {}
    video_result = result.get("video_result") or {}

    static_pose = {
        "enabled": pose_result.get("enabled", False),
        "main_pose": pose_result.get("pose_label", "unknown"),
        "main_pose_cn": pose_result.get("pose_label_cn", "未知"),
        "confidence": safe_float(pose_result.get("pose_confidence"), 0.0),
        "actions": pose_result.get("pose_actions", []),
        "dog_count": safe_int(pose_result.get("dog_count"), 0),
    }

    behavior = {
        "enabled": video_result.get("enabled", False),
        "main_behavior": video_result.get("main_behavior", "unknown"),
        "main_behavior_cn": video_result.get("main_behavior_cn", "未知"),
        "motion_state": video_result.get("motion_state", "unknown"),
        "motion_state_cn": video_result.get("motion_state_cn", "未知"),
        "behaviors": video_result.get("behaviors", []),
        "trend": video_result.get("trend", "unknown"),
        "trend_cn": video_result.get("trend_cn", "未知"),
        "motion_score": safe_int(video_result.get("motion_score"), 0),
        "avg_motion": safe_float(video_result.get("avg_motion"), 0.0),
        "avg_speed": safe_float(video_result.get("avg_speed"), 0.0),
        "frame_count": safe_int(video_result.get("frame_count"), 0),
        "valid_frame_count": safe_int(video_result.get("valid_frame_count"), 0),
        "fps": safe_float(video_result.get("fps"), 0.0),
        "summary": video_result.get("summary"),
    }

    return {
        "success": result.get("success"),
        "input_type": result.get("input_type"),
        "input_path": result.get("input_path"),

        "animal": result.get("animal"),
        "animal_cn": result.get("animal_cn"),
        "animal_confidence": result.get("animal_confidence"),

        "static_pose": static_pose,
        "video_behavior": behavior,

        "video_meta": {
            "fps": result.get("fps"),
            "total_frames": result.get("total_frames"),
            "sampled_count": result.get("sampled_count"),
            "pose_frame_count": result.get("pose_frame_count"),
            "annotated_pose_frame_count": result.get("annotated_pose_frame_count"),
            "pose_analysis_fps": result.get("pose_analysis_fps"),
            "annotated_pose_fps": result.get("annotated_pose_fps"),
            "output_video_fps": result.get("output_video_fps"),
        },

        "files": {
            "best_frame_path": result.get("best_frame_path"),
            "crop_path": result.get("crop_path"),
            "annotated_video_path": result.get("annotated_video_path"),
            "analysis_image_path": pose_result.get("analysis_image_path"),
            "source_pose_image_path": pose_result.get("source_pose_image_path"),
            "pose_json_path": pose_result.get("json_path"),
            "full_json_path": result.get("full_json_path"),
        },

        "message": result.get("message"),
    }

# =========================
# 6. 统一入口函数
# =========================

def analyze_pet(input_path: Any) -> Dict[str, Any]:
    try:
        input_path_obj = Path(str(input_path))

        if not input_path_obj.exists():
            return {
                "success": False,
                "message": f"输入文件不存在: {normalize_path(input_path_obj)}",
                "input_path": normalize_path(input_path_obj),
            }

        input_type = get_input_type(input_path_obj)

        if input_type == "unknown":
            return {
                "success": False,
                "message": f"不支持的文件类型: {input_path_obj.suffix}",
                "input_path": normalize_path(input_path_obj),
            }

        analyze_image, analyze_video, load_error = load_video_process()

        if load_error:
            return {
                "success": False,
                "message": load_error,
                "input_path": normalize_path(input_path_obj),
            }

        if input_type == "image":
            base_result = analyze_image(str(input_path_obj))
        else:
            base_result = analyze_video(str(input_path_obj))

        if not isinstance(base_result, dict):
            return {
                "success": False,
                "message": "video_process 返回结果格式错误。",
                "input_path": normalize_path(input_path_obj),
            }

        animal = base_result.get("animal", "unknown")

        pose_result = run_pose_analysis(
            input_type=input_type,
            input_path=str(input_path_obj),
            animal=animal,
            base_result=base_result,
        )

        final_result = build_final_result(
            input_type=input_type,
            input_path=str(input_path_obj),
            base_result=base_result,
            pose_result=pose_result,
        )

        full_json_path = save_full_result_json(final_result)
        final_result["full_json_path"] = full_json_path

        return final_result

    except Exception as exc:
        return {
            "success": False,
            "message": f"analyze_pet 执行失败: {exc}",
            "input_path": normalize_path(input_path),
            "traceback": traceback.format_exc(),
        }

# =========================
# 7. 兼容别名
# =========================

def analyze(input_path: Any) -> Dict[str, Any]:
    return analyze_pet(input_path)

def run(input_path: Any) -> Dict[str, Any]:
    return analyze_pet(input_path)

# =========================
# 8. 命令行测试
# =========================

if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "modules/detection/dog_pose/test_images/dog6.jpg"

    result = analyze_pet(input_path)
    display_result = build_display_result(result)

    print("===== 宠物状态识别摘要 =====")
    print(json.dumps(display_result, ensure_ascii=False, indent=4))
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import time
import glob
from pathlib import Path

# 导入 B 模块
from modules.detection.pet_analyzer import analyze_pet

# 导入 C 模块（情绪预测）
from modules.emotion.emotion_predictor import predict_emotion

# ==================== 导入 E 模块 ====================
try:
    from modules.scoring.score_calculator import calculate_scores
    from modules.scoring.text_generator import generate_comment
    E_AVAILABLE = True
except ModuleNotFoundError as e:
    E_AVAILABLE = False
    print(f"警告：E 模块导入失败 ({e})，将使用内置备用逻辑")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/results", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ==================== 工具函数 ====================
def to_relative_url(abs_path):
    if not abs_path:
        return ""
    abs_path = str(abs_path).replace("\\", "/")
    
    if abs_path.startswith("/static/"):
        return f"{abs_path}?t={int(time.time())}"
    if "static/" in abs_path:
        idx = abs_path.find("static/")
        if idx != -1:
            url = "/" + abs_path[idx:]
            return f"{url}?t={int(time.time())}"
    if "dog_pose/outputs/" in abs_path:
        try:
            filename = os.path.basename(abs_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{int(time.time())}{ext}"
            dest_path = f"static/results/{new_filename}"
            os.makedirs("static/results", exist_ok=True)
            shutil.copy2(abs_path, dest_path)
            return f"/{dest_path}?t={int(time.time())}"
        except Exception:
            return ""
    if abs_path.startswith("static/"):
        return f"/{abs_path}?t={int(time.time())}"
    return abs_path


def get_latest_image(pattern):
    files = glob.glob(f"static/results/{pattern}")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_latest_video(pattern="*.mp4"):
    files = glob.glob(f"static/results/{pattern}")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ==================== 内置备用函数 ====================
def fallback_calculate_scores(emotion_probs, motion_score, image_quality, animal_confidence, animal):
    if animal == "dog":
        good_index_name = "好狗指数"
    elif animal == "cat":
        good_index_name = "好猫指数"
    else:
        good_index_name = "好感指数"
    scores = {
        "开心指数": round(emotion_probs.get("happy", 0) * 100, 1),
        "活跃指数": motion_score if motion_score else 50,
        "生气指数": round(emotion_probs.get("angry", 0) * 100, 1),
        "疑惑指数": round(emotion_probs.get("anxious", 0) * 80 + 20, 1),
        "警觉指数": round(emotion_probs.get("anxious", 0) * 70 + emotion_probs.get("angry", 0) * 30, 1),
        good_index_name: round(
            emotion_probs.get("happy", 0) * 40 +
            emotion_probs.get("relaxed", 0) * 30 +
            (image_quality / 100) * 20 +
            animal_confidence * 10,
            1
        )
    }
    return scores

def fallback_generate_comment(animal, scores):
    if animal == "dog":
        pet_name = "狗狗"
        good_index = scores.get("好狗指数", 0)
        if good_index >= 80:
            return f"这只{pet_name}状态超棒！好狗指数高达{good_index}，快乐又乖巧，是只难得的小天使～"
        elif good_index >= 60:
            return f"这只{pet_name}状态不错，好狗指数{good_index}，活泼可爱，继续保持哦！"
        elif good_index >= 40:
            return f"这只{pet_name}状态一般，好狗指数{good_index}，可能有点紧张，多陪陪它吧。"
        else:
            return f"这只{pet_name}好狗指数{good_index}，状态不太好，可能需要休息或安抚一下～"
    elif animal == "cat":
        pet_name = "猫咪"
        good_index = scores.get("好猫指数", 0)
        if good_index >= 80:
            return f"这只{pet_name}太棒啦！好猫指数{good_index}，轻松自在，完全享受生活～"
        elif good_index >= 60:
            return f"这只{pet_name}状态不错，好猫指数{good_index}，优雅从容，是个合格的猫主子。"
        elif good_index >= 40:
            return f"这只{pet_name}好猫指数{good_index}，有点小紧张，可能需要一个安静的环境。"
        else:
            return f"这只{pet_name}好猫指数{good_index}，状态不太好，多给它一点时间和空间吧。"
    else:
        return "未能识别宠物类型，请上传清晰的猫或狗视频。"


# ==================== 主接口 ====================
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # 1. 保存上传的文件
    file_ext = Path(file.filename).suffix or ".mp4"
    unique_id = uuid.uuid4()
    video_filename = f"{unique_id}{file_ext}"
    video_path = f"uploads/{video_filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 调用 B 模块
    try:
        b_result_raw = analyze_pet(video_path)
    except Exception as e:
        return {"success": False, "message": f"B 模块分析失败: {str(e)}"}

    if not b_result_raw.get("success", False):
        return {"success": False, "message": b_result_raw.get("message", "B 模块返回失败")}

    # 提取 B 的结果
    input_type = b_result_raw.get("input_type", "unknown")
    animal = b_result_raw.get("animal", "unknown")
    animal_confidence = b_result_raw.get("animal_confidence", 0.0)
    best_frame_abs = b_result_raw.get("best_frame_path", "")
    crop_abs = b_result_raw.get("crop_path", "")
    
    # 获取 motion_score
    motion_score = b_result_raw.get("motion_score", 0)
    if not motion_score:
        motion_score = b_result_raw.get("legacy_motion_score", 0)
    image_quality = b_result_raw.get("image_quality", 80)

    # 处理最佳帧路径
    if not best_frame_abs or not os.path.exists(best_frame_abs):
        latest = get_latest_image("*_best_frame.jpg")
        if latest:
            best_frame_abs = latest
            print(f"自动选择最新最佳帧: {best_frame_abs}")
        else:
            best_frame_abs = ""

    best_frame_url = to_relative_url(best_frame_abs)

    # 3. 调用 C 模块（情绪预测）
    emotion_probs = None
    if crop_abs and Path(crop_abs).exists():
        try:
            emotion_probs = predict_emotion(crop_abs)
        except Exception:
            emotion_probs = {"happy": 0.25, "angry": 0.25, "relaxed": 0.25, "anxious": 0.25}
    else:
        emotion_probs = {"happy": 0.25, "angry": 0.25, "relaxed": 0.25, "anxious": 0.25}

    # 4. 计算六维指数和生成评论
    if E_AVAILABLE:
        try:
            scores = calculate_scores(
                emotion_probs=emotion_probs,
                motion_score=motion_score,
                image_quality=image_quality,
                animal_confidence=animal_confidence,
                animal=animal
            )
            comment = generate_comment(animal, scores)
        except Exception as e:
            print(f"E 模块调用失败，使用内置备用逻辑: {e}")
            scores = fallback_calculate_scores(
                emotion_probs=emotion_probs,
                motion_score=motion_score,
                image_quality=image_quality,
                animal_confidence=animal_confidence,
                animal=animal
            )
            comment = fallback_generate_comment(animal, scores)
    else:
        scores = fallback_calculate_scores(
            emotion_probs=emotion_probs,
            motion_score=motion_score,
            image_quality=image_quality,
            animal_confidence=animal_confidence,
            animal=animal
        )
        comment = fallback_generate_comment(animal, scores)

    # ================== ★★★ 从 video_result 和 pose_result 提取数据 ★★★ ==================
    # B 模块的数据在 video_result 和 pose_result 中
    video_result = b_result_raw.get("video_result", {})
    pose_result = b_result_raw.get("pose_result", {})

    # ----- 构建 video_behavior -----
    if video_result and isinstance(video_result, dict):
        # 提取 behavior_timeline 并确保包含 start_label 和 end_label
        behavior_timeline = video_result.get("behavior_timeline", [])
        if not behavior_timeline:
            # 如果 behavior_timeline 为空，从 behaviors 转换
            raw_behaviors = video_result.get("behaviors", [])
            for b in raw_behaviors:
                if isinstance(b, dict):
                    timeline_item = {
                        "start_time": b.get("start_time", 0),
                        "end_time": b.get("end_time", 0),
                        "duration": b.get("duration", 0),
                        "behavior": b.get("behavior", ""),
                        "behavior_cn": b.get("behavior_cn", ""),
                        "confidence": b.get("confidence", 1.0),
                        "start_label": f"{int(b.get('start_time', 0)):02d}:{int((b.get('start_time', 0) % 1) * 60):02d}",
                        "end_label": f"{int(b.get('end_time', 0)):02d}:{int((b.get('end_time', 0) % 1) * 60):02d}"
                    }
                    behavior_timeline.append(timeline_item)

        video_behavior = {
            "enabled": True,
            "main_behavior": video_result.get("main_behavior", ""),
            "main_behavior_cn": video_result.get("main_behavior_cn", ""),
            "motion_state": video_result.get("motion_state", ""),
            "motion_state_cn": video_result.get("motion_state_cn", ""),
            "behaviors": video_result.get("behaviors", []),
            "trend": video_result.get("trend", ""),
            "trend_cn": video_result.get("trend_cn", ""),
            "motion_score": video_result.get("motion_score", motion_score),
            "direction_score": video_result.get("direction_score", 0),
            "shake_score": video_result.get("shake_score", 0),
            "turning_score": video_result.get("turning_score", 0),
            "stretch_score": video_result.get("stretch_score", 0),
            "avg_motion": video_result.get("avg_motion", 0),
            "avg_speed": video_result.get("avg_speed", 0),
            "frame_count": video_result.get("frame_count", 0),
            "valid_frame_count": video_result.get("valid_frame_count", 0),
            "fps": video_result.get("fps", 0),
            "summary": video_result.get("summary", ""),
            "behavior_timeline": behavior_timeline
        }
        print("✅ 从 video_result 构建 video_behavior")
    else:
        # 如果 video_result 不存在，构建旧格式
        video_behavior = {
            "motion_score": motion_score,
            "direction_score": 0,
            "shake_score": 0,
            "turning_score": 0,
            "stretch_score": 0
        }
        print("🔄 回退构建旧格式 video_behavior")

    # ----- 构建 static_pose -----
    if pose_result and isinstance(pose_result, dict):
        # 提取姿态数据
        pose_actions = pose_result.get("pose_actions", [])
        # 构建 actions 列表（前端的 actions 格式）
        actions_list = []
        for action in pose_actions:
            if isinstance(action, dict):
                actions_list.append({
                    "pose": action.get("pose", ""),
                    "pose_cn": action.get("pose_cn", ""),
                    "score": action.get("score", 0),
                    "type": action.get("type", "")
                })

        static_pose = {
            "enabled": True,
            "main_pose": pose_result.get("pose_label", ""),
            "main_pose_cn": pose_result.get("pose_label_cn", ""),
            "confidence": pose_result.get("pose_confidence", 0.0),
            "actions": actions_list,
            "dog_count": pose_result.get("dog_count", 1),
            "keypoints": pose_result.get("keypoints", {})
        }
        print("✅ 从 pose_result 构建 static_pose")
    else:
        # 如果 pose_result 不存在，构建旧格式
        static_pose = {
            "main_pose": "未知",
            "main_pose_score": 0,
            "head_status": "未知",
            "head_status_score": 0,
            "mental_state": "未知",
            "mental_state_score": 0,
            "pose_confidence": 0.0
        }
        print("🔄 回退构建旧格式 static_pose")

    # ================== detection_url 与 detection_video_url 处理 ==================
    detection_url = best_frame_url
    # 使用 pose_result 中的 analysis_image_path
    analysis_image = pose_result.get("analysis_image_path", "")
    if analysis_image and os.path.exists(analysis_image):
        detection_url = to_relative_url(analysis_image)
    else:
        latest_dog_pose = get_latest_image("*dog_pose*.jpg")
        if latest_dog_pose:
            detection_url = to_relative_url(latest_dog_pose)
        else:
            detection_url = best_frame_url

    # 获取标注视频
    annotated_video_path = b_result_raw.get("annotated_video_path", "")
    if annotated_video_path and os.path.exists(annotated_video_path):
        detection_video_url = to_relative_url(annotated_video_path)
        print(f"✅ 使用 B 模块标注视频: {annotated_video_path}")
    else:
        # 回退：查找 static/results/ 下最新的 MP4
        video_candidates = glob.glob("static/results/*.mp4")
        if video_candidates:
            latest_video = max(video_candidates, key=os.path.getmtime)
            detection_video_url = to_relative_url(latest_video)
            print(f"自动选择最新标注视频: {latest_video}")
        else:
            detection_video_url = f"/uploads/{video_filename}?t={int(time.time())}"

    original_video_url = f"/uploads/{video_filename}?t={int(time.time())}"

    # 7. 返回最终结果
    return {
        "success": True,
        "input_type": input_type,
        "animal": animal,
        "animal_confidence": round(animal_confidence, 4),
        "best_frame_url": best_frame_url,
        "detection_url": detection_url,
        "original_video_url": original_video_url,
        "detection_video_url": detection_video_url,
        "emotion_probs": emotion_probs,
        "video_behavior": video_behavior,
        "static_pose": static_pose,
        "scores": scores,
        "comment": comment
    }


# ==================== 测试接口 ====================
@app.post("/test-c")
async def test_c(file: UploadFile = File(...)):
    test_path = f"uploads/test_{uuid.uuid4()}.jpg"
    with open(test_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        result = predict_emotion(test_path)
        return {"status": "success", "emotion_probs": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
def root():
    return {"message": "宠物状态分析后端已启动！请访问 /static/index.html 查看前端页面。"}


if __name__ == "__main__":
    import uvicorn
    # host/port 可用环境变量覆盖：
    #   Windows 上端口 8000 有时被系统保留(WinError 10013)，可改用其它端口，例如
    #   PowerShell:  $env:PORT=8080; python main.py
    #   CMD:         set PORT=8080 && python main.py
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
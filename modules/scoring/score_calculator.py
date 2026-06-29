"""
六维状态指数计算模块
路径：modules/scoring/score_calculator.py

A 模块调用方式：
    from modules.scoring.score_calculator import calculate_scores
    scores = calculate_scores(emotion_probs, motion_score, image_quality, animal_confidence, animal)
"""

def calculate_scores(emotion_probs, motion_score, image_quality,
                     animal_confidence, animal):
    """
    根据C的情绪概率和B的运动分数，计算六维状态指数

    参数:
        emotion_probs:   dict, C的返回 {"happy": 0.72, "angry": 0.08, ...}
        motion_score:    int,  B的返回中的 motion_score (0~100)
        image_quality:   int,  B的返回中的 image_quality (0~100)
        animal_confidence: float, B的返回中的 animal_confidence (0~1)
        animal:          str,  "dog" 或 "cat"

    返回:
        dict, 六维指数，每个值 0~100，取整
    """
    # 提取概率
    happy = emotion_probs.get("happy", 0.0)
    angry = emotion_probs.get("angry", 0.0)
    relaxed = emotion_probs.get("relaxed", 0.0)
    anxious = emotion_probs.get("anxious", 0.0)

    # 模型不确定性：1 减去最大概率
    model_uncertainty = 1.0 - max(happy, angry, relaxed, anxious)

    # ---- 六维指数计算 ----
    # 1. 开心指数（0~100）
    happy_score = round(happy * 100)

    # 2. 生气指数（0~100）
    angry_score = round(angry * 100)

    # 3. 活跃指数（0~100），直接使用B的motion_score
    active_score = int(motion_score)

    # 4. 警觉指数（0~100）
    alert_score = round(anxious * 70 + angry * 30)

    # 5. 疑惑指数（0~100）
    #    焦虑概率 × 50 + 模型不确定性 × 30
    confusion_score = round(anxious * 50 + model_uncertainty * 30)

    # 6. 好狗/好猫指数（0~100）
    good_score = round(
        happy_score * 0.4 +
        relaxed * 100 * 0.3 +
        image_quality * 0.2 +
        animal_confidence * 100 * 0.1
    )

    # 构建返回字典
    good_label = "好狗指数" if animal == "dog" else "好猫指数"

    scores = {
        "开心指数": happy_score,
        "活跃指数": active_score,
        "生气指数": angry_score,
        "疑惑指数": confusion_score,
        "警觉指数": alert_score,
        good_label: good_score,
    }

    # 确保所有值在 0~100 范围内
    for key in scores:
        scores[key] = max(0, min(100, scores[key]))

    return scores


# ============ 测试 ============
if __name__ == "__main__":
    # 模拟C的输出
    fake_emotion = {
        "happy": 0.72,
        "angry": 0.08,
        "relaxed": 0.15,
        "anxious": 0.05,
    }

    # 模拟B的输出
    result = calculate_scores(
        emotion_probs=fake_emotion,
        motion_score=72,
        image_quality=85,
        animal_confidence=0.93,
        animal="dog",
    )

    print("六维指数计算结果:")
    for name, val in result.items():
        bar = "█" * (val // 2)
        print(f"  {name}: {val:3d}  {bar}")

    # 输出示例：
    # 开心指数:  82  █████████████████████████████████████████
    # 活跃指数:  72  ████████████████████████████████████
    # 生气指数:   8  ████
    # 疑惑指数:  14  ███████
    # 警觉指数:   5  ██
    # 好狗指数:  72  ████████████████████████████████████
"""测试 C → E 完整链路（从项目根目录运行：python -m modules.emotion.test_c_e）"""
from modules.emotion.emotion_predictor import predict_emotion_fake
from modules.scoring.score_calculator import calculate_scores


# 1. 模拟C的输出
emotion = predict_emotion_fake()
print("C 情绪概率:", emotion)

# 2. E 计算六维指数
scores = calculate_scores(emotion, motion_score=72,
                          image_quality=85, animal_confidence=0.93,
                          animal="dog")
print("\nE 六维指数:")
for k, v in scores.items():
    print(f"  {k}: {v}")
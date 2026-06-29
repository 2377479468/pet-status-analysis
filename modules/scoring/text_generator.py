"""
趣味文案生成模块
路径：modules/scoring/text_generator.py

A 模块调用方式：
    from modules.scoring.text_generator import generate_comment
    comment = generate_comment(animal, scores)
"""

import random


def generate_comment(animal, scores):
    """
    根据六维指数生成趣味评价文案

    参数:
        animal: str, "dog" 或 "cat"
        scores: dict, E 的 calculate_scores() 返回的六维指数字典

    返回:
        str, 一段中文评价文案
    """
    animal_name = '狗狗' if animal == 'dog' else '猫咪'
    pronoun = '它'

    # 找到最高分维度（排除好狗/好猫指数）
    pure_scores = {k: v for k, v in scores.items()
                   if not k.startswith('好狗') and not k.startswith('好猫')}
    top_dim = max(pure_scores, key=pure_scores.get)
    top_val = pure_scores[top_dim]

    # 综合评分
    good_key = '好狗指数' if animal == 'dog' else '好猫指数'
    good_val = scores.get(good_key, 50)

    # ---- 根据最高分维度生成文案 ----
    if top_dim == '开心指数':
        if top_val >= 80:
            comments = [
                '这是一只心情超棒的' + animal_name + '！' + pronoun + '看起来非常开心，笑容满面，状态满分！',
                '哇！' + animal_name + '的开心指数爆表！' + pronoun + '一定刚吃完好吃的，或者正准备出去玩~',
                '好开心的' + animal_name + '！' + pronoun + '的尾巴一定摇得像小风扇一样！',
            ]
        else:
            comments = [
                '这只' + animal_name + '看起来心情不错，开心指数偏高，整体状态比较放松愉快。',
                animal_name + '今天心情挺好的，表情放松，状态良好。',
            ]

    elif top_dim == '生气指数':
        if top_val >= 60:
            comments = [
                '嗯...这只' + animal_name + '好像有点小脾气！' + pronoun + '的生气指数偏高，可能是没吃到零食或者被吵醒了~',
                '注意！' + animal_name + '正在生闷气模式中，建议用零食安抚一下' + pronoun + '的情绪！',
            ]
        else:
            comments = [
                '这只' + animal_name + '看起来有一点点不高兴，但整体还好，可能只是没睡醒~',
                animal_name + '的表情有点严肃，不过没有明显的不开心，再观察观察吧。',
            ]

    elif top_dim == '警觉指数':
        comments = [
            '这只' + animal_name + '处于比较警觉的状态，' + pronoun + '可能注意到了什么有趣的东西，正在认真观察周围环境。',
            animal_name + '的警觉指数较高，' + pronoun + '可能在守护家园，或者听到了奇怪的声音~',
            '嘘...' + animal_name + '正在侦察模式中！' + pronoun + '的耳朵竖得直直的，非常专注！',
        ]

    elif top_dim == '疑惑指数':
        comments = [
            '这只' + animal_name + '看起来有点困惑，' + pronoun + '的疑惑指数偏高，可能是看到了什么让' + pronoun + '不理解的东西~',
            animal_name + '的表情有点复杂，' + pronoun + '好像在想这是啥，能吃吗？',
            '哈哈，' + animal_name + '一脸迷茫！' + pronoun + '的疑惑指数告诉我们，' + pronoun + '正在思考喵生/狗生~',
        ]

    elif top_dim == '活跃指数':
        if top_val >= 70:
            comments = [
                '这只' + animal_name + '活力满满！' + pronoun + '的活跃指数很高，一定是一只特别爱运动的' + animal_name + '！',
                '能量爆棚的' + animal_name + '！看' + pronoun + '的活动量，主人平时一定没少带' + pronoun + '出去玩~',
            ]
        else:
            comments = [
                '这只' + animal_name + '比较安静，活跃指数偏低，可能是累了或者在休息。',
                animal_name + '今天看起来比较慵懒，活动量不大，可能是个喜欢葛优躺的' + animal_name + '~',
            ]

    else:
        comments = [
            '这是一只状态不错的' + animal_name + '，各项指数都比较均衡。',
        ]

    # ---- 追加综合评分评价 ----
    if good_val >= 80:
        suffix = ' 综合来看，' + pronoun + '是一只非常棒的' + animal_name + '！'
    elif good_val >= 60:
        suffix = ' 总的来说，' + animal_name + '的状态还不错。'
    elif good_val >= 40:
        suffix = ' ' + animal_name + '的状态一般，多陪陪' + pronoun + '吧~'
    else:
        suffix = ' ' + animal_name + '今天状态不太好，可能需要主人的关爱哦~'

    # 随机选一条评论 + 后缀
    comment = random.choice(comments) + suffix

    return comment


# ============ 测试 ============
if __name__ == '__main__':
    from modules.scoring.score_calculator import calculate_scores

    # 模拟开心的狗狗
    fake_emotion = {
        'happy': 0.72,
        'angry': 0.08,
        'relaxed': 0.15,
        'anxious': 0.05,
    }

    scores_dog = calculate_scores(
        emotion_probs=fake_emotion,
        motion_score=72,
        image_quality=85,
        animal_confidence=0.93,
        animal='dog',
    )
    print('=== 狗狗测试 ===')
    print(generate_comment('dog', scores_dog))
    print()

    # 模拟生气的猫咪
    fake_emotion_angry = {
        'happy': 0.05,
        'angry': 0.78,
        'relaxed': 0.02,
        'anxious': 0.15,
    }
    scores_cat = calculate_scores(
        emotion_probs=fake_emotion_angry,
        motion_score=25,
        image_quality=70,
        animal_confidence=0.88,
        animal='cat',
    )
    print('=== 猫咪测试 ===')
    print(generate_comment('cat', scores_cat))
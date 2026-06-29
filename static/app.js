/**
 * ===================================================
 *  宠物状态分析系统 - 全屏左右分栏 · 前端逻辑
 *  支持图片/视频上传 · 姿态VS对比 · 检测图展示
 * ===================================================
 */

// ==================== 配置 ====================

// 同源访问：前端由后端在同一端口伺服，手机/电脑只要打开 http://<电脑IP>:8000/static/index.html
// 即可自动命中同一台后端，无需写死 IP。本地调试也可临时改成 'http://localhost:8000'。
const API_BASE = window.location.origin;
const ANALYZE_URL = `${API_BASE}/analyze`;
const USE_MOCK = false;

// ==================== 假数据 ====================

const MOCK_DOG_ACTIVE = {
  success: true,
  animal: 'dog',
  animal_confidence: 0.912,
  best_frame_url: '/static/demo/dog_active.jpg',
  detection_url: '/static/demo/dog_active_pose.jpg',
  detection_video_url: null,
  emotion_probs: { happy: 0.60, angry: 0.03, relaxed: 0.25, anxious: 0.12 },
  video_behavior: {
    motion_score: 72,
    direction_score: 38,
    shake_score: 0,
    turning_score: 35,
    stretch_score: 0
  },
  static_pose: {
    main_pose: '站立',
    main_pose_score: 82,
    head_status: '抬头',
    head_status_score: 65,
    mental_state: '警觉',
    mental_state_score: 71,
    pose_confidence: 0.78
  },
  scores: {
    '开心指数': 75, '活跃指数': 72, '生气指数': 10,
    '疑惑指数': 28, '警觉指数': 35, '好狗指数': 85
  }
};

const MOCK_DOG_LAZY = {
  success: true,
  animal: 'dog',
  animal_confidence: 0.958,
  best_frame_url: '/static/demo/dog_lazy.jpg',
  detection_url: '/static/demo/dog_lazy_pose.jpg',
  detection_video_url: null,
  emotion_probs: { happy: 0.40, angry: 0.02, relaxed: 0.55, anxious: 0.03 },
  video_behavior: {
    motion_score: 12,
    direction_score: 3,
    shake_score: 0,
    turning_score: 0,
    stretch_score: 42
  },
  static_pose: {
    main_pose: '趴下',
    main_pose_score: 90,
    head_status: '低头',
    head_status_score: 88,
    mental_state: '放松',
    mental_state_score: 85,
    pose_confidence: 0.85
  },
  scores: {
    '开心指数': 60, '活跃指数': 18, '生气指数': 5,
    '疑惑指数': 15, '警觉指数': 22, '好狗指数': 82
  }
};

const MOCK_CAT = {
  success: true,
  animal: 'cat',
  animal_confidence: 0.89,
  best_frame_url: '/static/demo/cat.jpg',
  detection_url: '/static/demo/cat.jpg',
  detection_video_url: null,
  emotion_probs: { happy: 0.35, angry: 0.12, relaxed: 0.48, anxious: 0.05 },
  video_behavior: {
    motion_score: 22,
    direction_score: 8,
    shake_score: 0,
    turning_score: 0,
    stretch_score: 0
  },
  static_pose: {
    main_pose: '坐下',
    main_pose_score: 75,
    head_status: '抬头',
    head_status_score: 58,
    mental_state: '放松',
    mental_state_score: 80,
    pose_confidence: 0.72
  },
  scores: {
    '开心指数': 41, '活跃指数': 35, '生气指数': 14,
    '疑惑指数': 22, '警觉指数': 18, '好猫指数': 72
  }
};

const MOCK_LIST = [MOCK_DOG_ACTIVE, MOCK_DOG_LAZY, MOCK_CAT];

// ==================== 状态 ====================

let selectedFile = null;
let selectedFileType = null; // 'image' | 'video'
let imageBlobUrl = null;    // 图片上传后的本地预览 URL，作为最佳帧
let radarChartInstance = null;

// ==================== DOM 引用 ====================

const $ = (s) => document.querySelector(s);

// 上传
const uploadArea = $('#uploadArea');
const fileInput = $('#fileInput');
const imagePreview = $('#imagePreview');
const previewImage = $('#previewImage');
const imageName = $('#imageName');
const removeImageBtn = $('#removeImage');
const videoPreview = $('#videoPreview');
const previewVideo = $('#previewVideo');
const videoName = $('#videoName');
const removeVideoBtn = $('#removeVideo');
const analyzeBtn = $('#analyzeBtn');
const demoBtn = $('#demoBtn');

// 右面板
const rightEmpty = $('#rightEmpty');
const loadingOverlay = $('#loadingOverlay');
const errorOverlay = $('#errorOverlay');
const rightResults = $('#rightResults');

// 加载
const progressFill = $('#progressFill');
const progressPercent = $('#progressPercent');
const step1 = $('#step1'), step2 = $('#step2'), step3 = $('#step3'), step4 = $('#step4');

// 结果
const resultBanner = $('#resultBanner');
const dogIcon = $('#dogIcon'), catIcon = $('#catIcon');
const animalName = $('#animalName'), animalType = $('#animalType');
const confidenceValue = $('#confidenceValue');
const bestFrameImg = $('#bestFrameImg');
const framePlaceholder = $('#framePlaceholder');
const frameTitle = $('#frameTitle');
const detectionImg = $('#detectionImg');
const detectionVideo = $('#detectionVideo');
const detectionPlaceholder = $('#detectionPlaceholder');
const detectionTitle = $('#detectionTitle');
const staticPose = $('#staticPose');
const metricsRow = $('#metricsRow');
const staticBlock = $('#staticBlock');
const dynamicBlock = $('#dynamicBlock');
const dynamicBehavior = $('#dynamicBehavior');
const scoresGrid = $('#scoresGrid');
const commentText = $('#commentText');
const resetBtn = $('#resetBtn');

// 错误
const errorText = $('#errorText');
const retryBtn = $('#retryBtn');

// ==================== 右面板状态切换 ====================

function showRightState(state) {
  rightEmpty.style.display = 'none';
  loadingOverlay.style.display = 'none';
  errorOverlay.style.display = 'none';
  rightResults.style.display = 'none';

  if (state === 'empty')    rightEmpty.style.display = '';
  if (state === 'loading')  loadingOverlay.style.display = '';
  if (state === 'error')    errorOverlay.style.display = '';
  if (state === 'results')  rightResults.style.display = '';
}

// ==================== 上传 ====================

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  handleFile(file);
});

function handleFile(file) {
  selectedFile = file;
  const isVideo = file.type.startsWith('video/');
  selectedFileType = isVideo ? 'video' : 'image';

  // 隐藏上传区
  uploadArea.style.display = 'none';

  if (isVideo) {
    videoName.textContent = file.name;
    previewVideo.src = URL.createObjectURL(file);
    videoPreview.style.display = '';
    imagePreview.style.display = 'none';
    imageBlobUrl = null;
  } else {
    imageName.textContent = file.name;
    const url = URL.createObjectURL(file);
    imageBlobUrl = url;
    previewImage.src = url;
    imagePreview.style.display = '';
    videoPreview.style.display = 'none';
  }

  analyzeBtn.style.display = '';
}

function resetUpload() {
  selectedFile = null;
  selectedFileType = null;
  imageBlobUrl = null;
  fileInput.value = '';
  previewVideo.src = '';
  previewImage.src = '';
  uploadArea.style.display = '';
  imagePreview.style.display = 'none';
  videoPreview.style.display = 'none';
  analyzeBtn.style.display = 'none';
}

removeVideoBtn.addEventListener('click', resetUpload);
removeImageBtn.addEventListener('click', resetUpload);

// ==================== 演示按钮 ====================

demoBtn.addEventListener('click', () => {
  const mock = MOCK_LIST[Math.floor(Math.random() * MOCK_LIST.length)];
  showRightState('loading');
  animateSteps(() => { displayResult(mock); });
});

// ==================== 分析按钮 ====================

analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;

  if (USE_MOCK) {
    const mock = MOCK_LIST[Math.floor(Math.random() * MOCK_LIST.length)];
    showRightState('loading');
    animateSteps(() => displayResult(mock));
    return;
  }

  showRightState('loading');
  startLoading();

  const fd = new FormData();
  fd.append('file', selectedFile);

  fetch(ANALYZE_URL, { method: 'POST', body: fd })
    .then(r => r.ok ? r.json() : Promise.reject(new Error('服务器错误: ' + r.status)))
    .then(data => {
      if (!data.success) throw new Error(data.message || '分析失败');
      stopLoading();
      displayResult(data);
    })
    .catch(err => {
      stopLoading();
      showError(err.message || '网络连接失败');
      console.error(err);
    });
});

// ==================== 加载动画（进度条） ====================

function resetProgress() {
  progressFill.style.width = '0%';
  progressPercent.textContent = '0%';
  [step1, step2, step3, step4].forEach(s => { s.classList.remove('active', 'done'); });
  step1.classList.add('active');
}

function setProgress(pct, activeStepIndex) {
  progressFill.style.width = pct + '%';
  progressPercent.textContent = pct + '%';
  if (activeStepIndex !== undefined && activeStepIndex >= 0 && activeStepIndex < 4) {
    [step1, step2, step3, step4].forEach((s, i) => {
      s.classList.remove('active', 'done');
      if (i < activeStepIndex) s.classList.add('done');
      else if (i === activeStepIndex) s.classList.add('active');
    });
  }
}

function startLoading() {
  resetProgress();

  const stages = [
    { pct: 25, step: 0, delay: 0 },
    { pct: 50, step: 1, delay: 1200 },
    { pct: 75, step: 2, delay: 2400 },
    { pct: 90, step: 3, delay: 3600 },
  ];

  stages.forEach(({ pct, step, delay }) => {
    setTimeout(() => setProgress(pct, step), delay);
  });
}

function stopLoading() {
  // 瞬间填满到 100%
  setProgress(100, 3);
  [step1, step2, step3, step4].forEach(s => {
    s.classList.remove('active'); s.classList.add('done');
  });
}

function animateSteps(cb) {
  resetProgress();

  const stages = [
    { pct: 25, step: 0, delay: 0 },
    { pct: 50, step: 1, delay: 500 },
    { pct: 75, step: 2, delay: 1000 },
    { pct: 90, step: 3, delay: 1500 },
    { pct: 100, step: 3, delay: 2000 },
  ];

  stages.forEach(({ pct, step, delay }) => {
    setTimeout(() => setProgress(pct, step), delay);
  });

  setTimeout(() => {
    [step1, step2, step3, step4].forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    cb();
  }, 2500);
}

// ==================== 辅助函数：URL 加上时间戳 ====================
function addCacheBuster(url) {
  if (!url) return '';
  const separator = url.includes('?') ? '&' : '?';
  return url + separator + '_t=' + Date.now();
}

// ==================== 结果展示 ====================

function displayResult(data) {

  showRightState('results');

  // 1. 动物横幅
  const isDog = data.animal === 'dog';
  dogIcon.style.display = isDog ? '' : 'none';
  catIcon.style.display = isDog ? 'none' : '';
  animalName.textContent = isDog ? '狗狗' : '猫咪';
  animalType.textContent = isDog ? '犬类' : '猫类';
  confidenceValue.textContent = Math.round(data.animal_confidence * 100) + '%';
  resultBanner.className = 'result-banner ' + (isDog ? 'banner-dog' : 'banner-cat');

  // 2. 行1：图片→原图+检测图 / 视频→最佳帧检测+视频检测结果
  const isVideo = (selectedFileType === 'video');
  if (isVideo) {
    // ===== 视频模式：左侧显示姿态检测图（带线条），右侧显示标注视频 =====
    frameTitle.textContent = '最佳帧检测';
    detectionTitle.textContent = '视频检测结果';

    // 左侧：显示 detection_url（姿态图，带线条）
    const poseImgUrl = data.detection_url || data.best_frame_url;
    showImg(bestFrameImg, framePlaceholder, poseImgUrl);

    // ===== 右侧：优先使用 annotated_video_url，并添加时间戳防缓存 =====
    const rawVideoUrl = data.annotated_video_url || data.detection_video_url || data.original_video_url;
    const videoUrl = addCacheBuster(rawVideoUrl); // 添加时间戳

    if (videoUrl) {
      // 显示视频
      detectionVideo.src = videoUrl;
      detectionVideo.style.display = 'block';
      detectionVideo.controls = true;   // 显示控制条
      detectionVideo.autoplay = false;  // 避免自动播放被拦截
      detectionPlaceholder.style.display = 'none';
      detectionImg.style.display = 'none';
      detectionImg.src = '';

      // 视频加载成功
      detectionVideo.onloadeddata = () => {
        detectionPlaceholder.style.display = 'none';
        // 尝试自动播放（如果浏览器允许）
        detectionVideo.play().catch(() => {});
      };

      // 视频加载失败 → 降级到显示姿态检测图
      detectionVideo.onerror = () => {
        detectionVideo.style.display = 'none';
        const fallbackImg = data.detection_url || data.best_frame_url;
        if (fallbackImg) {
          detectionImg.src = fallbackImg;
          detectionImg.style.display = 'block';
          detectionPlaceholder.textContent = '视频无法播放，显示姿态检测图';
          detectionPlaceholder.style.display = 'block';
        } else {
          detectionPlaceholder.textContent = '视频加载失败，请检查网络或格式';
          detectionPlaceholder.style.display = 'block';
        }
      };

      // 显式加载
      detectionVideo.load();
    } else {
      // 无视频URL → 显示图片
      detectionVideo.style.display = 'none';
      detectionVideo.src = '';
      const fallbackImg = data.detection_url || data.best_frame_url;
      if (fallbackImg) {
        detectionImg.src = fallbackImg;
        detectionImg.style.display = 'block';
        detectionPlaceholder.textContent = '暂无标注视频，显示姿态检测图';
        detectionPlaceholder.style.display = 'block';
      } else {
        detectionPlaceholder.textContent = '暂无结果';
        detectionPlaceholder.style.display = 'block';
      }
    }
  } else {
    // ===== 图片模式：左侧显示原图，右侧显示姿态检测图 =====
    frameTitle.textContent = '最佳帧';
    detectionTitle.textContent = '姿态检测结果';

    const imgUrl = (selectedFileType === 'image' && imageBlobUrl) ? imageBlobUrl : data.best_frame_url;
    showImg(bestFrameImg, framePlaceholder, imgUrl);
    showDetectionImg(data.detection_url);
  }

  // 3. 行2：静态姿态指标 + 动态行为指标（视频时才有）
  if (isVideo) {
    // 视频：双列显示
    metricsRow.classList.add('split');
    dynamicBlock.style.display = '';
    renderStaticPose(data.static_pose || {});
    renderVideoBehavior(data.video_behavior || {});
  } else {
    // 图片：单列全宽
    metricsRow.classList.remove('split');
    dynamicBlock.style.display = 'none';
    renderStaticPose(data.static_pose || {});
  }

  // 4. 雷达图 + 详细指数
  renderRadarChart(data.scores, data.animal);
  renderScores(data.scores);

  // 5. 综合评价
  commentText.textContent = generateComment(data.animal, data.scores);
}

function showImg(imgEl, placeholderEl, url) {
  if (url) {
    imgEl.src = url;
    imgEl.style.display = '';
    placeholderEl.style.display = 'none';
    imgEl.onerror = () => { imgEl.style.display = 'none'; placeholderEl.style.display = ''; };
  } else {
    imgEl.style.display = 'none';
    placeholderEl.style.display = '';
  }
}

function showDetectionImg(url) {
  detectionVideo.style.display = 'none';
  detectionVideo.src = '';
  if (url) {
    detectionImg.src = url;
    detectionImg.style.display = '';
    detectionPlaceholder.style.display = 'none';
    detectionImg.onerror = () => { detectionImg.style.display = 'none'; detectionPlaceholder.style.display = ''; };
  } else {
    detectionImg.style.display = 'none';
    detectionPlaceholder.style.display = '';
  }
}

function showDetectionVideo(url) {
  detectionImg.style.display = 'none';
  detectionImg.src = '';
  if (url) {
    detectionVideo.src = url;
    detectionVideo.style.display = '';
    detectionPlaceholder.style.display = 'none';
    detectionVideo.onerror = () => { detectionVideo.style.display = 'none'; detectionPlaceholder.style.display = ''; };
  } else {
    detectionVideo.style.display = 'none';
    detectionPlaceholder.style.display = '';
  }
}
// ==================== 综合评价生成（基于 text_generator.py） ====================

function generateComment(animal, scores) {
  const animalName = animal === 'dog' ? '狗狗' : '猫咪';
  const pronoun = '它';

  // 找到最高分维度（排除好狗/好猫指数）
  const pureScores = {};
  for (const k in scores) {
    if (!k.startsWith('好狗') && !k.startsWith('好猫')) {
      pureScores[k] = scores[k];
    }
  }
  let topDim = '';
  let topVal = 0;
  for (const k in pureScores) {
    if (pureScores[k] > topVal) { topDim = k; topVal = pureScores[k]; }
  }

  // 综合评分
  const goodKey = animal === 'dog' ? '好狗指数' : '好猫指数';
  const goodVal = scores[goodKey] || 50;

  // ---- 根据最高分维度生成文案 ----
  let pool = [];
  if (topDim === '开心指数') {
    if (topVal >= 80) {
      pool = [
        '这是一只心情超棒的' + animalName + '！' + pronoun + '看起来非常开心，笑容满面，状态满分！',
        '哇！' + animalName + '的开心指数爆表！' + pronoun + '一定刚吃完好吃的，或者正准备出去玩~',
        '好开心的' + animalName + '！' + pronoun + '的尾巴一定摇得像小风扇一样！',
      ];
    } else {
      pool = [
        '这只' + animalName + '看起来心情不错，开心指数偏高，整体状态比较放松愉快。',
        animalName + '今天心情挺好的，表情放松，状态良好。',
      ];
    }
  } else if (topDim === '生气指数') {
    if (topVal >= 60) {
      pool = [
        '嗯...这只' + animalName + '好像有点小脾气！' + pronoun + '的生气指数偏高，可能是没吃到零食或者被吵醒了~',
        '注意！' + animalName + '正在生闷气模式中，建议用零食安抚一下' + pronoun + '的情绪！',
      ];
    } else {
      pool = [
        '这只' + animalName + '看起来有一点点不高兴，但整体还好，可能只是没睡醒~',
        animalName + '的表情有点严肃，不过没有明显的不开心，再观察观察吧。',
      ];
    }
  } else if (topDim === '警觉指数') {
    pool = [
      '这只' + animalName + '处于比较警觉的状态，' + pronoun + '可能注意到了什么有趣的东西，正在认真观察周围环境。',
      animalName + '的警觉指数较高，' + pronoun + '可能在守护家园，或者听到了奇怪的声音~',
      '嘘...' + animalName + '正在侦察模式中！' + pronoun + '的耳朵竖得直直的，非常专注！',
    ];
  } else if (topDim === '疑惑指数') {
    pool = [
      '这只' + animalName + '看起来有点困惑，' + pronoun + '的疑惑指数偏高，可能是看到了什么让' + pronoun + '不理解的东西~',
      animalName + '的表情有点复杂，' + pronoun + '好像在想"这是啥，能吃吗？"',
      '哈哈，' + animalName + '一脸迷茫！' + pronoun + '的疑惑指数告诉我们，' + pronoun + '正在思考喵生/狗生~',
    ];
  } else if (topDim === '活跃指数') {
    if (topVal >= 70) {
      pool = [
        '这只' + animalName + '活力满满！' + pronoun + '的活跃指数很高，一定是一只特别爱运动的' + animalName + '！',
        '能量爆棚的' + animalName + '！看' + pronoun + '的活动量，主人平时一定没少带' + pronoun + '出去玩~',
      ];
    } else {
      pool = [
        '这只' + animalName + '比较安静，活跃指数偏低，可能是累了或者在休息。',
        animalName + '今天看起来比较慵懒，活动量不大，可能是个喜欢葛优躺的' + animalName + '~',
      ];
    }
  } else {
    pool = ['这是一只状态不错的' + animalName + '，各项指数都比较均衡。'];
  }

  // ---- 追加综合评分评价 ----
  let suffix = '';
  if (goodVal >= 80) {
    suffix = ' 综合来看，' + pronoun + '是一只非常棒的' + animalName + '！';
  } else if (goodVal >= 60) {
    suffix = ' 总的来说，' + animalName + '的状态还不错。';
  } else if (goodVal >= 40) {
    suffix = ' ' + animalName + '的状态一般，多陪陪' + pronoun + '吧~';
  } else {
    suffix = ' ' + animalName + '今天状态不太好，可能需要主人的关爱哦~';
  }

  // 随机选一条评论 + 后缀
  const comment = pool[Math.floor(Math.random() * pool.length)] + suffix;

  return comment;
}

// ==================== 静态姿态指标（D同学设计风格 + 兼容新格式） ====================

function renderStaticPose(pose) {
  staticPose.innerHTML = '';

  if (!pose || Object.keys(pose).length === 0) {
    staticPose.innerHTML = '<p class="metrics-empty">暂无静态数据</p>';
    return;
  }

  // 检测是新格式（有 actions 数组）还是旧格式
  const isNewFormat = !!(pose.actions && pose.actions.length > 0);
  const isOldFormat = !!(pose.main_pose || pose.head_status || pose.mental_state);

  if (!isNewFormat && !isOldFormat) {
    staticPose.innerHTML = '<p class="metrics-empty">暂无静态数据</p>';
    return;
  }

  // ---- 新格式优先（有 actions） ----
  if (isNewFormat) {
    // 当前姿态（main_pose_cn）
    if (pose.main_pose_cn) {
      const tag = document.createElement('div');
      tag.className = 'spose-card';
      tag.innerHTML = `
        <span class="spose-label">当前姿态</span>
        <span class="spose-value" style="background:#e07b4c">${pose.main_pose_cn}</span>
      `;
      staticPose.appendChild(tag);
    } else if (pose.main_pose) {
      // 兼容旧字段
      const tag = document.createElement('div');
      tag.className = 'spose-card';
      tag.innerHTML = `
        <span class="spose-label">当前姿态</span>
        <span class="spose-value" style="background:#e07b4c">${pose.main_pose}</span>
      `;
      staticPose.appendChild(tag);
    }

    // 检测行为（从 actions 提取 pose_cn）
    if (pose.actions && pose.actions.length > 0) {
      const tag = document.createElement('div');
      tag.className = 'spose-card';
      const actText = pose.actions.map(a => a.pose_cn || a.name || a).join(' / ');
      tag.innerHTML = `
        <span class="spose-label">检测行为</span>
        <span class="spose-value" style="background:#5b8ec9">${actText}</span>
      `;
      staticPose.appendChild(tag);
    }

    // 置信度
    const conf = pose.confidence;
    if (conf != null && conf > 0) {
      const pct = Math.round(conf * 100);
      const confDiv = document.createElement('div');
      confDiv.className = 'spose-conf';
      confDiv.innerHTML = `
        <div class="spose-conf-head">
          <span class="spose-label">检测置信度</span>
          <span class="spose-conf-val">${pct}%</span>
        </div>
        <div class="spose-conf-track">
          <div class="spose-conf-fill" style="width:${pct}%"></div>
        </div>
      `;
      staticPose.appendChild(confDiv);
    }
    return;
  }

  // ---- 旧格式（兼容 D 同学原始设计） ----
  const items = [
    { label: '当前姿态', value: pose.main_pose, score: pose.main_pose_score, color: '#e07b4c' },
    { label: '头部状态', value: pose.head_status, score: pose.head_status_score, color: '#5b8ec9' },
    { label: '精神状态', value: pose.mental_state, score: pose.mental_state_score, color: '#5fa88a' }
  ];

  const visible = items.filter(i => i.value);

  visible.forEach(item => {
    const tag = document.createElement('div');
    tag.className = 'spose-card';
    tag.innerHTML = `
      <span class="spose-label">${item.label}</span>
      <span class="spose-value" style="background:${item.color}">${item.value}</span>
    `;
    staticPose.appendChild(tag);
  });

  // 置信度
  if (pose.pose_confidence != null && pose.pose_confidence > 0) {
    const pct = Math.round(pose.pose_confidence * 100);
    const conf = document.createElement('div');
    conf.className = 'spose-conf';
    conf.innerHTML = `
      <div class="spose-conf-head">
        <span class="spose-label">检测置信度</span>
        <span class="spose-conf-val">${pct}%</span>
      </div>
      <div class="spose-conf-track">
        <div class="spose-conf-fill" style="width:${pct}%"></div>
      </div>
    `;
    staticPose.appendChild(conf);
  }
}

// ==================== 动态行为指标（D同学原始设计 + 兼容新旧格式） ====================

function renderVideoBehavior(vb) {
  dynamicBehavior.innerHTML = '';

  if (!vb || Object.keys(vb).length === 0) {
    dynamicBehavior.innerHTML = '<p class="metrics-empty">暂无动态数据</p>';
    return;
  }

  const inner = document.createElement('div');
  inner.className = 'dyn-beh';

  // 判断数据格式：rich（后端完整格式）vs simple（旧mock只有5个分数）
  const isRich = !!(vb.main_behavior_cn || vb.motion_state_cn || vb.trend_cn || vb.behavior_timeline);

  if (isRich) {
    // ---- 富格式：标签卡片 + 运动分数 + 时间轴 ----

    // 主要行为
    if (vb.main_behavior_cn) {
      inner.innerHTML += `
        <div class="dyn-card">
          <span class="dyn-label">主要行为</span>
          <span class="dyn-tag" style="background:#5fa88a">${vb.main_behavior_cn}</span>
        </div>`;
    }

    // 运动状态
    if (vb.motion_state_cn) {
      inner.innerHTML += `
        <div class="dyn-card">
          <span class="dyn-label">运动状态</span>
          <span class="dyn-tag" style="background:#5b8ec9">${vb.motion_state_cn}</span>
        </div>`;
    }

    // 活跃趋势
    if (vb.trend_cn) {
      inner.innerHTML += `
        <div class="dyn-card">
          <span class="dyn-label">活跃趋势</span>
          <span class="dyn-tag" style="background:#8e7cc3">${vb.trend_cn}</span>
        </div>`;
    }

    // 运动分数
    if (vb.motion_score != null) {
      const pct = Math.round(Math.min(vb.motion_score / 100, 1) * 100);
      let mc = '#5fa88a';
      if (vb.motion_score >= 75) mc = '#d4686a';
      else if (vb.motion_score >= 45) mc = '#e07b4c';
      else if (vb.motion_score >= 18) mc = '#c9a94e';

      inner.innerHTML += `
        <div class="dyn-motion">
          <div class="dyn-motion-head">
            <span class="dyn-label">运动分数</span>
            <span class="dyn-motion-val" style="color:${mc}">${vb.motion_score}</span>
          </div>
          <div class="dyn-motion-track">
            <div class="dyn-motion-fill" style="width:${pct}%;background:${mc}"></div>
          </div>
        </div>`;
    }

    // 行为时间轴
    const timeline = vb.behavior_timeline || [];
    if (timeline.length > 0) {
      let tlHTML = '<div class="dyn-timeline"><div class="dyn-timeline-title">行为时间轴</div><div class="dyn-tl-list">';
      const dotColors = ['#e07b4c', '#5fa88a', '#5b8ec9', '#8e7cc3', '#c9a94e', '#d4686a'];
      timeline.forEach((item, i) => {
        const dotColor = dotColors[i % dotColors.length];
        const confPct = item.confidence ? Math.round(item.confidence * 100) + '%' : '';
        tlHTML += `
          <div class="dyn-tl-item">
            <span class="dyn-tl-time">${item.start_label || '00:00'} → ${item.end_label || '00:00'}</span>
            <span class="dyn-tl-dot" style="background:${dotColor}"></span>
            <span class="dyn-tl-beh">${item.behavior_cn || '未知'}</span>
            ${confPct ? `<span class="dyn-tl-conf">${confPct}</span>` : ''}
          </div>`;
      });
      tlHTML += '</div></div>';
      inner.innerHTML += tlHTML;
    }

    // 摘要（如果有）
    if (vb.summary) {
      inner.innerHTML += `<div class="dyn-motion"><span class="dyn-label" style="font-size:12px;color:#aaa;">${vb.summary}</span></div>`;
    }
  } else {
    // ---- 简单格式：5 个进度条 ----
    const metrics = [
      { key: 'motion_score', label: '活跃程度', color: '#e07b4c' },
      { key: 'direction_score', label: '方向变化', color: '#5b8ec9' },
      { key: 'shake_score', label: '摇晃程度', color: '#8e7cc3' },
      { key: 'turning_score', label: '转身程度', color: '#c9a94e' },
      { key: 'stretch_score', label: '伸展程度', color: '#d4686a' }
    ];

    const visible = metrics.filter(m => (vb[m.key] || 0) > 0);

    if (visible.length === 0) {
      dynamicBehavior.innerHTML = '<p class="metrics-empty">暂无动态数据</p>';
      return;
    }

    visible.forEach(m => {
      const val = vb[m.key];
      const pct = Math.round(Math.min(val / 100, 1) * 100);
      inner.innerHTML += `
        <div class="dyn-motion">
          <div class="dyn-motion-head">
            <span class="dyn-label">${m.label}</span>
            <span class="dyn-motion-val" style="color:${m.color}">${val}</span>
          </div>
          <div class="dyn-motion-track">
            <div class="dyn-motion-fill" style="width:${pct}%;background:${m.color}"></div>
          </div>
        </div>`;
    });
  }

  dynamicBehavior.appendChild(inner);
}

// ==================== ECharts 雷达图 ====================

function renderRadarChart(scores, animal) {
  const dom = $('#radarChart');
  dom.innerHTML = '';

  if (dom.clientWidth === 0 || dom.clientHeight === 0) {
    setTimeout(() => renderRadarChart(scores, animal), 100);
    return;
  }

  if (radarChartInstance) radarChartInstance.dispose();
  radarChartInstance = echarts.init(dom);

  const indicators = Object.keys(scores).map(k => ({ name: k, max: 100 }));
  const values = Object.values(scores);
  const mc = animal === 'dog' ? '#e07b4c' : '#5fa88a';
  const fc = animal === 'dog' ? 'rgba(224,123,76,0.3)' : 'rgba(95,168,138,0.3)';

  radarChartInstance.setOption({
    tooltip: { trigger: 'item', backgroundColor: 'rgba(255,255,255,0.95)', borderColor: '#e8e8e8', textStyle: { color: '#333', fontSize: 12 } },
    radar: {
      center: ['50%', '52%'], radius: '65%',
      indicator: indicators, shape: 'polygon', splitNumber: 4,
      axisName: { color: '#555', fontSize: 11, fontWeight: 'bold' },
      splitArea: { areaStyle: { color: ['#fafafa', '#f5f5f5', '#fafafa', '#f5f5f5'] } },
      axisLine: { lineStyle: { color: '#e0e0e0' } },
      splitLine: { lineStyle: { color: '#e0e0e0' } }
    },
    series: [{
      type: 'radar',
      data: [{ value: values, name: animal === 'dog' ? '狗狗' : '猫咪',
        areaStyle: { color: fc },
        lineStyle: { color: mc, width: 2 },
        itemStyle: { color: mc, borderColor: '#fff', borderWidth: 2 },
        symbol: 'circle', symbolSize: 5
      }]
    }]
  });

  window.addEventListener('resize', () => { if (radarChartInstance) radarChartInstance.resize(); });
}

// ==================== 详细指数 ====================

function renderScores(scores) {
  scoresGrid.innerHTML = '';
  const colors = ['#e07b4c', '#5fa88a', '#d4686a', '#c9a94e', '#7c6db8', '#6db8a8'];
  Object.entries(scores).forEach(([k, v], i) => {
    const row = document.createElement('div');
    row.className = 'score-row';
    row.innerHTML = `
      <div class="score-row-head">
        <span class="score-row-label">${k}</span>
        <span class="score-row-num">${v}</span>
      </div>
      <div class="score-row-track">
        <div class="score-row-fill" style="width:${v}%;background:${colors[i % colors.length]}"></div>
      </div>`;
    scoresGrid.appendChild(row);
  });
}

// ==================== 错误 ====================

function showError(msg) {
  showRightState('error');
  errorText.textContent = msg;
}

retryBtn.addEventListener('click', () => {
  showRightState('empty');
  resetUpload();
});

// ==================== 重置 ====================

resetBtn.addEventListener('click', () => {
  showRightState('empty');
  resetUpload();
  if (radarChartInstance) { radarChartInstance.dispose(); radarChartInstance = null; }
});

// ==================== 拖拽 ====================

uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('drag-over'); });
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
    return alert('请上传图片或视频文件');
  }
  handleFile(file);
});

console.log('Pet Status Analyzer — Ready | API:', ANALYZE_URL, '| Mock:', USE_MOCK ? 'ON' : 'OFF');
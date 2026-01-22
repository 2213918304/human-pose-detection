# 人体姿态识别系统

基于 YOLOv8 + MediaPipe 的实时人体姿态检测和分类系统

## 项目简介

本系统采用 YOLOv8 进行人体检测，MediaPipe 进行关键点估计，并通过规则分类器识别5种常见姿态：
- 站立 (Standing)
- 坐立 (Sitting)
- 躺下 (Lying Down)
- 蹲下 (Squatting)
- 弯腰 (Bending)

## 主要特性

- ✅ **高精度检测**：YOLOv8 人体检测 + MediaPipe 33个关键点估计
- ✅ **实时处理**：支持 GPU 加速，实时视频处理
- ✅ **多种姿态**：识别5种常见人体姿态
- ✅ **可视化**：实时显示检测框、关键点、骨架和姿态标签
- ✅ **数据导出**：支持检测结果导出为 JSON 格式
- ✅ **中文支持**：界面标签完整支持中文显示
- ✅ **灵活配置**：支持命令行参数和配置文件

## 系统架构

```
输入视频 → YOLOv8检测 → MediaPipe姿态估计 → 姿态分类 → 可视化输出
```

## 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于 GPU 加速)
- Windows / Linux / macOS

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd 姿态识别
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `ultralytics` - YOLOv8
- `mediapipe` - 姿态估计
- `opencv-python` - 图像处理
- `numpy` - 数值计算
- `protobuf` - 协议缓冲区（3.20.x版本，兼容MediaPipe）
- `Pillow` - 中文字体支持

**注意**：如果遇到 `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` 错误，需要降级protobuf：

```bash
pip install protobuf==3.20.3
```

### 3. 下载 YOLOv8 模型

首次运行时会自动下载 `yolov8n.pt` 模型（约 6MB）

## 快速开始

### 基本使用

```bash
# 处理视频文件（注意：使用 --output-video 而不是 --output）
python main.py --source video.mp4 --device cuda --save-video --output-video output.mp4

# 使用摄像头
python main.py --source 0 --device cuda

# 不显示实时画面（加快处理）
python main.py --source video.mp4 --device cuda --no-display --save-video --output-video output.mp4
```

### 命令行参数

```bash
python main.py [选项]

必选参数:
  --source SOURCE              视频源（文件路径或摄像头ID，默认0）

可选参数:
  --config CONFIG              配置文件路径
  --device {cpu,cuda}          运行设备（默认cpu）
  --detection-confidence CONF  YOLO检测置信度阈值（默认0.32）
  --tracking-confidence CONF   MediaPipe跟踪置信度阈值（默认0.5）
  --classification-confidence  姿态分类置信度阈值（默认0.6）
  
输出选项:
  --save-video                 保存输出视频
  --output-video PATH          输出视频路径（默认output.mp4）
  --export-data                导出检测数据
  --output-data PATH           数据导出路径（默认detections.json）
  
显示选项:
  --no-landmarks               不显示关键点
  --no-skeleton                不显示骨架
  --no-bbox                    不显示边界框
  --no-label                   不显示标签
  --no-display                 不显示实时画面
  --fps FPS                    目标帧率（默认30）
```

## 使用示例

### 1. 处理视频文件（GPU加速）

```bash
python main.py --source video.mp4 --device cuda --save-video --output-video result.mp4
```

### 2. 实时摄像头检测

```bash
python main.py --source 0 --device cuda
```

### 3. 调整检测阈值

```bash
# 降低阈值以提高躺下姿态检测率
python main.py --source video.mp4 --device cuda --detection-confidence 0.3 --save-video --output-video output.mp4
```

### 4. 导出检测数据

```bash
python main.py --source video.mp4 --device cuda --export-data --output-data detections.json
```

### 5. 批量处理（不显示画面）

```bash
python main.py --source video.mp4 --device cuda --no-display --save-video --output-video output.mp4
```

## 配置文件

可以使用 JSON 配置文件来设置默认参数：

```json
{
  "video_source": "video.mp4",
  "min_detection_confidence": 0.32,
  "min_tracking_confidence": 0.5,
  "classification_confidence_threshold": 0.6,
  "device": "cuda",
  "save_video": true,
  "output_video_path": "output.mp4",
  "show_landmarks": true,
  "show_skeleton": true,
  "show_bbox": true,
  "show_label": true,
  "show_display": true
}
```

使用配置文件：

```bash
python main.py --config config.json
```

## 项目结构

```
姿态识别/
├── src/                          # 源代码
│   ├── models/                   # 数据模型
│   │   └── data_models.py
│   ├── detection/                # 人体检测
│   │   └── yolo_detector.py
│   ├── pose/                     # 姿态估计和分类
│   │   ├── pose_estimator.py
│   │   └── pose_classifier.py
│   ├── visualization/            # 可视化
│   │   └── visualizer.py
│   ├── config/                   # 配置管理
│   │   └── config_manager.py
│   ├── export/                   # 数据导出
│   │   └── data_exporter.py
│   ├── utils/                    # 工具函数
│   │   ├── video_capture.py
│   │   └── logger.py
│   └── pose_detection_system.py  # 主系统
├── tests/                        # 测试代码
├── .kiro/                        # 规范文档
│   └── specs/
│       └── human-pose-detection/
├── main.py                       # 命令行入口
├── requirements.txt              # 依赖列表
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```

## 性能指标

### 检测精度（YOLO阈值0.32）

| 姿态 | 检测率 |
|------|--------|
| 站立 | 95%+ |
| 坐立 | 90%+ |
| 躺下 | 85%+ |
| 蹲下 | 85%+ |
| 弯腰 | 90%+ |

### 处理速度

- **GPU (CUDA)**: 18-25 FPS
- **CPU**: 3-5 FPS

## 技术细节

### YOLOv8 检测器

- 模型：yolov8n.pt (nano版本)
- 置信度阈值：0.32（可调）
- 只检测人体类别（COCO class 0）

### MediaPipe 姿态估计

- 模型复杂度：1 (Full模型)
- 关键点数量：33个
- 检测置信度：0.5
- 跟踪置信度：0.5

### 姿态分类器

基于几何特征的规则分类：
- 躯干角度
- 膝盖角度
- 髋部高度
- 身体宽高比

## 常见问题

### 1. protobuf版本错误？

如果遇到 `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` 错误：

```bash
pip install protobuf==3.20.3
```

MediaPipe需要protobuf 3.x版本，不兼容4.x版本。

### 2. 躺下姿态检测不到？

降低YOLO检测阈值：

```bash
python main.py --source video.mp4 --device cuda --detection-confidence 0.3
```

### 2. 有误检和闪动？

提高YOLO检测阈值：

```bash
python main.py --source video.mp4 --device cuda --detection-confidence 0.4
```

### 3. 中文标签显示为方框？

确保系统安装了中文字体（微软雅黑或黑体）

### 4. GPU加速不工作？

检查CUDA安装：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. 处理速度太慢？

- 使用GPU加速（`--device cuda`）
- 禁用实时显示（`--no-display`）
- 降低目标帧率（`--fps 15`）

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_yolo_detector.py

# 查看覆盖率
pytest --cov=src --cov-report=html
```

## 开发计划

- [ ] 支持更多姿态类别
- [ ] 添加姿态序列分析
- [ ] 支持多人跟踪
- [ ] 优化检测速度
- [ ] 添加Web界面

## 许可证

MIT License

## 作者

**罗泽**
- Email: ratku@qq.com

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe)
- [OpenCV](https://opencv.org/)

## 更新日志

### v1.0.0 (2026-01-22)

- ✅ 初始版本发布
- ✅ 支持5种姿态识别
- ✅ GPU加速支持
- ✅ 中文界面支持
- ✅ 数据导出功能
- ✅ 完整测试覆盖

---

如有问题或建议，请联系：ratku@qq.com

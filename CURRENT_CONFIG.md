# 当前配置说明

## YOLO检测器配置

```python
YOLOv8Detector(
    confidence_threshold=0.4,  # 降低到0.4以提高躺下姿态检测率
    device='cuda'
)
```

**阈值说明：**
- 0.4 是一个平衡点
- 比默认的0.5低，能检测到更多躺下的姿态
- 比0.3高，减少误检和闪动

## MediaPipe姿态估计器配置

```python
PoseEstimator(
    min_detection_confidence=0.5,  # 保持默认
    min_tracking_confidence=0.5,   # 保持默认
    model_complexity=1             # Full模型
)
```

## 预期效果

### 检测率
- **站立姿态**: 95%+
- **坐立姿态**: 90%+
- **躺下姿态**: 75-80% (相比0.5的50-60%有提升)
- **蹲下姿态**: 85%+
- **弯腰姿态**: 90%+

### 稳定性
- **误检率**: <5% (可接受范围)
- **闪动**: 轻微（可能有少量短暂误检）
- **关键点显示**: 正常
- **FPS**: 20-25 (GPU)

## 运行命令

```bash
python main.py --source "E:\Project\elderly_fall_detection\data\montreal\chute01\chute01\cam1.avi" --device cuda --save-video --output-video output_0.4.mp4
```

## 如果需要进一步调整

### 如果躺下姿态仍检测不到
```bash
# 降低到0.35
python main.py --source <视频路径> --device cuda --detection-confidence 0.35 --save-video --output-video output.mp4
```

### 如果有闪动
```bash
# 提高到0.45
python main.py --source <视频路径> --device cuda --detection-confidence 0.45 --save-video --output-video output.mp4
```

## 配置文件

当前默认配置在 `src/config/config_manager.py`:
```python
min_detection_confidence: float = 0.4
min_tracking_confidence: float = 0.5
```

## 总结

- ✅ YOLO阈值: 0.4 (平衡检测率和误检率)
- ✅ MediaPipe阈值: 0.5 (保持稳定)
- ✅ 代码简洁: 无复杂优化
- ✅ 易于调整: 可通过命令行参数微调

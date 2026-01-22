# 快速修复指南

## 问题1: protobuf版本错误

### 错误信息
```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

### 原因
MediaPipe 不兼容 protobuf 4.x 版本，需要使用 3.20.x 版本。

### 解决方案
```bash
pip install protobuf==3.20.3
```

或者重新安装所有依赖：
```bash
pip install -r requirements.txt
```

---

## 问题2: 命令行参数错误

### 错误信息
```
error: ambiguous option: --output could match --output-video, --output-data
```

### 原因
`--output` 参数有歧义，系统不知道是指视频输出还是数据输出。

### 解决方案
使用完整的参数名：

**正确命令**：
```bash
python main.py --source "E:\Project\elderly_fall_detection\data\montreal\chute01\chute01\cam1.avi" --device cuda --save-video --output-video output_optimized_v2.mp4
```

**错误命令**（不要使用）：
```bash
python main.py --source "..." --device cuda --save-video --output output_optimized_v2.mp4
```

---

## 完整的运行步骤

### 1. 修复protobuf版本
```bash
pip install protobuf==3.20.3
```

### 2. 运行系统（正确命令）
```bash
python main.py --source "E:\Project\elderly_fall_detection\data\montreal\chute01\chute01\cam1.avi" --device cuda --save-video --output-video output_optimized_v2.mp4
```

### 3. 可选：不显示实时画面（加快处理）
```bash
python main.py --source "E:\Project\elderly_fall_detection\data\montreal\chute01\chute01\cam1.avi" --device cuda --save-video --output-video output_optimized_v2.mp4 --no-display
```

---

## 其他有用的命令

### 调整YOLO阈值
```bash
# 当前阈值是0.32，可以根据需要调整
python main.py --source "E:\Project\elderly_fall_detection\data\montreal\chute01\chute01\cam1.avi" --device cuda --detection-confidence 0.3 --save-video --output-video output.mp4
```

### 导出检测数据
```bash
python main.py --source "E:\Project\elderly_fall_detection\data\montreal\chute01\chute01\cam1.avi" --device cuda --export-data --output-data detections.json
```

### 查看所有参数
```bash
python main.py --help
```

---

## 验证修复

运行以下命令验证protobuf版本：
```bash
python -c "import mediapipe; print('MediaPipe版本:', mediapipe.__version__)"
python -c "import google.protobuf; print('Protobuf版本:', google.protobuf.__version__)"
```

应该看到：
- MediaPipe版本: 0.10.x
- Protobuf版本: 3.20.x

---

## 联系方式

如有问题，请联系：
- 作者：罗泽
- Email: 2213918304@qq.com

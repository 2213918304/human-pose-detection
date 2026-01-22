"""
快速测试优化参数
"""

from src.detection.yolo_detector import YOLOv8Detector
from src.pose.pose_estimator import PoseEstimator

print("=" * 60)
print("当前优化参数")
print("=" * 60)

# YOLO参数
detector = YOLOv8Detector(device='cuda')
info = detector.get_model_info()
print("\nYOLO检测器:")
print(f"  置信度阈值: {info['confidence_threshold']}")
print(f"  IOU阈值: {info['iou_threshold']}")
print(f"  图像尺寸: {info['img_size']}")
print(f"  TTA增强: 否（已禁用）")
print(f"  最大检测数: 10")
print(f"  类别无关NMS: 否（已禁用）")
print(f"  最小框尺寸: 30x30像素")
print(f"  宽高比范围: 0.2~2.0")
detector.close()

# MediaPipe参数
estimator = PoseEstimator()
info = estimator.get_info()
print("\nMediaPipe姿态估计器:")
print(f"  检测置信度: {info['min_detection_confidence']}")
print(f"  跟踪置信度: {info['min_tracking_confidence']}")
print(f"  模型复杂度: {info['model_complexity']} (Full)")
print(f"  时间平滑: 是（3帧）")
print(f"  可见性阈值: 0.4")
estimator.close()

print("\n" + "=" * 60)
print("优化说明")
print("=" * 60)
print("\n✅ 提高YOLO阈值到0.35 - 减少误检")
print("✅ 降低IOU阈值到0.6 - 减少重叠框")
print("✅ 禁用TTA - 减少误检和提高速度")
print("✅ 禁用类别无关NMS - 提高精度")
print("✅ 添加尺寸过滤 - 过滤小目标误检")
print("✅ 添加宽高比过滤 - 过滤异常形状")
print("✅ 保持大图像尺寸 - 检测躺下姿态")
print("✅ 保持时间平滑 - 减少关键点抖动")

print("\n预期效果:")
print("  - 误检率大幅降低（闪动问题解决）")
print("  - 躺下姿态仍能检测（0.35阈值足够）")
print("  - 关键点稳定（时间平滑）")
print("  - FPS提升（禁用TTA）")

print("\n" + "=" * 60)
print("现在可以重新运行测试！")
print("=" * 60)

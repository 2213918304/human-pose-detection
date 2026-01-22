#!/usr/bin/env python3
"""
人体姿态识别系统演示脚本

展示系统的核心功能，无需实际视频输入。
"""

import numpy as np
import cv2
from src.models.data_models import (
    Landmark, BoundingBox, PersonDetection, 
    PoseEstimation, PoseClassification, PoseState
)
from src.visualization.visualizer import Visualizer
from src.export.data_exporter import DataExporter
from src.config.config_manager import ConfigManager

def create_demo_landmarks(pose_type="standing"):
    """创建演示用的关键点数据"""
    landmarks = []
    
    if pose_type == "standing":
        # 站立姿态：头部在上，脚在下
        positions = [
            # 头部和面部 (0-10)
            (0.5, 0.1), (0.48, 0.12), (0.52, 0.12), (0.46, 0.14), (0.54, 0.14),
            (0.44, 0.16), (0.56, 0.16), (0.42, 0.18), (0.58, 0.18), (0.48, 0.15), (0.52, 0.15),
            # 上身 (11-16)
            (0.45, 0.25), (0.55, 0.25), (0.43, 0.35), (0.57, 0.35), (0.42, 0.45), (0.58, 0.45),
            # 手部 (17-22)
            (0.41, 0.48), (0.59, 0.48), (0.40, 0.50), (0.60, 0.50), (0.39, 0.52), (0.61, 0.52),
            # 下身 (23-28)
            (0.47, 0.55), (0.53, 0.55), (0.46, 0.70), (0.54, 0.70), (0.45, 0.85), (0.55, 0.85),
            # 脚部 (29-32)
            (0.44, 0.90), (0.56, 0.90), (0.43, 0.92), (0.57, 0.92)
        ]
    elif pose_type == "sitting":
        # 坐姿：上身直立，腿部弯曲
        positions = [
            # 头部和面部 (0-10)
            (0.5, 0.15), (0.48, 0.17), (0.52, 0.17), (0.46, 0.19), (0.54, 0.19),
            (0.44, 0.21), (0.56, 0.21), (0.42, 0.23), (0.58, 0.23), (0.48, 0.20), (0.52, 0.20),
            # 上身 (11-16)
            (0.45, 0.30), (0.55, 0.30), (0.43, 0.40), (0.57, 0.40), (0.42, 0.50), (0.58, 0.50),
            # 手部 (17-22)
            (0.41, 0.53), (0.59, 0.53), (0.40, 0.55), (0.60, 0.55), (0.39, 0.57), (0.61, 0.57),
            # 下身 (23-28) - 腿部弯曲
            (0.47, 0.60), (0.53, 0.60), (0.40, 0.70), (0.60, 0.70), (0.35, 0.75), (0.65, 0.75),
            # 脚部 (29-32)
            (0.33, 0.78), (0.67, 0.78), (0.32, 0.80), (0.68, 0.80)
        ]
    else:  # lying
        # 躺姿：身体水平
        positions = [
            # 头部和面部 (0-10)
            (0.2, 0.5), (0.22, 0.48), (0.22, 0.52), (0.24, 0.46), (0.24, 0.54),
            (0.26, 0.44), (0.26, 0.56), (0.28, 0.42), (0.28, 0.58), (0.23, 0.48), (0.23, 0.52),
            # 上身 (11-16)
            (0.35, 0.45), (0.35, 0.55), (0.45, 0.43), (0.45, 0.57), (0.55, 0.42), (0.55, 0.58),
            # 手部 (17-22)
            (0.58, 0.41), (0.58, 0.59), (0.60, 0.40), (0.60, 0.60), (0.62, 0.39), (0.62, 0.61),
            # 下身 (23-28)
            (0.65, 0.48), (0.65, 0.52), (0.75, 0.47), (0.75, 0.53), (0.85, 0.46), (0.85, 0.54),
            # 脚部 (29-32)
            (0.90, 0.45), (0.90, 0.55), (0.92, 0.44), (0.92, 0.56)
        ]
    
    for x, y in positions:
        landmarks.append(Landmark(x=x, y=y, visibility=0.9))
    
    return landmarks

def create_demo_frame(width=640, height=480):
    """创建演示用的图像帧"""
    # 创建白色背景
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 添加标题
    cv2.putText(frame, "Human Pose Detection System Demo", 
                (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return frame

def main():
    """主演示函数"""
    print("=" * 60)
    print("人体姿态识别系统 - 功能演示")
    print("=" * 60)
    
    # 1. 配置管理演示
    print("\n1. 配置管理")
    print("-" * 60)
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(f"✓ 默认配置加载成功")
    print(f"  - 检测置信度: {config.detection_confidence}")
    print(f"  - 跟踪置信度: {config.tracking_confidence}")
    print(f"  - 分类阈值: {config.classification_confidence}")
    print(f"  - 目标帧率: {config.target_fps}")
    
    # 2. 数据模型演示
    print("\n2. 数据模型")
    print("-" * 60)
    
    # 创建三种不同姿态的演示数据
    poses = {
        "standing": ("站立", PoseState.STANDING),
        "sitting": ("坐立", PoseState.SITTING),
        "lying": ("躺下", PoseState.LYING)
    }
    
    detections = []
    for i, (pose_key, (pose_name, pose_state)) in enumerate(poses.items()):
        # 创建边界框
        bbox = BoundingBox(
            x=50 + i * 200,
            y=100,
            width=150,
            height=300
        )
        
        # 创建人体检测
        person = PersonDetection(
            bbox=bbox,
            confidence=0.95,
            person_id=i + 1
        )
        
        # 创建姿态估计
        landmarks = create_demo_landmarks(pose_key)
        pose_estimation = PoseEstimation(landmarks=landmarks)
        
        # 创建姿态分类
        classification = PoseClassification(
            state=pose_state,
            confidence=0.88
        )
        
        detections.append({
            'person': person,
            'pose': pose_estimation,
            'classification': classification,
            'name': pose_name
        })
        
        print(f"✓ 创建 {pose_name} 姿态数据")
        print(f"  - 边界框: ({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height})")
        print(f"  - 关键点数量: {len(landmarks)}")
        print(f"  - 姿态状态: {pose_state.value}")
    
    # 3. 可视化演示
    print("\n3. 可视化系统")
    print("-" * 60)
    
    visualizer = Visualizer()
    frame = create_demo_frame()
    
    # 绘制所有检测结果
    from src.models.data_models import DetectionResult
    detection_results = []
    for det in detections:
        result = DetectionResult(
            person=det['person'],
            pose=det['pose'],
            classification=det['classification']
        )
        detection_results.append(result)
    
    output_frame = visualizer.draw(frame.copy(), detection_results)
    print(f"✓ 可视化 {len(detections)} 个检测结果")
    print(f"  - 显示边界框: {visualizer.config.show_bbox}")
    print(f"  - 显示关键点: {visualizer.config.show_landmarks}")
    print(f"  - 显示骨架: {visualizer.config.show_skeleton}")
    print(f"  - 显示标签: {visualizer.config.show_label}")
    
    # 保存演示图像
    output_path = "demo_output.jpg"
    cv2.imwrite(output_path, output_frame)
    print(f"✓ 演示图像已保存: {output_path}")
    
    # 4. 数据导出演示
    print("\n4. 数据导出")
    print("-" * 60)
    
    exporter = DataExporter()
    exporter.set_metadata({
        "source": "demo",
        "resolution": "640x480",
        "fps": 30
    })
    
    # 添加帧数据
    for frame_idx in range(3):
        exporter.add_frame_data(frame_idx, detection_results)
    
    # 导出数据
    export_path = "demo_detections.json"
    exporter.export(export_path)
    print(f"✓ 检测数据已导出: {export_path}")
    print(f"  - 总帧数: {exporter.get_frame_count()}")
    print(f"  - 每帧检测数: {len(detection_results)}")
    
    # 5. 系统统计
    print("\n5. 系统统计")
    print("-" * 60)
    print(f"✓ 总检测人数: {len(detections)}")
    print(f"✓ 姿态类型:")
    for det in detections:
        print(f"  - {det['name']}: 置信度 {det['classification'].confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {output_path} (可视化结果)")
    print(f"  - {export_path} (检测数据)")
    print(f"\n要运行实时检测，请使用:")
    print(f"  python main.py --source 0  (使用摄像头)")
    print(f"  python main.py --source video.mp4  (使用视频文件)")

if __name__ == "__main__":
    main()

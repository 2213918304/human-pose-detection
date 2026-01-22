"""
Visualizer 模块测试

测试可视化功能，包括边界框、关键点、骨架和标签的绘制。
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st, settings

from src.visualization import Visualizer, VisualizationConfig
from src.models import (
    DetectionResult, BoundingBox, Landmark, PoseClassification, PoseState
)


# ============================================================================
# 辅助函数
# ============================================================================

def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """创建测试帧（黑色背景）"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def create_landmark(x: float, y: float, z: float = 0.0, visibility: float = 1.0) -> Landmark:
    """创建测试关键点"""
    return Landmark(x=x, y=y, z=z, visibility=visibility)


def create_test_landmarks() -> list:
    """创建33个测试关键点（站立姿态）"""
    landmarks = []
    
    # 面部（0-10）
    for i in range(11):
        landmarks.append(create_landmark(0.5, 0.1 + i * 0.01))
    
    # 上半身（11-22）
    landmarks.extend([
        create_landmark(0.45, 0.3),  # 11: LEFT_SHOULDER
        create_landmark(0.55, 0.3),  # 12: RIGHT_SHOULDER
        create_landmark(0.40, 0.5),  # 13: LEFT_ELBOW
        create_landmark(0.60, 0.5),  # 14: RIGHT_ELBOW
        create_landmark(0.38, 0.7),  # 15: LEFT_WRIST
        create_landmark(0.62, 0.7),  # 16: RIGHT_WRIST
    ])
    
    # 手指（17-22）
    for i in range(6):
        landmarks.append(create_landmark(0.5, 0.72 + i * 0.01))
    
    # 下半身（23-32）
    landmarks.extend([
        create_landmark(0.45, 0.6),  # 23: LEFT_HIP
        create_landmark(0.55, 0.6),  # 24: RIGHT_HIP
        create_landmark(0.44, 0.8),  # 25: LEFT_KNEE
        create_landmark(0.56, 0.8),  # 26: RIGHT_KNEE
        create_landmark(0.43, 0.95),  # 27: LEFT_ANKLE
        create_landmark(0.57, 0.95),  # 28: RIGHT_ANKLE
        create_landmark(0.42, 0.97),  # 29: LEFT_HEEL
        create_landmark(0.43, 0.98),  # 30: LEFT_FOOT_INDEX
        create_landmark(0.58, 0.97),  # 31: RIGHT_HEEL
        create_landmark(0.57, 0.98),  # 32: RIGHT_FOOT_INDEX
    ])
    
    return landmarks


def create_test_detection(
    person_id: int = 1,
    bbox: BoundingBox = None,
    with_landmarks: bool = True,
    with_classification: bool = True
) -> DetectionResult:
    """创建测试检测结果"""
    if bbox is None:
        bbox = BoundingBox(x=100, y=50, width=200, height=400)
    
    landmarks = create_test_landmarks() if with_landmarks else None
    
    classification = None
    if with_classification:
        classification = PoseClassification(
            pose_state=PoseState.STANDING,
            confidence=0.95,
            features={"torso_angle": 5.0, "knee_angle": 170.0}
        )
    
    return DetectionResult(
        person_id=person_id,
        bounding_box=bbox,
        confidence=0.9,
        landmarks=landmarks,
        pose_classification=classification,
        timestamp=0.0
    )


# ============================================================================
# 单元测试
# ============================================================================

class TestVisualizerInitialization:
    """测试可视化器初始化"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        viz = Visualizer()
        
        assert viz.show_landmarks is True
        assert viz.show_skeleton is True
        assert viz.show_bbox is True
        assert viz.show_label is True
        assert isinstance(viz.config, VisualizationConfig)
    
    def test_custom_initialization(self):
        """测试自定义初始化"""
        config = VisualizationConfig(
            bbox_color=(255, 0, 0),
            bbox_thickness=3
        )
        
        viz = Visualizer(
            show_landmarks=False,
            show_skeleton=False,
            show_bbox=True,
            show_label=False,
            config=config
        )
        
        assert viz.show_landmarks is False
        assert viz.show_skeleton is False
        assert viz.show_bbox is True
        assert viz.show_label is False
        assert viz.config.bbox_color == (255, 0, 0)
        assert viz.config.bbox_thickness == 3


class TestVisualizerDrawing:
    """测试绘制功能"""
    
    def test_draw_empty_detections(self):
        """测试空检测列表"""
        viz = Visualizer()
        frame = create_test_frame()
        
        result = viz.draw(frame, [])
        
        assert result.shape == frame.shape
        assert np.array_equal(result, frame)  # 应该没有变化
    
    def test_draw_single_detection(self):
        """测试单个检测结果"""
        viz = Visualizer()
        frame = create_test_frame()
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)  # 应该有变化
    
    def test_draw_multiple_detections(self):
        """测试多个检测结果"""
        viz = Visualizer()
        frame = create_test_frame()
        
        detections = [
            create_test_detection(person_id=1, bbox=BoundingBox(50, 50, 150, 350)),
            create_test_detection(person_id=2, bbox=BoundingBox(300, 50, 150, 350))
        ]
        
        result = viz.draw(frame, detections)
        
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)
    
    def test_draw_with_none_frame(self):
        """测试空帧"""
        viz = Visualizer()
        
        result = viz.draw(None, [create_test_detection()])
        
        assert result is None
    
    def test_draw_with_empty_frame(self):
        """测试空数组帧"""
        viz = Visualizer()
        empty_frame = np.array([])
        
        result = viz.draw(empty_frame, [create_test_detection()])
        
        assert result.size == 0


class TestBoundingBoxDrawing:
    """测试边界框绘制"""
    
    def test_draw_bbox_enabled(self):
        """测试启用边界框绘制"""
        viz = Visualizer(show_bbox=True, show_landmarks=False, 
                        show_skeleton=False, show_label=False)
        frame = create_test_frame()
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        # 检查边界框区域是否有变化
        bbox = detection.bounding_box
        roi = result[bbox.y:bbox.y+5, bbox.x:bbox.x+5]
        assert not np.all(roi == 0)  # 应该有绿色边框
    
    def test_draw_bbox_disabled(self):
        """测试禁用边界框绘制"""
        viz = Visualizer(show_bbox=False, show_landmarks=False,
                        show_skeleton=False, show_label=False)
        frame = create_test_frame()
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        # 帧应该没有变化
        assert np.array_equal(result, frame)
    
    def test_bbox_with_person_id(self):
        """测试边界框包含人员ID"""
        viz = Visualizer(show_bbox=True, show_landmarks=False,
                        show_skeleton=False, show_label=False)
        frame = create_test_frame()
        detection = create_test_detection(person_id=42)
        
        result = viz.draw(frame, [detection])
        
        # 检查ID标签区域
        bbox = detection.bounding_box
        id_roi = result[max(0, bbox.y-20):bbox.y, bbox.x:bbox.x+50]
        assert not np.all(id_roi == 0)  # 应该有ID标签


class TestLandmarksDrawing:
    """测试关键点绘制"""
    
    def test_draw_landmarks_enabled(self):
        """测试启用关键点绘制"""
        viz = Visualizer(show_landmarks=True, show_bbox=False,
                        show_skeleton=False, show_label=False)
        frame = create_test_frame(640, 480)
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        # 应该有红色关键点
        assert not np.array_equal(result, frame)
        # 检查是否有红色像素（关键点颜色）
        red_pixels = np.sum(result[:, :, 2] > 0)
        assert red_pixels > 0
    
    def test_draw_landmarks_disabled(self):
        """测试禁用关键点绘制"""
        viz = Visualizer(show_landmarks=False, show_bbox=False,
                        show_skeleton=False, show_label=False)
        frame = create_test_frame()
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        assert np.array_equal(result, frame)
    
    def test_draw_landmarks_with_low_visibility(self):
        """测试低可见性关键点"""
        viz = Visualizer(show_landmarks=True, show_bbox=False,
                        show_skeleton=False, show_label=False)
        frame = create_test_frame(640, 480)
        
        # 创建低可见性关键点
        landmarks = [create_landmark(0.5, 0.5, visibility=0.3) for _ in range(33)]
        detection = DetectionResult(
            person_id=1,
            bounding_box=BoundingBox(100, 50, 200, 400),
            confidence=0.9,
            landmarks=landmarks
        )
        
        result = viz.draw(frame, [detection])
        
        # 低可见性关键点不应该被绘制
        assert np.array_equal(result, frame)


class TestSkeletonDrawing:
    """测试骨架绘制"""
    
    def test_draw_skeleton_enabled(self):
        """测试启用骨架绘制"""
        viz = Visualizer(show_skeleton=True, show_landmarks=False,
                        show_bbox=False, show_label=False)
        frame = create_test_frame(640, 480)
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        # 应该有蓝色骨架线
        assert not np.array_equal(result, frame)
        # 检查是否有蓝色像素（骨架颜色）
        blue_pixels = np.sum(result[:, :, 0] > 0)
        assert blue_pixels > 0
    
    def test_draw_skeleton_disabled(self):
        """测试禁用骨架绘制"""
        viz = Visualizer(show_skeleton=False, show_landmarks=False,
                        show_bbox=False, show_label=False)
        frame = create_test_frame()
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        assert np.array_equal(result, frame)
    
    def test_skeleton_connections_count(self):
        """测试骨架连接数量"""
        # MediaPipe 33点模型应该有特定数量的连接
        assert len(Visualizer.SKELETON_CONNECTIONS) > 0
        
        # 所有连接索引应该在有效范围内
        for start, end in Visualizer.SKELETON_CONNECTIONS:
            assert 0 <= start < 33
            assert 0 <= end < 33


class TestLabelDrawing:
    """测试标签绘制"""
    
    def test_draw_label_enabled(self):
        """测试启用标签绘制"""
        viz = Visualizer(show_label=True, show_landmarks=False,
                        show_skeleton=False, show_bbox=False)
        frame = create_test_frame(640, 480)
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        # 应该有标签文字
        assert not np.array_equal(result, frame)
    
    def test_draw_label_disabled(self):
        """测试禁用标签绘制"""
        viz = Visualizer(show_label=False, show_landmarks=False,
                        show_skeleton=False, show_bbox=False)
        frame = create_test_frame()
        detection = create_test_detection()
        
        result = viz.draw(frame, [detection])
        
        assert np.array_equal(result, frame)
    
    def test_draw_label_without_classification(self):
        """测试无分类结果时的标签"""
        viz = Visualizer(show_label=True, show_landmarks=False,
                        show_skeleton=False, show_bbox=False)
        frame = create_test_frame()
        detection = create_test_detection(with_classification=False)
        
        result = viz.draw(frame, [detection])
        
        # 无分类结果时不应该绘制标签
        assert np.array_equal(result, frame)
    
    def test_label_position_adjustment(self):
        """测试标签位置调整（避免超出帧范围）"""
        viz = Visualizer(show_label=True, show_landmarks=False,
                        show_skeleton=False, show_bbox=False)
        frame = create_test_frame(640, 480)
        
        # 创建靠近底部的检测
        bbox = BoundingBox(x=100, y=400, width=200, height=70)
        detection = create_test_detection(bbox=bbox)
        
        result = viz.draw(frame, [detection])
        
        # 应该成功绘制（标签应该调整到上方）
        assert not np.array_equal(result, frame)


class TestVisualizerConfiguration:
    """测试配置管理"""
    
    def test_get_config(self):
        """测试获取配置"""
        config = VisualizationConfig(bbox_color=(255, 0, 0))
        viz = Visualizer(config=config)
        
        retrieved_config = viz.get_config()
        
        assert retrieved_config.bbox_color == (255, 0, 0)
    
    def test_set_config(self):
        """测试设置配置"""
        viz = Visualizer()
        new_config = VisualizationConfig(
            bbox_color=(0, 0, 255),
            landmark_radius=5
        )
        
        viz.set_config(new_config)
        
        assert viz.config.bbox_color == (0, 0, 255)
        assert viz.config.landmark_radius == 5
    
    def test_set_show_landmarks(self):
        """测试设置关键点显示"""
        viz = Visualizer(show_landmarks=True)
        
        viz.set_show_landmarks(False)
        assert viz.show_landmarks is False
        
        viz.set_show_landmarks(True)
        assert viz.show_landmarks is True
    
    def test_set_show_skeleton(self):
        """测试设置骨架显示"""
        viz = Visualizer(show_skeleton=True)
        
        viz.set_show_skeleton(False)
        assert viz.show_skeleton is False
    
    def test_set_show_bbox(self):
        """测试设置边界框显示"""
        viz = Visualizer(show_bbox=True)
        
        viz.set_show_bbox(False)
        assert viz.show_bbox is False
    
    def test_set_show_label(self):
        """测试设置标签显示"""
        viz = Visualizer(show_label=True)
        
        viz.set_show_label(False)
        assert viz.show_label is False


class TestMultiPersonVisualization:
    """测试多人可视化"""
    
    def test_draw_two_people(self):
        """测试绘制两个人"""
        viz = Visualizer()
        frame = create_test_frame(640, 480)
        
        detections = [
            create_test_detection(
                person_id=1,
                bbox=BoundingBox(50, 50, 150, 350)
            ),
            create_test_detection(
                person_id=2,
                bbox=BoundingBox(300, 50, 150, 350)
            )
        ]
        
        result = viz.draw(frame, detections)
        
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)
    
    def test_draw_multiple_people_with_different_poses(self):
        """测试绘制不同姿态的多人"""
        viz = Visualizer()
        frame = create_test_frame(640, 480)
        
        detections = [
            create_test_detection(person_id=1, bbox=BoundingBox(50, 50, 120, 300)),
            create_test_detection(person_id=2, bbox=BoundingBox(200, 50, 120, 300)),
            create_test_detection(person_id=3, bbox=BoundingBox(350, 50, 120, 300))
        ]
        
        # 修改姿态分类
        detections[1].pose_classification.pose_state = PoseState.SITTING
        detections[2].pose_classification.pose_state = PoseState.SQUATTING
        
        result = viz.draw(frame, detections)
        
        assert not np.array_equal(result, frame)


class TestVisualizationConfig:
    """测试可视化配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = VisualizationConfig()
        
        assert config.bbox_color == (0, 255, 0)
        assert config.bbox_thickness == 2
        assert config.landmark_radius == 3
        assert config.landmark_color == (0, 0, 255)
        assert config.skeleton_thickness == 2
        assert config.skeleton_color == (255, 0, 0)
        assert config.label_font_scale == 0.6
        assert config.label_thickness == 2
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = VisualizationConfig(
            bbox_color=(255, 255, 0),
            bbox_thickness=3,
            landmark_radius=5,
            skeleton_thickness=3
        )
        
        assert config.bbox_color == (255, 255, 0)
        assert config.bbox_thickness == 3
        assert config.landmark_radius == 5
        assert config.skeleton_thickness == 3


# ============================================================================
# 属性测试
# ============================================================================

class TestVisualizationProperties:
    """
    属性 7: 可视化完整性
    
    验证需求: 4.1, 4.3, 4.4
    
    可视化系统应该：
    1. 对任何有效的检测结果都能成功绘制
    2. 不会修改原始帧
    3. 输出帧的尺寸与输入帧相同
    4. 绘制操作是幂等的（多次绘制相同结果）
    """
    
    @given(
        width=st.integers(min_value=320, max_value=1920),
        height=st.integers(min_value=240, max_value=1080),
        num_detections=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_7_visualization_completeness(
        self,
        width: int,
        height: int,
        num_detections: int
    ):
        """
        属性 7: 可视化完整性
        
        验证可视化系统对任意有效输入都能正确处理。
        """
        viz = Visualizer()
        frame = create_test_frame(width, height)
        original_frame = frame.copy()
        
        # 创建随机检测结果
        detections = []
        for i in range(num_detections):
            bbox_width = min(200, width // 2)
            bbox_height = min(400, height - 100)
            bbox_x = min(i * 150, width - bbox_width - 10)
            bbox_y = 50
            
            bbox = BoundingBox(
                x=bbox_x,
                y=bbox_y,
                width=bbox_width,
                height=bbox_height
            )
            
            detections.append(create_test_detection(
                person_id=i + 1,
                bbox=bbox
            ))
        
        # 绘制结果
        result = viz.draw(frame, detections)
        
        # 验证属性
        # 1. 输出帧尺寸与输入相同
        assert result.shape == frame.shape
        
        # 2. 原始帧未被修改
        assert np.array_equal(frame, original_frame)
        
        # 3. 如果有检测结果，输出应该与输入不同
        if num_detections > 0:
            assert not np.array_equal(result, frame)
        else:
            assert np.array_equal(result, frame)
        
        # 4. 幂等性：多次绘制相同结果
        result2 = viz.draw(frame.copy(), detections)
        assert np.array_equal(result, result2)
    
    @given(
        show_landmarks=st.booleans(),
        show_skeleton=st.booleans(),
        show_bbox=st.booleans(),
        show_label=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_property_7_configuration_consistency(
        self,
        show_landmarks: bool,
        show_skeleton: bool,
        show_bbox: bool,
        show_label: bool
    ):
        """
        属性 7: 配置一致性
        
        验证不同配置组合都能正常工作。
        """
        viz = Visualizer(
            show_landmarks=show_landmarks,
            show_skeleton=show_skeleton,
            show_bbox=show_bbox,
            show_label=show_label
        )
        
        frame = create_test_frame(640, 480)
        detection = create_test_detection()
        
        # 应该能成功绘制
        result = viz.draw(frame, [detection])
        
        assert result.shape == frame.shape
        
        # 如果所有选项都关闭，输出应该与输入相同
        if not any([show_landmarks, show_skeleton, show_bbox, show_label]):
            assert np.array_equal(result, frame)
    
    @given(
        bbox_x=st.integers(min_value=0, max_value=500),
        bbox_y=st.integers(min_value=0, max_value=300),
        bbox_width=st.integers(min_value=50, max_value=200),
        bbox_height=st.integers(min_value=100, max_value=400)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_7_bbox_position_robustness(
        self,
        bbox_x: int,
        bbox_y: int,
        bbox_width: int,
        bbox_height: int
    ):
        """
        属性 7: 边界框位置鲁棒性
        
        验证任意位置的边界框都能正确绘制。
        """
        viz = Visualizer()
        frame = create_test_frame(640, 480)
        
        # 确保边界框在帧范围内
        bbox_x = min(bbox_x, 640 - bbox_width)
        bbox_y = min(bbox_y, 480 - bbox_height)
        
        bbox = BoundingBox(x=bbox_x, y=bbox_y, width=bbox_width, height=bbox_height)
        detection = create_test_detection(bbox=bbox)
        
        # 应该能成功绘制
        result = viz.draw(frame, [detection])
        
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

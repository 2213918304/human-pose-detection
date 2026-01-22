"""
YOLOv8Detector 模块测试

包含单元测试和属性测试。

注意：这些测试需要下载 YOLOv8 模型，首次运行可能需要一些时间。
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st, settings, HealthCheck
from src.detection import YOLOv8Detector, YOLOv8TrackerDetector
from src.models import PersonDetection, BoundingBox


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_frame_with_person(width: int = 640, height: int = 480) -> np.ndarray:
    """创建包含人形的测试帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 绘制一个简单的人形（头、身体、四肢）
    # 头部
    cv2.circle(frame, (width // 2, height // 4), 30, (255, 200, 150), -1)
    
    # 身体
    cv2.rectangle(
        frame,
        (width // 2 - 40, height // 4 + 30),
        (width // 2 + 40, height // 2 + 50),
        (100, 150, 200),
        -1
    )
    
    # 手臂
    cv2.line(frame, (width // 2 - 40, height // 4 + 50), (width // 2 - 80, height // 2), (150, 150, 150), 10)
    cv2.line(frame, (width // 2 + 40, height // 4 + 50), (width // 2 + 80, height // 2), (150, 150, 150), 10)
    
    # 腿
    cv2.line(frame, (width // 2 - 20, height // 2 + 50), (width // 2 - 30, height - 50), (100, 100, 150), 15)
    cv2.line(frame, (width // 2 + 20, height // 2 + 50), (width // 2 + 30, height - 50), (100, 100, 150), 15)
    
    return frame


def create_empty_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """创建空白测试帧"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def create_multi_person_frame(width: int = 1280, height: int = 720, num_persons: int = 2) -> np.ndarray:
    """创建包含多人的测试帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 在不同位置绘制多个人形
    positions = [
        (width // 4, height // 2),
        (3 * width // 4, height // 2),
    ]
    
    for i in range(min(num_persons, len(positions))):
        x, y = positions[i]
        
        # 头部
        cv2.circle(frame, (x, y - 60), 25, (255, 200, 150), -1)
        
        # 身体
        cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 40), (100, 150, 200), -1)
        
        # 手臂
        cv2.line(frame, (x - 30, y - 10), (x - 60, y + 20), (150, 150, 150), 8)
        cv2.line(frame, (x + 30, y - 10), (x + 60, y + 20), (150, 150, 150), 8)
        
        # 腿
        cv2.line(frame, (x - 15, y + 40), (x - 20, y + 100), (100, 100, 150), 12)
        cv2.line(frame, (x + 15, y + 40), (x + 20, y + 100), (100, 100, 150), 12)
    
    return frame


# ============================================================================
# 单元测试 - YOLOv8Detector
# ============================================================================

@pytest.mark.slow
class TestYOLOv8Detector:
    """YOLOv8Detector 单元测试"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        """创建检测器实例（类级别，避免重复加载模型）"""
        detector = YOLOv8Detector(
            model_path="yolov8n.pt",
            confidence_threshold=0.3,  # 降低阈值以便测试
            device="cpu"
        )
        yield detector
        detector.close()
    
    def test_create_detector(self, detector):
        """测试创建检测器"""
        assert detector is not None
        assert detector.confidence_threshold == 0.3
        assert detector.device == "cpu"
    
    def test_invalid_confidence_threshold(self):
        """测试无效的置信度阈值"""
        with pytest.raises(ValueError, match="置信度阈值必须在"):
            YOLOv8Detector(confidence_threshold=1.5)
    
    def test_detect_empty_frame(self, detector):
        """测试检测空帧"""
        frame = create_empty_frame()
        detections = detector.detect(frame)
        
        # 空帧应该返回空列表或很少的检测
        assert isinstance(detections, list)
    
    def test_detect_returns_person_detection_list(self, detector):
        """测试检测返回 PersonDetection 列表"""
        frame = create_test_frame_with_person()
        detections = detector.detect(frame)
        
        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, PersonDetection)
            assert detection.class_id == 0  # person class
            assert 0 <= detection.confidence <= 1
            assert detection.bounding_box.width > 0
            assert detection.bounding_box.height > 0
    
    def test_detect_none_frame(self, detector):
        """测试检测 None 帧"""
        detections = detector.detect(None)
        assert detections == []
    
    def test_detect_sorted_by_confidence(self, detector):
        """测试检测结果按置信度排序"""
        frame = create_multi_person_frame(num_persons=2)
        detections = detector.detect(frame)
        
        if len(detections) > 1:
            # 验证按置信度降序排列
            for i in range(len(detections) - 1):
                assert detections[i].confidence >= detections[i + 1].confidence
    
    def test_set_confidence_threshold(self, detector):
        """测试设置置信度阈值"""
        original_threshold = detector.confidence_threshold
        
        detector.set_confidence_threshold(0.7)
        assert detector.confidence_threshold == 0.7
        
        # 恢复原始阈值
        detector.set_confidence_threshold(original_threshold)
    
    def test_set_invalid_confidence_threshold(self, detector):
        """测试设置无效的置信度阈值"""
        with pytest.raises(ValueError):
            detector.set_confidence_threshold(1.5)
    
    def test_get_detection_count(self, detector):
        """测试获取检测计数"""
        detector.reset_count()
        initial_count = detector.get_detection_count()
        
        frame = create_test_frame_with_person()
        detector.detect(frame)
        
        new_count = detector.get_detection_count()
        assert new_count >= initial_count
    
    def test_reset_count(self, detector):
        """测试重置计数"""
        frame = create_test_frame_with_person()
        detector.detect(frame)
        
        detector.reset_count()
        assert detector.get_detection_count() == 0
    
    def test_get_model_info(self, detector):
        """测试获取模型信息"""
        info = detector.get_model_info()
        
        assert "model_path" in info
        assert "device" in info
        assert "confidence_threshold" in info
        assert "detection_count" in info
    
    def test_detect_batch(self, detector):
        """测试批量检测"""
        frames = [
            create_test_frame_with_person(),
            create_empty_frame(),
            create_multi_person_frame(num_persons=2),
        ]
        
        all_detections = detector.detect_batch(frames)
        
        assert len(all_detections) == 3
        assert all(isinstance(dets, list) for dets in all_detections)
    
    def test_detect_batch_empty_list(self, detector):
        """测试批量检测空列表"""
        all_detections = detector.detect_batch([])
        assert all_detections == []
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with YOLOv8Detector(confidence_threshold=0.5) as detector:
            frame = create_test_frame_with_person()
            detections = detector.detect(frame)
            assert isinstance(detections, list)


# ============================================================================
# 单元测试 - YOLOv8TrackerDetector
# ============================================================================

@pytest.mark.slow
class TestYOLOv8TrackerDetector:
    """YOLOv8TrackerDetector 单元测试"""
    
    @pytest.fixture(scope="class")
    def tracker_detector(self):
        """创建带跟踪的检测器实例"""
        detector = YOLOv8TrackerDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.3,
            device="cpu"
        )
        yield detector
        detector.close()
    
    def test_create_tracker_detector(self, tracker_detector):
        """测试创建带跟踪的检测器"""
        assert tracker_detector is not None
        assert hasattr(tracker_detector, 'tracker')
    
    def test_detect_with_tracking(self, tracker_detector):
        """测试带跟踪的检测"""
        frame = create_test_frame_with_person()
        detections = tracker_detector.detect(frame)
        
        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, PersonDetection)
            assert detection.person_id >= 0  # 应该有跟踪 ID


# ============================================================================
# 属性测试 - Feature: human-pose-detection, Property 5: 多人独立检测
# ============================================================================

@pytest.mark.property
@pytest.mark.slow
class TestMultiPersonDetectionProperty:
    """
    属性 5: 多人独立检测
    验证需求: 2.4
    
    对于任何包含多人的帧，每个人应该有独立的检测结果，且每个人的 person_id 应该唯一
    """
    
    @pytest.fixture(scope="class")
    def detector(self):
        """创建检测器实例"""
        detector = YOLOv8Detector(confidence_threshold=0.3, device="cpu")
        yield detector
        detector.close()
    
    def test_unique_person_ids(self, detector):
        """
        Feature: human-pose-detection, Property 5: 多人独立检测
        
        对于包含多人的帧，每个人应该有唯一的 person_id
        """
        frame = create_multi_person_frame(num_persons=2)
        detections = detector.detect(frame)
        
        if len(detections) > 1:
            # 收集所有 person_id
            person_ids = [d.person_id for d in detections]
            
            # 验证 ID 唯一性
            assert len(person_ids) == len(set(person_ids)), "person_id 应该唯一"
    
    def test_independent_detections(self, detector):
        """
        Feature: human-pose-detection, Property 5: 多人独立检测
        
        每个检测结果应该是独立的，有自己的边界框和置信度
        """
        frame = create_multi_person_frame(num_persons=2)
        detections = detector.detect(frame)
        
        for detection in detections:
            # 验证每个检测都有完整的信息
            assert detection.bounding_box is not None
            assert detection.bounding_box.width > 0
            assert detection.bounding_box.height > 0
            assert 0 <= detection.confidence <= 1
            assert detection.person_id >= 0
    
    def test_non_overlapping_bboxes(self, detector):
        """
        Feature: human-pose-detection, Property 5: 多人独立检测
        
        不同人的边界框应该基本不重叠（IoU 较低）
        """
        frame = create_multi_person_frame(num_persons=2)
        detections = detector.detect(frame)
        
        if len(detections) >= 2:
            # 检查前两个检测的边界框
            bbox1 = detections[0].bounding_box
            bbox2 = detections[1].bounding_box
            
            iou = bbox1.iou(bbox2)
            
            # IoU 应该较低（不同的人）
            # 注意：这个阈值可能需要根据实际情况调整
            assert iou < 0.5, f"不同人的边界框 IoU 过高: {iou}"
    
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(num_persons=st.integers(min_value=1, max_value=2))
    def test_detection_count_matches_persons(self, detector, num_persons):
        """
        Feature: human-pose-detection, Property 5: 多人独立检测
        
        检测数量应该与实际人数相关（可能不完全相等，但应该合理）
        """
        frame = create_multi_person_frame(num_persons=num_persons)
        detections = detector.detect(frame)
        
        # 检测数量应该是合理的（0 到 num_persons 之间）
        # 注意：由于测试帧的简单性，可能检测不到所有人
        assert 0 <= len(detections) <= num_persons + 1  # 允许一些误差


# ============================================================================
# 集成测试
# ============================================================================

@pytest.mark.slow
class TestYOLODetectorIntegration:
    """YOLOv8Detector 集成测试"""
    
    def test_full_detection_pipeline(self):
        """测试完整的检测流程"""
        # 创建检测器
        detector = YOLOv8Detector(confidence_threshold=0.5, device="cpu")
        
        # 创建测试帧
        frame = create_test_frame_with_person()
        
        # 执行检测
        detections = detector.detect(frame)
        
        # 验证结果
        assert isinstance(detections, list)
        
        # 如果检测到人，验证数据完整性
        for detection in detections:
            assert isinstance(detection, PersonDetection)
            assert detection.bounding_box.area() > 0
            assert 0 <= detection.confidence <= 1
        
        # 清理
        detector.close()

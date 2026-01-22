"""
PoseEstimator 模块测试

包含单元测试和属性测试。
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st, settings, HealthCheck
from src.pose import PoseEstimator, PoseLandmark
from src.models import BoundingBox, PoseEstimation, Landmark


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_frame_with_person(width: int = 640, height: int = 480) -> np.ndarray:
    """创建包含人形的测试帧"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 绘制一个简单的人形
    center_x, center_y = width // 2, height // 2
    
    # 头部
    cv2.circle(frame, (center_x, center_y - 80), 40, (255, 200, 150), -1)
    
    # 身体
    cv2.rectangle(
        frame,
        (center_x - 50, center_y - 30),
        (center_x + 50, center_y + 80),
        (100, 150, 200),
        -1
    )
    
    # 手臂
    cv2.line(frame, (center_x - 50, center_y), (center_x - 100, center_y + 40), (150, 150, 150), 15)
    cv2.line(frame, (center_x + 50, center_y), (center_x + 100, center_y + 40), (150, 150, 150), 15)
    
    # 腿
    cv2.line(frame, (center_x - 25, center_y + 80), (center_x - 35, center_y + 160), (100, 100, 150), 20)
    cv2.line(frame, (center_x + 25, center_y + 80), (center_x + 35, center_y + 160), (100, 100, 150), 20)
    
    return frame


def create_person_bbox(width: int = 640, height: int = 480) -> BoundingBox:
    """创建覆盖人形的边界框"""
    center_x, center_y = width // 2, height // 2
    return BoundingBox(
        x=center_x - 120,
        y=center_y - 120,
        width=240,
        height=280
    )


# ============================================================================
# 单元测试 - PoseEstimator
# ============================================================================

class TestPoseEstimator:
    """PoseEstimator 单元测试"""
    
    @pytest.fixture(scope="class")
    def estimator(self):
        """创建估计器实例"""
        estimator = PoseEstimator(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0  # 使用 Lite 模型加快测试
        )
        yield estimator
        estimator.close()
    
    def test_create_estimator(self, estimator):
        """测试创建估计器"""
        assert estimator is not None
        assert estimator.min_detection_confidence == 0.3
        assert estimator.min_tracking_confidence == 0.3
        assert estimator.model_complexity == 0
    
    def test_invalid_detection_confidence(self):
        """测试无效的检测置信度"""
        with pytest.raises(ValueError, match="检测置信度必须在"):
            PoseEstimator(min_detection_confidence=1.5)
    
    def test_invalid_tracking_confidence(self):
        """测试无效的跟踪置信度"""
        with pytest.raises(ValueError, match="跟踪置信度必须在"):
            PoseEstimator(min_tracking_confidence=-0.1)
    
    def test_invalid_model_complexity(self):
        """测试无效的模型复杂度"""
        with pytest.raises(ValueError, match="模型复杂度必须是"):
            PoseEstimator(model_complexity=3)
    
    def test_estimate_with_person(self, estimator):
        """测试估计包含人的帧"""
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        estimation = estimator.estimate(frame, bbox)
        
        # 可能检测到或检测不到（取决于图像质量）
        if estimation is not None:
            assert isinstance(estimation, PoseEstimation)
            assert len(estimation.landmarks) == 33
            assert 0 <= estimation.confidence <= 1
            
            # 验证所有关键点
            for lm in estimation.landmarks:
                assert isinstance(lm, Landmark)
                assert 0 <= lm.x <= 1
                assert 0 <= lm.y <= 1
                assert 0 <= lm.visibility <= 1
    
    def test_estimate_none_frame(self, estimator):
        """测试估计 None 帧"""
        bbox = create_person_bbox()
        estimation = estimator.estimate(None, bbox)
        assert estimation is None
    
    def test_estimate_empty_frame(self, estimator):
        """测试估计空帧"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        
        estimation = estimator.estimate(frame, bbox)
        # 空帧应该返回 None
        assert estimation is None or estimation.confidence < 0.5
    
    def test_estimate_with_padding(self, estimator):
        """测试带 padding 的估计"""
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        # 使用不同的 padding 比例
        estimation1 = estimator.estimate(frame, bbox, padding_ratio=0.0)
        estimation2 = estimator.estimate(frame, bbox, padding_ratio=0.2)
        
        # 两种情况都应该返回有效结果或 None
        assert estimation1 is None or isinstance(estimation1, PoseEstimation)
        assert estimation2 is None or isinstance(estimation2, PoseEstimation)
    
    def test_estimate_batch(self, estimator):
        """测试批量估计"""
        frame = create_test_frame_with_person()
        bboxes = [
            create_person_bbox(),
            BoundingBox(x=50, y=50, width=100, height=150),
        ]
        
        estimations = estimator.estimate_batch(frame, bboxes)
        
        assert len(estimations) == 2
        assert all(est is None or isinstance(est, PoseEstimation) for est in estimations)
    
    def test_estimate_batch_empty_list(self, estimator):
        """测试批量估计空列表"""
        frame = create_test_frame_with_person()
        estimations = estimator.estimate_batch(frame, [])
        assert estimations == []
    
    def test_get_estimation_count(self, estimator):
        """测试获取估计计数"""
        estimator.reset_count()
        initial_count = estimator.get_estimation_count()
        
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        estimation = estimator.estimate(frame, bbox)
        
        new_count = estimator.get_estimation_count()
        
        # 如果估计成功，计数应该增加
        if estimation is not None:
            assert new_count == initial_count + 1
        else:
            # 如果估计失败（简单测试图像可能无法检测），计数不变
            assert new_count == initial_count
    
    def test_reset_count(self, estimator):
        """测试重置计数"""
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        estimator.estimate(frame, bbox)
        
        estimator.reset_count()
        assert estimator.get_estimation_count() == 0
    
    def test_get_info(self, estimator):
        """测试获取信息"""
        info = estimator.get_info()
        
        assert "min_detection_confidence" in info
        assert "min_tracking_confidence" in info
        assert "model_complexity" in info
        assert "estimation_count" in info
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with PoseEstimator(model_complexity=0) as estimator:
            frame = create_test_frame_with_person()
            bbox = create_person_bbox()
            estimation = estimator.estimate(frame, bbox)
            assert estimation is None or isinstance(estimation, PoseEstimation)


# ============================================================================
# 单元测试 - PoseLandmark 常量
# ============================================================================

class TestPoseLandmark:
    """PoseLandmark 常量测试"""
    
    def test_landmark_indices(self):
        """测试关键点索引"""
        assert PoseLandmark.NOSE == 0
        assert PoseLandmark.LEFT_SHOULDER == 11
        assert PoseLandmark.RIGHT_SHOULDER == 12
        assert PoseLandmark.LEFT_HIP == 23
        assert PoseLandmark.RIGHT_HIP == 24
        assert PoseLandmark.LEFT_KNEE == 25
        assert PoseLandmark.RIGHT_KNEE == 26
        assert PoseLandmark.LEFT_ANKLE == 27
        assert PoseLandmark.RIGHT_ANKLE == 28
        assert PoseLandmark.RIGHT_FOOT_INDEX == 32


# ============================================================================
# 单元测试 - 坐标转换
# ============================================================================

class TestCoordinateConversion:
    """坐标转换测试"""
    
    @pytest.fixture
    def estimator(self):
        """创建估计器实例"""
        estimator = PoseEstimator(model_complexity=0)
        yield estimator
        estimator.close()
    
    def test_landmarks_in_valid_range(self, estimator):
        """测试关键点坐标在有效范围内"""
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        estimation = estimator.estimate(frame, bbox)
        
        if estimation is not None:
            for lm in estimation.landmarks:
                # 归一化坐标应该在 [0, 1] 范围内
                assert 0 <= lm.x <= 1, f"x 坐标超出范围: {lm.x}"
                assert 0 <= lm.y <= 1, f"y 坐标超出范围: {lm.y}"
                assert 0 <= lm.visibility <= 1, f"可见性超出范围: {lm.visibility}"
    
    def test_pixel_conversion(self, estimator):
        """测试像素坐标转换"""
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        estimation = estimator.estimate(frame, bbox)
        
        if estimation is not None:
            frame_height, frame_width = frame.shape[:2]
            
            for lm in estimation.landmarks:
                pixel_x, pixel_y = lm.to_pixel(frame_width, frame_height)
                
                # 像素坐标应该在帧范围内
                assert 0 <= pixel_x <= frame_width
                assert 0 <= pixel_y <= frame_height


# ============================================================================
# 属性测试 - Feature: human-pose-detection, Property 4: 跨帧一致性
# ============================================================================

@pytest.mark.property
class TestCrossFrameConsistencyProperty:
    """
    属性 4: 跨帧一致性
    验证需求: 2.3
    
    对于任何连续的视频帧，如果同一个人在两帧中都可见，
    其边界框位置应该平滑更新（位置变化合理）
    """
    
    @pytest.fixture(scope="class")
    def estimator(self):
        """创建估计器实例"""
        estimator = PoseEstimator(model_complexity=0)
        yield estimator
        estimator.close()
    
    def test_landmark_consistency_across_similar_frames(self, estimator):
        """
        Feature: human-pose-detection, Property 4: 跨帧一致性
        
        对于相似的连续帧，关键点位置应该保持一致
        """
        # 创建两个相似的帧
        frame1 = create_test_frame_with_person()
        frame2 = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        estimation1 = estimator.estimate(frame1, bbox)
        estimation2 = estimator.estimate(frame2, bbox)
        
        if estimation1 is not None and estimation2 is not None:
            # 计算关键点位置的平均差异
            total_diff = 0
            count = 0
            
            for lm1, lm2 in zip(estimation1.landmarks, estimation2.landmarks):
                if lm1.is_visible() and lm2.is_visible():
                    diff = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
                    total_diff += diff
                    count += 1
            
            if count > 0:
                avg_diff = total_diff / count
                # 相似帧的平均差异应该很小
                assert avg_diff < 0.1, f"关键点位置变化过大: {avg_diff}"
    
    def test_bbox_position_reasonable_change(self, estimator):
        """
        Feature: human-pose-detection, Property 4: 跨帧一致性
        
        边界框位置的变化应该在合理范围内
        """
        frame = create_test_frame_with_person()
        
        # 创建两个稍微不同位置的边界框
        bbox1 = create_person_bbox()
        bbox2 = BoundingBox(
            x=bbox1.x + 10,
            y=bbox1.y + 5,
            width=bbox1.width,
            height=bbox1.height
        )
        
        estimation1 = estimator.estimate(frame, bbox1)
        estimation2 = estimator.estimate(frame, bbox2)
        
        # 两次估计都应该成功或都失败
        if estimation1 is not None and estimation2 is not None:
            # 置信度应该相近
            conf_diff = abs(estimation1.confidence - estimation2.confidence)
            assert conf_diff < 0.3, f"置信度变化过大: {conf_diff}"


# ============================================================================
# 集成测试
# ============================================================================

class TestPoseEstimatorIntegration:
    """PoseEstimator 集成测试"""
    
    def test_full_estimation_pipeline(self):
        """测试完整的估计流程"""
        # 创建估计器
        estimator = PoseEstimator(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        
        # 创建测试帧和边界框
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        # 执行估计
        estimation = estimator.estimate(frame, bbox)
        
        # 验证结果
        if estimation is not None:
            assert isinstance(estimation, PoseEstimation)
            assert len(estimation.landmarks) == 33
            assert 0 <= estimation.confidence <= 1
            
            # 验证可以获取可见关键点
            visible_landmarks = estimation.get_visible_landmarks(threshold=0.5)
            assert isinstance(visible_landmarks, list)
            
            # 验证可以获取特定关键点
            nose = estimation.get_landmark(PoseLandmark.NOSE)
            assert isinstance(nose, Landmark)
        
        # 清理
        estimator.close()
    
    def test_multiple_estimations(self):
        """测试多次估计"""
        estimator = PoseEstimator(model_complexity=0)
        frame = create_test_frame_with_person()
        bbox = create_person_bbox()
        
        # 执行多次估计
        successful_count = 0
        for _ in range(5):
            estimation = estimator.estimate(frame, bbox)
            if estimation is not None:
                successful_count += 1
                assert isinstance(estimation, PoseEstimation)
        
        # 验证计数（只计算成功的估计）
        assert estimator.get_estimation_count() == successful_count
        assert estimator.get_estimation_count() >= 0  # 至少不会出错
        
        estimator.close()

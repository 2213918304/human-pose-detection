"""
数据模型测试

包含单元测试和属性测试。
"""

import pytest
from hypothesis import given, strategies as st, assume
from src.models import (
    PoseState,
    Landmark,
    BoundingBox,
    PersonDetection,
    PoseEstimation,
    PoseClassification,
    DetectionResult,
)


# ============================================================================
# Hypothesis 策略定义
# ============================================================================

@st.composite
def landmark_strategy(draw):
    """生成有效的 Landmark"""
    return Landmark(
        x=draw(st.floats(min_value=0.0, max_value=1.0)),
        y=draw(st.floats(min_value=0.0, max_value=1.0)),
        z=draw(st.floats(min_value=-1.0, max_value=1.0)),
        visibility=draw(st.floats(min_value=0.0, max_value=1.0)),
    )


@st.composite
def bounding_box_strategy(draw):
    """生成有效的 BoundingBox"""
    return BoundingBox(
        x=draw(st.integers(min_value=0, max_value=1920)),
        y=draw(st.integers(min_value=0, max_value=1080)),
        width=draw(st.integers(min_value=1, max_value=500)),
        height=draw(st.integers(min_value=1, max_value=500)),
    )


@st.composite
def person_detection_strategy(draw):
    """生成有效的 PersonDetection"""
    return PersonDetection(
        person_id=draw(st.integers(min_value=0, max_value=100)),
        bounding_box=draw(bounding_box_strategy()),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        class_id=0,
    )


@st.composite
def pose_estimation_strategy(draw):
    """生成有效的 PoseEstimation（33个关键点）"""
    landmarks = [draw(landmark_strategy()) for _ in range(33)]
    return PoseEstimation(
        landmarks=landmarks,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
    )


@st.composite
def pose_classification_strategy(draw):
    """生成有效的 PoseClassification"""
    return PoseClassification(
        pose_state=draw(st.sampled_from(list(PoseState))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        features=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=-180.0, max_value=180.0),
            max_size=10
        )),
    )


# ============================================================================
# 单元测试 - Landmark
# ============================================================================

class TestLandmark:
    """Landmark 单元测试"""
    
    def test_create_valid_landmark(self):
        """测试创建有效的 Landmark"""
        lm = Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
        assert lm.x == 0.5
        assert lm.y == 0.5
        assert lm.z == 0.0
        assert lm.visibility == 1.0
    
    def test_invalid_x_coordinate(self):
        """测试无效的 x 坐标"""
        with pytest.raises(ValueError, match="x 坐标必须在"):
            Landmark(x=1.5, y=0.5, z=0.0, visibility=1.0)
    
    def test_invalid_y_coordinate(self):
        """测试无效的 y 坐标"""
        with pytest.raises(ValueError, match="y 坐标必须在"):
            Landmark(x=0.5, y=-0.1, z=0.0, visibility=1.0)
    
    def test_invalid_visibility(self):
        """测试无效的可见性"""
        with pytest.raises(ValueError, match="可见性必须在"):
            Landmark(x=0.5, y=0.5, z=0.0, visibility=1.5)
    
    def test_to_pixel(self):
        """测试像素坐标转换"""
        lm = Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
        pixel_x, pixel_y = lm.to_pixel(640, 480)
        assert pixel_x == 320
        assert pixel_y == 240
    
    def test_is_visible(self):
        """测试可见性判断"""
        lm_visible = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.8)
        lm_invisible = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.3)
        
        assert lm_visible.is_visible(threshold=0.5)
        assert not lm_invisible.is_visible(threshold=0.5)


# ============================================================================
# 单元测试 - BoundingBox
# ============================================================================

class TestBoundingBox:
    """BoundingBox 单元测试"""
    
    def test_create_valid_bbox(self):
        """测试创建有效的 BoundingBox"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        assert bbox.x == 100
        assert bbox.y == 100
        assert bbox.width == 200
        assert bbox.height == 300
    
    def test_invalid_width(self):
        """测试无效的宽度"""
        with pytest.raises(ValueError, match="宽度必须大于 0"):
            BoundingBox(x=100, y=100, width=0, height=300)
    
    def test_invalid_height(self):
        """测试无效的高度"""
        with pytest.raises(ValueError, match="高度必须大于 0"):
            BoundingBox(x=100, y=100, width=200, height=-10)
    
    def test_area(self):
        """测试面积计算"""
        bbox = BoundingBox(x=0, y=0, width=10, height=20)
        assert bbox.area() == 200
    
    def test_center(self):
        """测试中心点计算"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        center_x, center_y = bbox.center()
        assert center_x == 200
        assert center_y == 250
    
    def test_expand(self):
        """测试边界框扩展"""
        bbox = BoundingBox(x=100, y=100, width=100, height=100)
        expanded = bbox.expand(padding_ratio=0.1)
        
        assert expanded.x == 90
        assert expanded.y == 90
        assert expanded.width == 120
        assert expanded.height == 120
    
    def test_clip(self):
        """测试边界框裁剪"""
        bbox = BoundingBox(x=1800, y=1000, width=300, height=200)
        clipped = bbox.clip(frame_width=1920, frame_height=1080)
        
        assert clipped.x == 1800
        assert clipped.y == 1000
        assert clipped.width == 120  # 裁剪到帧边界
        assert clipped.height == 80
    
    def test_iou_no_overlap(self):
        """测试无重叠的 IoU"""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        bbox2 = BoundingBox(x=20, y=20, width=10, height=10)
        assert bbox1.iou(bbox2) == 0.0
    
    def test_iou_full_overlap(self):
        """测试完全重叠的 IoU"""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        bbox2 = BoundingBox(x=0, y=0, width=10, height=10)
        assert bbox1.iou(bbox2) == 1.0
    
    def test_iou_partial_overlap(self):
        """测试部分重叠的 IoU"""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        bbox2 = BoundingBox(x=5, y=5, width=10, height=10)
        iou = bbox1.iou(bbox2)
        assert 0.0 < iou < 1.0


# ============================================================================
# 单元测试 - PersonDetection
# ============================================================================

class TestPersonDetection:
    """PersonDetection 单元测试"""
    
    def test_create_valid_detection(self):
        """测试创建有效的 PersonDetection"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        detection = PersonDetection(
            person_id=0,
            bounding_box=bbox,
            confidence=0.95,
        )
        assert detection.person_id == 0
        assert detection.confidence == 0.95
        assert detection.class_id == 0
    
    def test_invalid_confidence(self):
        """测试无效的置信度"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        with pytest.raises(ValueError, match="置信度必须在"):
            PersonDetection(person_id=0, bounding_box=bbox, confidence=1.5)
    
    def test_invalid_person_id(self):
        """测试无效的 person_id"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        with pytest.raises(ValueError, match="person_id 必须非负"):
            PersonDetection(person_id=-1, bounding_box=bbox, confidence=0.9)


# ============================================================================
# 单元测试 - PoseEstimation
# ============================================================================

class TestPoseEstimation:
    """PoseEstimation 单元测试"""
    
    def test_create_valid_pose_estimation(self):
        """测试创建有效的 PoseEstimation"""
        landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
        pose = PoseEstimation(landmarks=landmarks, confidence=0.9)
        assert len(pose.landmarks) == 33
        assert pose.confidence == 0.9
    
    def test_invalid_landmark_count(self):
        """测试无效的关键点数量"""
        landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(20)]
        with pytest.raises(ValueError, match="必须有 33 个关键点"):
            PoseEstimation(landmarks=landmarks, confidence=0.9)
    
    def test_get_visible_landmarks(self):
        """测试获取可见关键点"""
        landmarks = [
            Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9),
            Landmark(x=0.5, y=0.5, z=0.0, visibility=0.3),
        ] + [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.8) for _ in range(31)]
        
        pose = PoseEstimation(landmarks=landmarks, confidence=0.9)
        visible = pose.get_visible_landmarks(threshold=0.5)
        assert len(visible) == 32  # 1个不可见
    
    def test_get_landmark(self):
        """测试获取指定关键点"""
        landmarks = [Landmark(x=float(i)/33, y=0.5, z=0.0, visibility=1.0) for i in range(33)]
        pose = PoseEstimation(landmarks=landmarks, confidence=0.9)
        
        lm = pose.get_landmark(10)
        assert lm.x == pytest.approx(10/33, rel=1e-5)
    
    def test_get_landmark_invalid_index(self):
        """测试无效的关键点索引"""
        landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
        pose = PoseEstimation(landmarks=landmarks, confidence=0.9)
        
        with pytest.raises(IndexError, match="关键点索引必须在"):
            pose.get_landmark(33)


# ============================================================================
# 单元测试 - DetectionResult
# ============================================================================

class TestDetectionResult:
    """DetectionResult 单元测试"""
    
    def test_create_detection_result_without_pose(self):
        """测试创建不含姿态的检测结果"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        result = DetectionResult(
            person_id=0,
            bounding_box=bbox,
            confidence=0.95,
        )
        assert not result.has_pose()
    
    def test_create_detection_result_with_pose(self):
        """测试创建包含姿态的检测结果"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
        classification = PoseClassification(
            pose_state=PoseState.STANDING,
            confidence=0.9,
        )
        
        result = DetectionResult(
            person_id=0,
            bounding_box=bbox,
            confidence=0.95,
            landmarks=landmarks,
            pose_classification=classification,
        )
        assert result.has_pose()
    
    def test_to_dict(self):
        """测试转换为字典"""
        bbox = BoundingBox(x=100, y=100, width=200, height=300)
        result = DetectionResult(
            person_id=0,
            bounding_box=bbox,
            confidence=0.95,
            timestamp=1.5,
        )
        
        data = result.to_dict()
        assert data["person_id"] == 0
        assert data["confidence"] == 0.95
        assert data["timestamp"] == 1.5
        assert data["bounding_box"]["x"] == 100


# ============================================================================
# 属性测试 - Feature: human-pose-detection, Property 3: 检测完整性
# ============================================================================

@pytest.mark.property
class TestDetectionIntegrityProperty:
    """
    属性 3: 检测完整性
    验证需求: 2.1, 2.2
    
    对于任何检测到的人体，系统必须生成包含边界框和置信度的完整检测结果
    """
    
    @given(person_detection_strategy())
    def test_person_detection_has_valid_bbox_and_confidence(self, detection):
        """
        Feature: human-pose-detection, Property 3: 检测完整性
        
        对于任何 PersonDetection，必须包含有效的边界框和置信度
        """
        # 验证边界框存在且有效
        assert detection.bounding_box is not None
        assert detection.bounding_box.width > 0
        assert detection.bounding_box.height > 0
        assert detection.bounding_box.area() > 0
        
        # 验证置信度在有效范围内
        assert 0.0 <= detection.confidence <= 1.0
        
        # 验证 person_id 非负
        assert detection.person_id >= 0
    
    @given(
        bounding_box_strategy(),
        st.floats(min_value=0.0, max_value=1.0),
        st.integers(min_value=0, max_value=100),
    )
    def test_detection_result_completeness(self, bbox, confidence, person_id):
        """
        Feature: human-pose-detection, Property 3: 检测完整性
        
        对于任何检测结果，必须包含完整的必需字段
        """
        result = DetectionResult(
            person_id=person_id,
            bounding_box=bbox,
            confidence=confidence,
        )
        
        # 验证所有必需字段存在
        assert result.person_id >= 0
        assert result.bounding_box is not None
        assert 0.0 <= result.confidence <= 1.0
        
        # 验证可以转换为字典
        data = result.to_dict()
        assert "person_id" in data
        assert "bounding_box" in data
        assert "confidence" in data
    
    @given(landmark_strategy(), st.integers(min_value=1, max_value=1920), st.integers(min_value=1, max_value=1080))
    def test_landmark_pixel_conversion_is_valid(self, landmark, width, height):
        """
        Feature: human-pose-detection, Property 3: 检测完整性
        
        对于任何关键点，像素坐标转换必须产生有效的坐标
        """
        pixel_x, pixel_y = landmark.to_pixel(width, height)
        
        # 验证像素坐标在有效范围内
        assert 0 <= pixel_x <= width
        assert 0 <= pixel_y <= height
    
    @given(bounding_box_strategy())
    def test_bounding_box_has_positive_area(self, bbox):
        """
        Feature: human-pose-detection, Property 3: 检测完整性
        
        对于任何边界框，面积必须为正
        """
        assert bbox.area() > 0
        assert bbox.width > 0
        assert bbox.height > 0
    
    @given(pose_estimation_strategy())
    def test_pose_estimation_has_33_landmarks(self, pose):
        """
        Feature: human-pose-detection, Property 3: 检测完整性
        
        对于任何姿态估计，必须包含 33 个关键点
        """
        assert len(pose.landmarks) == 33
        assert 0.0 <= pose.confidence <= 1.0
        
        # 验证所有关键点都是有效的 Landmark 对象
        for lm in pose.landmarks:
            assert isinstance(lm, Landmark)
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0
            assert 0.0 <= lm.visibility <= 1.0
    
    @given(pose_classification_strategy())
    def test_pose_classification_has_state_and_confidence(self, classification):
        """
        Feature: human-pose-detection, Property 3: 检测完整性
        
        对于任何姿态分类，必须包含姿态状态和置信度
        """
        assert classification.pose_state in PoseState
        assert 0.0 <= classification.confidence <= 1.0
        assert isinstance(classification.features, dict)


# ============================================================================
# 属性测试 - 边界框属性
# ============================================================================

@pytest.mark.property
class TestBoundingBoxProperties:
    """边界框的通用属性测试"""
    
    @given(bounding_box_strategy(), st.floats(min_value=0.0, max_value=0.5))
    def test_expand_increases_size(self, bbox, padding_ratio):
        """扩展边界框应该增加尺寸"""
        expanded = bbox.expand(padding_ratio)
        assert expanded.area() >= bbox.area()
    
    @given(bounding_box_strategy())
    def test_iou_is_symmetric(self, bbox1):
        """IoU 应该是对称的"""
        bbox2 = BoundingBox(
            x=bbox1.x + 10,
            y=bbox1.y + 10,
            width=bbox1.width,
            height=bbox1.height,
        )
        assert bbox1.iou(bbox2) == pytest.approx(bbox2.iou(bbox1), rel=1e-5)
    
    @given(bounding_box_strategy())
    def test_iou_with_self_is_one(self, bbox):
        """边界框与自身的 IoU 应该为 1"""
        assert bbox.iou(bbox) == pytest.approx(1.0, rel=1e-5)
    
    @given(bounding_box_strategy(), st.integers(min_value=100, max_value=1920), st.integers(min_value=100, max_value=1080))
    def test_clip_keeps_bbox_in_bounds(self, bbox, frame_width, frame_height):
        """裁剪后的边界框应该在帧范围内"""
        clipped = bbox.clip(frame_width, frame_height)
        
        assert clipped.x >= 0
        assert clipped.y >= 0
        assert clipped.x + clipped.width <= frame_width
        assert clipped.y + clipped.height <= frame_height

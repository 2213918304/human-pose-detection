"""
核心数据模型

定义系统中使用的所有数据结构。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import numpy as np


class PoseState(Enum):
    """姿态状态枚举"""
    STANDING = "站立"
    SITTING = "坐立"
    LYING_DOWN = "躺下"
    SQUATTING = "蹲下"
    BENDING = "弯腰"
    UNKNOWN = "未知"


@dataclass
class Landmark:
    """
    关键点数据
    
    Attributes:
        x: 归一化x坐标 [0, 1]
        y: 归一化y坐标 [0, 1]
        z: 深度信息（相对于髋部中点）
        visibility: 可见性得分 [0, 1]
    """
    x: float
    y: float
    z: float
    visibility: float
    
    def __post_init__(self):
        """验证数据范围"""
        if not (0 <= self.x <= 1):
            raise ValueError(f"x 坐标必须在 [0, 1] 范围内，得到: {self.x}")
        if not (0 <= self.y <= 1):
            raise ValueError(f"y 坐标必须在 [0, 1] 范围内，得到: {self.y}")
        if not (0 <= self.visibility <= 1):
            raise ValueError(f"可见性必须在 [0, 1] 范围内，得到: {self.visibility}")
    
    def to_pixel(self, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """
        转换为像素坐标
        
        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
        
        Returns:
            (x, y) 像素坐标元组
        """
        pixel_x = int(self.x * frame_width)
        pixel_y = int(self.y * frame_height)
        return (pixel_x, pixel_y)
    
    def is_visible(self, threshold: float = 0.5) -> bool:
        """
        判断关键点是否可见
        
        Args:
            threshold: 可见性阈值
        
        Returns:
            是否可见
        """
        return self.visibility >= threshold


@dataclass
class BoundingBox:
    """
    边界框
    
    Attributes:
        x: 左上角x坐标（像素）
        y: 左上角y坐标（像素）
        width: 宽度（像素）
        height: 高度（像素）
    """
    x: int
    y: int
    width: int
    height: int
    
    def __post_init__(self):
        """验证数据有效性"""
        if self.width <= 0:
            raise ValueError(f"宽度必须大于 0，得到: {self.width}")
        if self.height <= 0:
            raise ValueError(f"高度必须大于 0，得到: {self.height}")
    
    def area(self) -> int:
        """
        计算边界框面积
        
        Returns:
            面积（像素）
        """
        return self.width * self.height
    
    def center(self) -> Tuple[int, int]:
        """
        计算边界框中心点
        
        Returns:
            (x, y) 中心点坐标
        """
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        return (center_x, center_y)
    
    def expand(self, padding_ratio: float = 0.1) -> 'BoundingBox':
        """
        扩展边界框
        
        Args:
            padding_ratio: 扩展比例（相对于当前尺寸）
        
        Returns:
            扩展后的新边界框
        """
        pad_w = int(self.width * padding_ratio)
        pad_h = int(self.height * padding_ratio)
        
        new_x = max(0, self.x - pad_w)
        new_y = max(0, self.y - pad_h)
        new_width = self.width + 2 * pad_w
        new_height = self.height + 2 * pad_h
        
        return BoundingBox(new_x, new_y, new_width, new_height)
    
    def clip(self, frame_width: int, frame_height: int) -> 'BoundingBox':
        """
        裁剪边界框到帧范围内
        
        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
        
        Returns:
            裁剪后的边界框
        """
        x = max(0, min(self.x, frame_width - 1))
        y = max(0, min(self.y, frame_height - 1))
        
        # 确保边界框不超出帧范围
        max_width = frame_width - x
        max_height = frame_height - y
        
        width = min(self.width, max_width)
        height = min(self.height, max_height)
        
        return BoundingBox(x, y, width, height)
    
    def iou(self, other: 'BoundingBox') -> float:
        """
        计算与另一个边界框的 IoU (Intersection over Union)
        
        Args:
            other: 另一个边界框
        
        Returns:
            IoU 值 [0, 1]
        """
        # 计算交集
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class PersonDetection:
    """
    人体检测结果
    
    Attributes:
        person_id: 跟踪ID
        bounding_box: 边界框
        confidence: 检测置信度 [0, 1]
        class_id: 类别ID（0=person）
    """
    person_id: int
    bounding_box: BoundingBox
    confidence: float
    class_id: int = 0  # 默认为 person 类别
    
    def __post_init__(self):
        """验证数据有效性"""
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"置信度必须在 [0, 1] 范围内，得到: {self.confidence}")
        if self.person_id < 0:
            raise ValueError(f"person_id 必须非负，得到: {self.person_id}")


@dataclass
class PoseEstimation:
    """
    姿态估计结果
    
    Attributes:
        landmarks: 33个关键点
        world_landmarks: 3D世界坐标（可选）
        confidence: 整体置信度
    """
    landmarks: List[Landmark]
    confidence: float
    world_landmarks: Optional[List[Landmark]] = None
    
    def __post_init__(self):
        """验证数据有效性"""
        if len(self.landmarks) != 33:
            raise ValueError(f"必须有 33 个关键点，得到: {len(self.landmarks)}")
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"置信度必须在 [0, 1] 范围内，得到: {self.confidence}")
        if self.world_landmarks is not None and len(self.world_landmarks) != 33:
            raise ValueError(f"world_landmarks 必须有 33 个点，得到: {len(self.world_landmarks)}")
    
    def get_visible_landmarks(self, threshold: float = 0.5) -> List[Tuple[int, Landmark]]:
        """
        获取可见的关键点
        
        Args:
            threshold: 可见性阈值
        
        Returns:
            (索引, 关键点) 元组列表
        """
        return [
            (i, lm) for i, lm in enumerate(self.landmarks)
            if lm.is_visible(threshold)
        ]
    
    def get_landmark(self, index: int) -> Landmark:
        """
        获取指定索引的关键点
        
        Args:
            index: 关键点索引 [0, 32]
        
        Returns:
            关键点
        """
        if not (0 <= index < 33):
            raise IndexError(f"关键点索引必须在 [0, 32] 范围内，得到: {index}")
        return self.landmarks[index]


@dataclass
class PoseClassification:
    """
    姿态分类结果
    
    Attributes:
        pose_state: 姿态状态
        confidence: 分类置信度 [0, 1]
        features: 用于分类的特征值（调试用）
    """
    pose_state: PoseState
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据有效性"""
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"置信度必须在 [0, 1] 范围内，得到: {self.confidence}")


@dataclass
class DetectionResult:
    """
    完整的检测结果（包含检测、姿态估计和分类）
    
    Attributes:
        person_id: 人员唯一标识
        bounding_box: 边界框
        confidence: 检测置信度
        landmarks: 33个关键点（可选）
        pose_classification: 姿态分类结果（可选）
        timestamp: 时间戳
    """
    person_id: int
    bounding_box: BoundingBox
    confidence: float
    landmarks: Optional[List[Landmark]] = None
    pose_classification: Optional[PoseClassification] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        """验证数据有效性"""
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"置信度必须在 [0, 1] 范围内，得到: {self.confidence}")
        if self.person_id < 0:
            raise ValueError(f"person_id 必须非负，得到: {self.person_id}")
        if self.landmarks is not None and len(self.landmarks) != 33:
            raise ValueError(f"landmarks 必须有 33 个点，得到: {len(self.landmarks)}")
    
    def has_pose(self) -> bool:
        """检查是否包含姿态信息"""
        return self.landmarks is not None and self.pose_classification is not None
    
    def to_dict(self) -> Dict:
        """
        转换为字典格式（用于 JSON 序列化）
        
        Returns:
            字典表示
        """
        result = {
            "person_id": self.person_id,
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
        
        if self.landmarks:
            result["landmarks"] = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
                for lm in self.landmarks
            ]
        
        if self.pose_classification:
            result["pose_state"] = self.pose_classification.pose_state.value
            result["pose_confidence"] = self.pose_classification.confidence
            result["features"] = self.pose_classification.features
        
        return result

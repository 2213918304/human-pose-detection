"""
姿态分类模块

基于关键点几何特征分类人体姿态。
"""

import numpy as np
import math
from typing import List, Dict, Tuple
from src.models import Landmark, PoseState, PoseClassification
from src.pose.pose_estimator import PoseLandmark
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_angle(p1: Landmark, p2: Landmark, p3: Landmark) -> float:
    """
    计算三点形成的角度（p2 为顶点）
    
    Args:
        p1: 第一个点
        p2: 顶点
        p3: 第三个点
    
    Returns:
        角度（度数）[0, 180]
    """
    # 计算向量
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y])
    
    # 计算向量长度
    len1 = np.linalg.norm(vector1)
    len2 = np.linalg.norm(vector2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # 计算夹角（使用点积）
    cos_angle = np.dot(vector1, vector2) / (len1 * len2)
    
    # 限制在 [-1, 1] 范围内（避免浮点误差）
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 转换为角度
    angle = math.acos(cos_angle)
    angle_degrees = math.degrees(angle)
    
    return angle_degrees


def midpoint(p1: Landmark, p2: Landmark) -> Landmark:
    """
    计算两点的中点
    
    Args:
        p1: 第一个点
        p2: 第二个点
    
    Returns:
        中点
    """
    return Landmark(
        x=(p1.x + p2.x) / 2,
        y=(p1.y + p2.y) / 2,
        z=(p1.z + p2.z) / 2,
        visibility=min(p1.visibility, p2.visibility)
    )


def calculate_torso_angle(landmarks: List[Landmark]) -> float:
    """
    计算躯干与垂直方向的夹角
    
    Args:
        landmarks: 关键点列表（33个）
    
    Returns:
        躯干角度（度数）[0, 90]
    """
    # 肩膀中点
    shoulder_mid = midpoint(
        landmarks[PoseLandmark.LEFT_SHOULDER],
        landmarks[PoseLandmark.RIGHT_SHOULDER]
    )
    
    # 髋部中点
    hip_mid = midpoint(
        landmarks[PoseLandmark.LEFT_HIP],
        landmarks[PoseLandmark.RIGHT_HIP]
    )
    
    # 计算躯干向量（从肩膀指向髋部）
    torso_vector = np.array([hip_mid.x - shoulder_mid.x, hip_mid.y - shoulder_mid.y])
    
    # 垂直向量（向下）
    vertical_vector = np.array([0, 1])
    
    # 计算夹角
    len_torso = np.linalg.norm(torso_vector)
    if len_torso == 0:
        return 0.0
    
    cos_angle = np.dot(torso_vector, vertical_vector) / len_torso
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = math.acos(cos_angle)
    angle_degrees = math.degrees(angle)
    
    # 返回与垂直方向的偏离角度（0度表示完全垂直，90度表示完全水平）
    return angle_degrees


def calculate_knee_angle(landmarks: List[Landmark], side: str) -> float:
    """
    计算膝盖角度（髋-膝-踝）
    
    Args:
        landmarks: 关键点列表
        side: 'left' 或 'right'
    
    Returns:
        膝盖角度（度数）[0, 180]
    """
    if side == 'left':
        hip = landmarks[PoseLandmark.LEFT_HIP]
        knee = landmarks[PoseLandmark.LEFT_KNEE]
        ankle = landmarks[PoseLandmark.LEFT_ANKLE]
    else:
        hip = landmarks[PoseLandmark.RIGHT_HIP]
        knee = landmarks[PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
    
    return calculate_angle(hip, knee, ankle)


def get_hip_height_ratio(landmarks: List[Landmark]) -> float:
    """
    获取髋部相对于画面的高度比例
    
    Args:
        landmarks: 关键点列表
    
    Returns:
        高度比例 [0, 1]，0 表示在顶部，1 表示在底部
    """
    hip_mid = midpoint(
        landmarks[PoseLandmark.LEFT_HIP],
        landmarks[PoseLandmark.RIGHT_HIP]
    )
    
    # y 坐标归一化，0 在顶部，1 在底部
    # 返回相对高度（1 - y 表示从底部算起的高度）
    return 1.0 - hip_mid.y


def calculate_body_aspect_ratio(landmarks: List[Landmark]) -> float:
    """
    计算身体宽高比
    
    Args:
        landmarks: 关键点列表
    
    Returns:
        宽高比
    """
    # 肩膀宽度
    shoulder_width = abs(
        landmarks[PoseLandmark.LEFT_SHOULDER].x -
        landmarks[PoseLandmark.RIGHT_SHOULDER].x
    )
    
    # 身体高度（肩膀到髋部）
    shoulder_mid = midpoint(
        landmarks[PoseLandmark.LEFT_SHOULDER],
        landmarks[PoseLandmark.RIGHT_SHOULDER]
    )
    hip_mid = midpoint(
        landmarks[PoseLandmark.LEFT_HIP],
        landmarks[PoseLandmark.RIGHT_HIP]
    )
    
    body_height = abs(shoulder_mid.y - hip_mid.y)
    
    if body_height == 0:
        return 0.0
    
    return shoulder_width / body_height


def calculate_shoulder_hip_distance(landmarks: List[Landmark]) -> float:
    """
    计算肩髋距离
    
    Args:
        landmarks: 关键点列表
    
    Returns:
        欧氏距离
    """
    shoulder_mid = midpoint(
        landmarks[PoseLandmark.LEFT_SHOULDER],
        landmarks[PoseLandmark.RIGHT_SHOULDER]
    )
    hip_mid = midpoint(
        landmarks[PoseLandmark.LEFT_HIP],
        landmarks[PoseLandmark.RIGHT_HIP]
    )
    
    distance = math.sqrt(
        (shoulder_mid.x - hip_mid.x) ** 2 +
        (shoulder_mid.y - hip_mid.y) ** 2
    )
    
    return distance


class PoseClassifier:
    """
    姿态分类器
    
    基于关键点几何特征分类人体姿态。
    
    Attributes:
        confidence_threshold: 分类置信度阈值
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        初始化姿态分类器
        
        Args:
            confidence_threshold: 分类置信度阈值 [0, 1]
        
        Raises:
            ValueError: 如果阈值无效
        """
        if not (0 <= confidence_threshold <= 1):
            raise ValueError(f"置信度阈值必须在 [0, 1] 范围内，得到: {confidence_threshold}")
        
        self.confidence_threshold = confidence_threshold
        self._classification_count = 0
        
        logger.info(f"姿态分类器初始化，置信度阈值: {confidence_threshold}")
    
    def classify(self, landmarks: List[Landmark]) -> PoseClassification:
        """
        分类姿态
        
        Args:
            landmarks: 关键点列表（33个）
        
        Returns:
            PoseClassification 对象
        """
        if len(landmarks) != 33:
            raise ValueError(f"需要 33 个关键点，得到: {len(landmarks)}")
        
        # 检查关键点可见性
        if not self._check_visibility(landmarks):
            logger.debug("关键点可见性过低")
            return PoseClassification(
                pose_state=PoseState.UNKNOWN,
                confidence=0.0,
                features={}
            )
        
        # 提取特征
        features = self.extract_features(landmarks)
        
        # 分类决策
        pose_state, confidence = self._classify_by_rules(features)
        
        self._classification_count += 1
        
        logger.debug(f"分类结果: {pose_state.value}, 置信度: {confidence:.2f}")
        
        return PoseClassification(
            pose_state=pose_state,
            confidence=confidence,
            features=features
        )
    
    def extract_features(self, landmarks: List[Landmark]) -> Dict[str, float]:
        """
        提取姿态特征（增强版）
        
        Args:
            landmarks: 关键点列表
        
        Returns:
            特征字典
        """
        features = {
            'torso_angle': calculate_torso_angle(landmarks),
            'knee_angle_left': calculate_knee_angle(landmarks, 'left'),
            'knee_angle_right': calculate_knee_angle(landmarks, 'right'),
            'hip_height_ratio': get_hip_height_ratio(landmarks),
            'body_aspect_ratio': calculate_body_aspect_ratio(landmarks),
            'shoulder_hip_distance': calculate_shoulder_hip_distance(landmarks),
        }
        
        # 平均膝盖角度
        features['knee_angle_avg'] = (
            features['knee_angle_left'] + features['knee_angle_right']
        ) / 2
        
        # 新增特征：脚踝高度（用于区分站立和坐立）
        ankle_mid = midpoint(
            landmarks[PoseLandmark.LEFT_ANKLE],
            landmarks[PoseLandmark.RIGHT_ANKLE]
        )
        features['ankle_height_ratio'] = 1.0 - ankle_mid.y
        
        # 新增特征：肩膀高度
        shoulder_mid = midpoint(
            landmarks[PoseLandmark.LEFT_SHOULDER],
            landmarks[PoseLandmark.RIGHT_SHOULDER]
        )
        features['shoulder_height_ratio'] = 1.0 - shoulder_mid.y
        
        # 新增特征：髋膝距离（用于区分蹲和坐）
        hip_mid = midpoint(
            landmarks[PoseLandmark.LEFT_HIP],
            landmarks[PoseLandmark.RIGHT_HIP]
        )
        knee_mid = midpoint(
            landmarks[PoseLandmark.LEFT_KNEE],
            landmarks[PoseLandmark.RIGHT_KNEE]
        )
        features['hip_knee_distance'] = abs(hip_mid.y - knee_mid.y)
        
        return features
    
    def _check_visibility(self, landmarks: List[Landmark], threshold: float = 0.5) -> bool:
        """
        检查关键关键点的可见性
        
        Args:
            landmarks: 关键点列表
            threshold: 可见性阈值
        
        Returns:
            是否可见
        """
        # 检查关键点：肩膀、髋部、膝盖、脚踝
        key_indices = [
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_ANKLE,
        ]
        
        visible_count = sum(
            1 for idx in key_indices
            if landmarks[idx].visibility >= threshold
        )
        
        # 至少 6 个关键点可见
        return visible_count >= 6
    
    def _classify_by_rules(self, features: Dict[str, float]) -> Tuple[PoseState, float]:
        """
        基于规则的分类决策树（优化版）
        
        Args:
            features: 特征字典
        
        Returns:
            (姿态状态, 置信度) 元组
        """
        torso_angle = features['torso_angle']
        knee_angle = features['knee_angle_avg']
        hip_height = features['hip_height_ratio']
        body_aspect_ratio = features['body_aspect_ratio']
        
        # 1. 躺下 (Lying Down) - 躯干接近水平，髋部很低
        # 优化：增加宽高比判断，躺下时身体更"扁平"
        if torso_angle > 60 and hip_height < 0.4:
            confidence = self._calculate_confidence([
                (torso_angle > 60, 0.5),
                (hip_height < 0.4, 0.3),
                (body_aspect_ratio > 1.5, 0.2),  # 躺下时宽高比更大
            ])
            return PoseState.LYING_DOWN, confidence
        
        # 2. 蹲下 (Squatting) - 膝盖弯曲很大，髋部较低但不是最低
        # 优化：调整膝盖角度范围和髋部高度
        if knee_angle < 100 and 0.15 < hip_height < 0.45 and torso_angle < 45:
            confidence = self._calculate_confidence([
                (knee_angle < 100, 0.5),
                (0.15 < hip_height < 0.45, 0.3),
                (torso_angle < 45, 0.2),  # 蹲下时躯干相对直立
            ])
            return PoseState.SQUATTING, confidence
        
        # 3. 坐立 (Sitting) - 膝盖中等弯曲，髋部中等高度
        # 优化：更宽松的膝盖角度范围，更精确的髋部高度
        if 80 < knee_angle < 130 and 0.25 < hip_height < 0.55 and torso_angle < 50:
            confidence = self._calculate_confidence([
                (80 < knee_angle < 130, 0.5),
                (0.25 < hip_height < 0.55, 0.3),
                (torso_angle < 50, 0.2),
            ])
            return PoseState.SITTING, confidence
        
        # 4. 弯腰 (Bending) - 躯干前倾，腿相对直
        # 优化：降低躯干角度阈值，增加膝盖角度要求
        if torso_angle > 35 and knee_angle > 150 and hip_height > 0.3:
            confidence = self._calculate_confidence([
                (torso_angle > 35, 0.5),
                (knee_angle > 150, 0.3),
                (hip_height > 0.3, 0.2),
            ])
            return PoseState.BENDING, confidence
        
        # 5. 站立 (Standing) - 躯干直立，腿伸直，髋部较高
        # 优化：更严格的条件
        if torso_angle < 25 and knee_angle > 155 and hip_height > 0.45:
            confidence = self._calculate_confidence([
                (torso_angle < 25, 0.4),
                (knee_angle > 155, 0.3),
                (hip_height > 0.45, 0.3),
            ])
            return PoseState.STANDING, confidence
        
        # 6. 额外的躺下判断（侧躺或其他姿势）
        if hip_height < 0.25:
            confidence = self._calculate_confidence([
                (hip_height < 0.25, 0.6),
                (torso_angle > 45, 0.4),
            ])
            return PoseState.LYING_DOWN, confidence
        
        # 7. 额外的坐立判断（更宽松的条件）
        if knee_angle < 140 and 0.2 < hip_height < 0.6 and torso_angle < 60:
            confidence = self._calculate_confidence([
                (knee_angle < 140, 0.4),
                (0.2 < hip_height < 0.6, 0.4),
                (torso_angle < 60, 0.2),
            ])
            return PoseState.SITTING, confidence
        
        # 8. 未知 (Unknown)
        return PoseState.UNKNOWN, 0.0
    
    def _calculate_confidence(self, conditions: List[Tuple[bool, float]]) -> float:
        """
        计算分类置信度
        
        Args:
            conditions: (条件是否满足, 权重) 列表
        
        Returns:
            置信度 [0, 1]
        """
        total_weight = sum(weight for _, weight in conditions)
        satisfied_weight = sum(weight for satisfied, weight in conditions if satisfied)
        
        if total_weight == 0:
            return 0.0
        
        return satisfied_weight / total_weight
    
    def get_classification_count(self) -> int:
        """
        获取分类计数
        
        Returns:
            累计分类次数
        """
        return self._classification_count
    
    def reset_count(self) -> None:
        """重置分类计数"""
        self._classification_count = 0
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        设置置信度阈值
        
        Args:
            threshold: 新的置信度阈值 [0, 1]
        
        Raises:
            ValueError: 如果阈值无效
        """
        if not (0 <= threshold <= 1):
            raise ValueError(f"置信度阈值必须在 [0, 1] 范围内，得到: {threshold}")
        
        self.confidence_threshold = threshold
        logger.info(f"置信度阈值已更新为: {threshold}")
    
    def get_info(self) -> dict:
        """
        获取分类器信息
        
        Returns:
            包含分类器信息的字典
        """
        return {
            "confidence_threshold": self.confidence_threshold,
            "classification_count": self._classification_count,
        }

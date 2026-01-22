"""
MediaPipe 姿态估计模块

使用 MediaPipe Pose 在检测区域内估计人体关键点。
"""

import numpy as np
import cv2
from typing import Optional, List
import mediapipe as mp
from src.models import BoundingBox, Landmark, PoseEstimation
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PoseEstimator:
    """
    MediaPipe 姿态估计器
    
    在 YOLOv8 检测的边界框区域内使用 MediaPipe 估计 33 个关键点。
    
    Attributes:
        mp_pose: MediaPipe Pose 对象
        min_detection_confidence: 检测置信度阈值
        min_tracking_confidence: 跟踪置信度阈值
        model_complexity: 模型复杂度 (0, 1, 2)
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        初始化姿态估计器
        
        Args:
            min_detection_confidence: 检测置信度阈值 [0, 1]
            min_tracking_confidence: 跟踪置信度阈值 [0, 1]
            model_complexity: 模型复杂度 (0, 1, 2)
        
        Raises:
            ValueError: 如果参数无效
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._estimation_count = 0
        
        # 验证参数
        if not (0 <= min_detection_confidence <= 1):
            raise ValueError(f"检测置信度必须在 [0, 1] 范围内，得到: {min_detection_confidence}")
        if not (0 <= min_tracking_confidence <= 1):
            raise ValueError(f"跟踪置信度必须在 [0, 1] 范围内，得到: {min_tracking_confidence}")
        if model_complexity not in [0, 1, 2]:
            raise ValueError(f"模型复杂度必须是 0, 1 或 2，得到: {model_complexity}")
        
        # 初始化 MediaPipe Pose
        self._init_mediapipe()
    
    def _init_mediapipe(self) -> None:
        """初始化 MediaPipe Pose"""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,  # 视频模式
                model_complexity=self.model_complexity,
                smooth_landmarks=True,  # 平滑关键点
                enable_segmentation=False,  # 不需要分割
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            logger.info("MediaPipe Pose 初始化成功")
            logger.info(f"模型复杂度: {self.model_complexity}")
            logger.info(f"检测置信度: {self.min_detection_confidence}")
            logger.info(f"跟踪置信度: {self.min_tracking_confidence}")
            
        except Exception as e:
            logger.error(f"初始化 MediaPipe Pose 失败: {e}")
            raise
    
    def estimate(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        padding_ratio: float = 0.1
    ) -> Optional[PoseEstimation]:
        """
        在边界框区域内估计姿态关键点
        
        Args:
            frame: 输入帧（BGR 格式）
            bbox: 人体边界框
            padding_ratio: 边界框扩展比例
        
        Returns:
            PoseEstimation 对象，失败时返回 None
        """
        if frame is None or frame.size == 0:
            logger.warning("输入帧为空")
            return None
        
        try:
            # 扩展边界框并裁剪到帧范围内
            frame_height, frame_width = frame.shape[:2]
            expanded_bbox = bbox.expand(padding_ratio).clip(frame_width, frame_height)
            
            # 裁剪区域
            x, y, w, h = expanded_bbox.x, expanded_bbox.y, expanded_bbox.width, expanded_bbox.height
            cropped = frame[y:y+h, x:x+w]
            
            if cropped.size == 0:
                logger.warning("裁剪区域为空")
                return None
            
            # 转换为 RGB（MediaPipe 需要 RGB）
            rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
            # 运行姿态估计
            results = self.pose.process(rgb_cropped)
            
            if results.pose_landmarks is None:
                logger.debug("未检测到姿态关键点")
                return None
            
            # 转换关键点坐标（从裁剪区域到原始帧）
            landmarks = self._convert_landmarks(
                results.pose_landmarks,
                expanded_bbox,
                frame_width,
                frame_height
            )
            
            # 计算整体置信度（使用可见性的平均值）
            confidence = np.mean([lm.visibility for lm in landmarks])
            
            self._estimation_count += 1
            
            return PoseEstimation(
                landmarks=landmarks,
                confidence=float(confidence)
            )
            
        except Exception as e:
            logger.error(f"姿态估计过程中发生错误: {e}")
            return None
    
    def _convert_landmarks(
        self,
        mp_landmarks,
        bbox: BoundingBox,
        frame_width: int,
        frame_height: int
    ) -> List[Landmark]:
        """
        转换 MediaPipe 关键点坐标
        
        将关键点从裁剪区域的归一化坐标转换为原始帧的归一化坐标。
        
        Args:
            mp_landmarks: MediaPipe 关键点对象
            bbox: 裁剪区域的边界框
            frame_width: 原始帧宽度
            frame_height: 原始帧高度
        
        Returns:
            Landmark 列表（33个）
        """
        landmarks = []
        
        for mp_lm in mp_landmarks.landmark:
            # MediaPipe 返回的是相对于裁剪区域的归一化坐标
            # 需要转换为相对于原始帧的归一化坐标
            
            # 先转换为裁剪区域的像素坐标
            crop_x = mp_lm.x * bbox.width
            crop_y = mp_lm.y * bbox.height
            
            # 转换为原始帧的像素坐标
            frame_x = bbox.x + crop_x
            frame_y = bbox.y + crop_y
            
            # 归一化到 [0, 1]
            norm_x = frame_x / frame_width
            norm_y = frame_y / frame_height
            
            # 确保在有效范围内
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            landmark = Landmark(
                x=norm_x,
                y=norm_y,
                z=mp_lm.z,  # 深度信息保持不变
                visibility=mp_lm.visibility
            )
            
            landmarks.append(landmark)
        
        return landmarks
    
    def estimate_batch(
        self,
        frame: np.ndarray,
        bboxes: List[BoundingBox],
        padding_ratio: float = 0.1
    ) -> List[Optional[PoseEstimation]]:
        """
        批量估计多个边界框的姿态
        
        Args:
            frame: 输入帧
            bboxes: 边界框列表
            padding_ratio: 边界框扩展比例
        
        Returns:
            PoseEstimation 列表（可能包含 None）
        """
        results = []
        
        for bbox in bboxes:
            estimation = self.estimate(frame, bbox, padding_ratio)
            results.append(estimation)
        
        return results
    
    def get_estimation_count(self) -> int:
        """
        获取总估计数量
        
        Returns:
            累计估计次数
        """
        return self._estimation_count
    
    def reset_count(self) -> None:
        """重置估计计数"""
        self._estimation_count = 0
    
    def get_info(self) -> dict:
        """
        获取估计器信息
        
        Returns:
            包含估计器信息的字典
        """
        return {
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "model_complexity": self.model_complexity,
            "estimation_count": self._estimation_count,
        }
    
    def close(self) -> None:
        """释放资源"""
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
                self.pose = None
                logger.info("MediaPipe Pose 已释放")
            except Exception as e:
                logger.debug(f"释放 MediaPipe Pose 时出现警告: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
    
    def __del__(self):
        """析构函数"""
        self.close()


# MediaPipe 关键点索引常量
class PoseLandmark:
    """MediaPipe Pose 关键点索引"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

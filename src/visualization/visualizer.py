"""
可视化模块

在视频帧上绘制检测和分类结果。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from src.models import DetectionResult, Landmark, BoundingBox, PoseState
from src.pose.pose_estimator import PoseLandmark
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class VisualizationConfig:
    """可视化配置"""
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # BGR格式 - 绿色
    bbox_thickness: int = 2
    landmark_radius: int = 3
    landmark_color: Tuple[int, int, int] = (0, 0, 255)  # 红色
    skeleton_thickness: int = 2
    skeleton_color: Tuple[int, int, int] = (255, 0, 0)  # 蓝色
    label_font_scale: float = 0.6
    label_thickness: int = 2
    label_bg_color: Tuple[int, int, int] = (0, 0, 0)  # 黑色背景
    label_text_color: Tuple[int, int, int] = (255, 255, 255)  # 白色文字


class Visualizer:
    """
    可视化器
    
    在视频帧上绘制检测结果、关键点、骨架和姿态标签。
    
    Attributes:
        show_landmarks: 是否显示关键点
        show_skeleton: 是否显示骨架
        show_bbox: 是否显示边界框
        show_label: 是否显示标签
        config: 可视化配置
    """
    
    # MediaPipe 骨架连接定义（关键点索引对）
    SKELETON_CONNECTIONS = [
        # 面部
        (0, 1), (1, 2), (2, 3), (3, 7),  # 左眼到左耳
        (0, 4), (4, 5), (5, 6), (6, 8),  # 右眼到右耳
        (9, 10),  # 嘴巴
        
        # 躯干
        (11, 12),  # 肩膀
        (11, 23), (12, 24),  # 肩膀到髋部
        (23, 24),  # 髋部
        
        # 左臂
        (11, 13), (13, 15),  # 肩膀-肘部-手腕
        (15, 17), (15, 19), (15, 21),  # 手腕到手指
        (17, 19),  # 手指连接
        
        # 右臂
        (12, 14), (14, 16),  # 肩膀-肘部-手腕
        (16, 18), (16, 20), (16, 22),  # 手腕到手指
        (18, 20),  # 手指连接
        
        # 左腿
        (23, 25), (25, 27),  # 髋部-膝盖-脚踝
        (27, 29), (27, 31),  # 脚踝到脚
        (29, 31),  # 脚部连接
        
        # 右腿
        (24, 26), (26, 28),  # 髋部-膝盖-脚踝
        (28, 30), (28, 32),  # 脚踝到脚
        (30, 32),  # 脚部连接
    ]
    
    def __init__(
        self,
        show_landmarks: bool = True,
        show_skeleton: bool = True,
        show_bbox: bool = True,
        show_label: bool = True,
        config: Optional[VisualizationConfig] = None
    ):
        """
        初始化可视化器
        
        Args:
            show_landmarks: 是否显示关键点
            show_skeleton: 是否显示骨架
            show_bbox: 是否显示边界框
            show_label: 是否显示标签
            config: 可视化配置（可选）
        """
        self.show_landmarks = show_landmarks
        self.show_skeleton = show_skeleton
        self.show_bbox = show_bbox
        self.show_label = show_label
        self.config = config or VisualizationConfig()
        
        logger.info("可视化器初始化完成")
        logger.info(f"显示设置: 边界框={show_bbox}, 关键点={show_landmarks}, "
                   f"骨架={show_skeleton}, 标签={show_label}")
    
    def draw(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult]
    ) -> np.ndarray:
        """
        在帧上绘制检测结果
        
        Args:
            frame: 输入帧（BGR格式）
            detections: 检测结果列表
        
        Returns:
            绘制后的帧
        """
        if frame is None or frame.size == 0:
            logger.warning("输入帧为空")
            return frame
        
        # 创建副本以避免修改原始帧
        output_frame = frame.copy()
        
        for detection in detections:
            # 绘制边界框
            if self.show_bbox:
                self._draw_bbox(output_frame, detection.bounding_box, detection.person_id)
            
            # 绘制关键点和骨架
            if detection.landmarks is not None:
                if self.show_skeleton:
                    self._draw_skeleton(output_frame, detection.landmarks)
                
                if self.show_landmarks:
                    self._draw_landmarks(output_frame, detection.landmarks)
            
            # 绘制姿态标签
            if self.show_label and detection.pose_classification is not None:
                self._draw_label(
                    output_frame,
                    detection.bounding_box,
                    detection.pose_classification.pose_state,
                    detection.pose_classification.confidence
                )
        
        return output_frame
    
    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        person_id: int
    ) -> None:
        """
        绘制边界框
        
        Args:
            frame: 输入帧
            bbox: 边界框
            person_id: 人员ID
        """
        # 绘制矩形
        cv2.rectangle(
            frame,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            self.config.bbox_color,
            self.config.bbox_thickness
        )
        
        # 绘制ID标签
        id_text = f"ID:{person_id}"
        text_size = cv2.getTextSize(
            id_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )[0]
        
        # ID标签背景
        cv2.rectangle(
            frame,
            (bbox.x, bbox.y - text_size[1] - 4),
            (bbox.x + text_size[0] + 4, bbox.y),
            self.config.bbox_color,
            -1
        )
        
        # ID文字
        cv2.putText(
            frame,
            id_text,
            (bbox.x + 2, bbox.y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # 黑色文字
            1,
            cv2.LINE_AA
        )
    
    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List[Landmark]
    ) -> None:
        """
        绘制关键点
        
        Args:
            frame: 输入帧
            landmarks: 关键点列表
        """
        frame_height, frame_width = frame.shape[:2]
        
        for landmark in landmarks:
            if landmark.visibility > 0.5:  # 只绘制可见的关键点
                x, y = landmark.to_pixel(frame_width, frame_height)
                
                cv2.circle(
                    frame,
                    (x, y),
                    self.config.landmark_radius,
                    self.config.landmark_color,
                    -1
                )
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: List[Landmark]
    ) -> None:
        """
        绘制骨架连接
        
        Args:
            frame: 输入帧
            landmarks: 关键点列表
        """
        frame_height, frame_width = frame.shape[:2]
        
        for connection in self.SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            
            # 检查关键点是否可见
            if (landmarks[start_idx].visibility > 0.5 and
                landmarks[end_idx].visibility > 0.5):
                
                start_point = landmarks[start_idx].to_pixel(frame_width, frame_height)
                end_point = landmarks[end_idx].to_pixel(frame_width, frame_height)
                
                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    self.config.skeleton_color,
                    self.config.skeleton_thickness,
                    cv2.LINE_AA
                )
    
    def _draw_label(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        pose_state: PoseState,
        confidence: float
    ) -> None:
        """
        绘制姿态标签（支持中文）
        
        Args:
            frame: 输入帧
            bbox: 边界框
            pose_state: 姿态状态
            confidence: 置信度
        """
        # 创建标签文本
        label_text = f"{pose_state.value} ({confidence:.2f})"
        
        # 转换为PIL图像以支持中文
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            # Windows系统字体路径
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        except:
            try:
                # 尝试其他常见字体
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
        
        # 获取文本边界框
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 标签位置（边界框下方）
        label_x = bbox.x
        label_y = bbox.y + bbox.height + 10
        
        # 确保标签不超出帧范围
        frame_height = frame.shape[0]
        if label_y + text_height > frame_height:
            label_y = bbox.y - text_height - 10  # 如果下方空间不足，放在上方
        
        # 绘制标签背景
        padding = 4
        draw.rectangle(
            [label_x - padding, label_y - padding,
             label_x + text_width + padding, label_y + text_height + padding],
            fill=(0, 0, 0)
        )
        
        # 绘制标签文字
        draw.text(
            (label_x, label_y),
            label_text,
            font=font,
            fill=(255, 255, 255)
        )
        
        # 转换回OpenCV格式
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        frame[:] = frame_cv
    
    def set_show_landmarks(self, show: bool) -> None:
        """设置是否显示关键点"""
        self.show_landmarks = show
        logger.info(f"关键点显示: {show}")
    
    def set_show_skeleton(self, show: bool) -> None:
        """设置是否显示骨架"""
        self.show_skeleton = show
        logger.info(f"骨架显示: {show}")
    
    def set_show_bbox(self, show: bool) -> None:
        """设置是否显示边界框"""
        self.show_bbox = show
        logger.info(f"边界框显示: {show}")
    
    def set_show_label(self, show: bool) -> None:
        """设置是否显示标签"""
        self.show_label = show
        logger.info(f"标签显示: {show}")
    
    def get_config(self) -> VisualizationConfig:
        """获取可视化配置"""
        return self.config
    
    def set_config(self, config: VisualizationConfig) -> None:
        """设置可视化配置"""
        self.config = config
        logger.info("可视化配置已更新")

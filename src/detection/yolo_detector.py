"""
YOLOv8 人体检测模块

使用 YOLOv8 检测视频帧中的所有人体。
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
from ultralytics import YOLO
from src.models import BoundingBox, PersonDetection
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class YOLOv8Detector:
    """
    YOLOv8 人体检测器
    
    使用 YOLOv8 模型检测视频帧中的人体，返回边界框和置信度。
    
    Attributes:
        model: YOLOv8 模型
        confidence_threshold: 检测置信度阈值
        device: 运行设备（'cpu' 或 'cuda'）
    """
    
    # COCO 数据集中 person 类别的 ID
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.32,
        device: str = "cpu"
    ):
        """
        初始化 YOLOv8 检测器
        
        Args:
            model_path: YOLOv8 模型路径（默认使用 nano 版本）
            confidence_threshold: 检测置信度阈值 [0, 1]
            device: 运行设备（'cpu' 或 'cuda'）
        
        Raises:
            ValueError: 如果参数无效
            FileNotFoundError: 如果模型文件不存在且无法下载
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._detection_count = 0
        
        # 验证参数
        if not (0 <= confidence_threshold <= 1):
            raise ValueError(f"置信度阈值必须在 [0, 1] 范围内，得到: {confidence_threshold}")
        
        if device not in ["cpu", "cuda"]:
            logger.warning(f"未知设备类型: {device}，使用 'cpu'")
            self.device = "cpu"
        
        # 加载模型
        self._load_model()
    
    def _load_model(self) -> None:
        """加载 YOLOv8 模型"""
        try:
            logger.info(f"加载 YOLOv8 模型: {self.model_path}")
            
            # 如果是预训练模型名称（如 yolov8n.pt），YOLO 会自动下载
            self.model = YOLO(self.model_path)
            
            # 设置设备
            self.model.to(self.device)
            
            logger.info(f"模型加载成功，运行在: {self.device}")
            logger.info(f"置信度阈值: {self.confidence_threshold}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise FileNotFoundError(
                f"无法加载 YOLOv8 模型: {self.model_path}。"
                f"请确保模型文件存在或网络连接正常以下载预训练模型。"
            ) from e
    
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        检测帧中的所有人体
        
        Args:
            frame: 输入帧（BGR 格式）
        
        Returns:
            PersonDetection 列表，按置信度降序排列
        """
        if frame is None or frame.size == 0:
            logger.warning("输入帧为空")
            return []
        
        try:
            # 运行检测
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=[self.PERSON_CLASS_ID],
                verbose=False
            )
            
            # 解析结果
            detections = self._parse_results(results[0])
            
            self._detection_count += len(detections)
            
            if detections:
                logger.debug(f"检测到 {len(detections)} 个人体")
            
            return detections
            
        except Exception as e:
            logger.error(f"检测过程中发生错误: {e}")
            return []
    
    def _parse_results(self, result) -> List[PersonDetection]:
        """
        解析 YOLO 检测结果
        
        Args:
            result: YOLO 检测结果对象
        
        Returns:
            PersonDetection 列表
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # 获取边界框、置信度和类别
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        
        # 如果有跟踪 ID，使用它；否则使用索引
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = list(range(len(boxes)))
        
        # 创建 PersonDetection 对象
        for i, (box, conf, track_id) in enumerate(zip(boxes, confidences, track_ids)):
            x1, y1, x2, y2 = box
            
            # 转换为 BoundingBox 格式
            bbox = BoundingBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1)
            )
            
            detection = PersonDetection(
                person_id=int(track_id),
                bounding_box=bbox,
                confidence=float(conf),
                class_id=self.PERSON_CLASS_ID
            )
            
            detections.append(detection)
        
        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[PersonDetection]]:
        """
        批量检测多帧
        
        Args:
            frames: 帧列表
        
        Returns:
            每帧的 PersonDetection 列表
        """
        if not frames:
            return []
        
        try:
            # 批量推理
            results = self.model(
                frames,
                conf=self.confidence_threshold,
                classes=[self.PERSON_CLASS_ID],
                verbose=False
            )
            
            # 解析每帧的结果
            all_detections = []
            for result in results:
                detections = self._parse_results(result)
                all_detections.append(detections)
                self._detection_count += len(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"批量检测过程中发生错误: {e}")
            return [[] for _ in frames]
    
    def get_detection_count(self) -> int:
        """
        获取总检测数量
        
        Returns:
            累计检测到的人体数量
        """
        return self._detection_count
    
    def reset_count(self) -> None:
        """重置检测计数"""
        self._detection_count = 0
    
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
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "detection_count": self._detection_count,
        }
    
    def close(self) -> None:
        """释放模型资源"""
        if hasattr(self, 'model'):
            del self.model
            logger.info("YOLOv8 模型已释放")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
    
    def __del__(self):
        """析构函数"""
        self.close()


class YOLOv8TrackerDetector(YOLOv8Detector):
    """
    带跟踪功能的 YOLOv8 检测器
    
    使用 YOLOv8 的内置跟踪功能，为每个人分配持久的 ID。
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.32,
        device: str = "cpu",
        tracker: str = "bytetrack.yaml"
    ):
        """
        初始化带跟踪的检测器
        
        Args:
            model_path: YOLOv8 模型路径
            confidence_threshold: 检测置信度阈值
            device: 运行设备
            tracker: 跟踪器配置文件
        """
        super().__init__(model_path, confidence_threshold, device)
        self.tracker = tracker
        logger.info(f"启用跟踪功能: {tracker}")
    
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        检测并跟踪帧中的所有人体
        
        Args:
            frame: 输入帧（BGR 格式）
        
        Returns:
            PersonDetection 列表，包含持久的跟踪 ID
        """
        if frame is None or frame.size == 0:
            logger.warning("输入帧为空")
            return []
        
        try:
            # 运行检测和跟踪
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                classes=[self.PERSON_CLASS_ID],
                tracker=self.tracker,
                persist=True,  # 持久化跟踪
                verbose=False
            )
            
            # 解析结果
            detections = self._parse_results(results[0])
            
            self._detection_count += len(detections)
            
            if detections:
                logger.debug(f"检测并跟踪到 {len(detections)} 个人体")
            
            return detections
            
        except Exception as e:
            logger.error(f"检测和跟踪过程中发生错误: {e}")
            return []

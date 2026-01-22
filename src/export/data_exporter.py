"""
数据导出器

将检测结果导出为 JSON 格式。
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models import DetectionResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataExporter:
    """
    数据导出器
    
    收集帧数据并导出为 JSON 格式。
    
    Attributes:
        output_path: 输出文件路径
        metadata: 元数据（视频源、帧率等）
        frames_data: 帧数据列表
    """
    
    def __init__(self, output_path: str):
        """
        初始化数据导出器
        
        Args:
            output_path: 输出文件路径
        """
        self.output_path = output_path
        self.metadata: Dict[str, Any] = {
            "video_source": "",
            "total_frames": 0,
            "fps": 0.0
        }
        self.frames_data: List[Dict[str, Any]] = []
        
        logger.info(f"数据导出器初始化完成，输出路径: {output_path}")
    
    def set_metadata(
        self,
        video_source: str = "",
        fps: float = 0.0
    ) -> None:
        """
        设置元数据
        
        Args:
            video_source: 视频源
            fps: 帧率
        """
        self.metadata["video_source"] = video_source
        self.metadata["fps"] = fps
        logger.info(f"元数据已设置: 视频源={video_source}, FPS={fps}")
    
    def add_frame_data(
        self,
        frame_number: int,
        timestamp: float,
        detections: List[DetectionResult]
    ) -> None:
        """
        添加帧数据
        
        Args:
            frame_number: 帧号
            timestamp: 时间戳
            detections: 检测结果列表
        """
        frame_data = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "detections": [self._detection_to_dict(det) for det in detections]
        }
        
        self.frames_data.append(frame_data)
        logger.debug(f"添加帧数据: 帧号={frame_number}, 检测数={len(detections)}")
    
    def _detection_to_dict(self, detection: DetectionResult) -> Dict[str, Any]:
        """
        将检测结果转换为字典
        
        Args:
            detection: 检测结果
        
        Returns:
            字典表示
        """
        result = {
            "person_id": detection.person_id,
            "bounding_box": {
                "x": detection.bounding_box.x,
                "y": detection.bounding_box.y,
                "width": detection.bounding_box.width,
                "height": detection.bounding_box.height
            },
            "confidence": detection.confidence
        }
        
        # 添加关键点（如果存在）
        if detection.landmarks is not None:
            result["landmarks"] = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in detection.landmarks
            ]
        
        # 添加姿态分类（如果存在）
        if detection.pose_classification is not None:
            result["pose_state"] = detection.pose_classification.pose_state.value
            result["pose_confidence"] = detection.pose_classification.confidence
            result["features"] = detection.pose_classification.features
        
        return result
    
    def export(self) -> None:
        """
        导出数据到 JSON 文件
        
        Raises:
            IOError: 如果文件写入失败
        """
        # 更新总帧数
        self.metadata["total_frames"] = len(self.frames_data)
        
        # 构建完整数据
        export_data = {
            "metadata": self.metadata,
            "frames": self.frames_data
        }
        
        try:
            # 确保目录存在
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"数据已导出到 {self.output_path}，共 {len(self.frames_data)} 帧")
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            raise IOError(f"导出数据失败: {e}")
    
    def clear(self) -> None:
        """清空所有数据"""
        self.frames_data.clear()
        self.metadata["total_frames"] = 0
        logger.info("数据已清空")
    
    def get_frame_count(self) -> int:
        """
        获取已收集的帧数
        
        Returns:
            帧数
        """
        return len(self.frames_data)
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取导出器信息
        
        Returns:
            包含输出路径、帧数和元数据的字典
        """
        return {
            "output_path": self.output_path,
            "frame_count": len(self.frames_data),
            "metadata": self.metadata
        }
    
    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """
        从文件加载导出的数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            加载的数据字典
        
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件格式无效
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"从 {file_path} 加载数据成功")
            return data
        except FileNotFoundError:
            logger.error(f"文件 {file_path} 不存在")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"文件格式错误: {e}")
            raise ValueError(f"文件格式错误: {e}")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

"""
配置管理器

管理系统配置参数，支持从 JSON 文件加载和保存配置。
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Union, Dict
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SystemConfig:
    """
    系统配置
    
    包含所有系统运行所需的配置参数。
    
    Attributes:
        video_source: 视频源（0表示默认摄像头，或视频文件路径）
        target_fps: 目标帧率
        min_detection_confidence: 最小检测置信度
        min_tracking_confidence: 最小跟踪置信度
        classification_confidence_threshold: 分类置信度阈值
        show_landmarks: 是否显示关键点
        show_skeleton: 是否显示骨架
        show_bbox: 是否显示边界框
        show_label: 是否显示标签
        save_video: 是否保存视频
        output_video_path: 输出视频路径
        export_data: 是否导出数据
        export_data_path: 导出数据路径
    """
    # 视频源配置
    video_source: Union[int, str] = 0  # 0表示默认摄像头
    target_fps: int = 30
    
    # 检测配置
    min_detection_confidence: float = 0.32
    min_tracking_confidence: float = 0.5
    
    # 分类配置
    classification_confidence_threshold: float = 0.6
    
    # 可视化配置
    show_landmarks: bool = True
    show_skeleton: bool = True
    show_bbox: bool = True
    show_label: bool = True
    
    # 输出配置
    save_video: bool = False
    output_video_path: str = "output.mp4"
    export_data: bool = False
    export_data_path: str = "detections.json"
    
    # 性能配置
    device: str = "cuda"  # cpu 或 cuda
    show_display: bool = True  # 是否显示实时画面
    
    def __post_init__(self):
        """验证配置参数"""
        if self.target_fps <= 0:
            raise ValueError(f"target_fps 必须大于 0，得到: {self.target_fps}")
        
        if not (0 <= self.min_detection_confidence <= 1):
            raise ValueError(
                f"min_detection_confidence 必须在 [0, 1] 范围内，得到: {self.min_detection_confidence}"
            )
        
        if not (0 <= self.min_tracking_confidence <= 1):
            raise ValueError(
                f"min_tracking_confidence 必须在 [0, 1] 范围内，得到: {self.min_tracking_confidence}"
            )
        
        if not (0 <= self.classification_confidence_threshold <= 1):
            raise ValueError(
                f"classification_confidence_threshold 必须在 [0, 1] 范围内，"
                f"得到: {self.classification_confidence_threshold}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
        
        Returns:
            SystemConfig 实例
        """
        return cls(**config_dict)


class ConfigManager:
    """
    配置管理器
    
    管理系统配置的加载、保存和访问。
    
    Attributes:
        config: 系统配置对象
        config_path: 配置文件路径
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径（可选）。如果提供，将从文件加载配置。
        """
        self.config = SystemConfig()
        self.config_path = config_path
        
        if config_path is not None:
            try:
                self.load(config_path)
                logger.info(f"从 {config_path} 加载配置成功")
            except FileNotFoundError:
                logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}，使用默认配置")
        else:
            logger.info("使用默认配置")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值（如果键不存在）
        
        Returns:
            配置值
        """
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        
        Raises:
            AttributeError: 如果键不存在
            ValueError: 如果值无效
        """
        if not hasattr(self.config, key):
            raise AttributeError(f"配置键 '{key}' 不存在")
        
        # 创建新配置对象以触发验证
        config_dict = self.config.to_dict()
        config_dict[key] = value
        
        try:
            new_config = SystemConfig.from_dict(config_dict)
            self.config = new_config
            logger.info(f"配置 '{key}' 已更新为: {value}")
        except (ValueError, TypeError) as e:
            logger.error(f"设置配置 '{key}' 失败: {e}")
            raise
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存配置到 JSON 文件
        
        Args:
            path: 保存路径（可选）。如果未提供，使用初始化时的路径。
        
        Raises:
            ValueError: 如果未提供路径且初始化时也未提供
        """
        save_path = path or self.config_path
        
        if save_path is None:
            raise ValueError("必须提供保存路径")
        
        try:
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到 {save_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        从 JSON 文件加载配置
        
        Args:
            path: 配置文件路径
        
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果配置无效
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            self.config = SystemConfig.from_dict(config_dict)
            self.config_path = path
            logger.info(f"从 {path} 加载配置成功")
        except FileNotFoundError:
            logger.error(f"配置文件 {path} 不存在")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """
        获取配置对象
        
        Returns:
            SystemConfig 实例
        """
        return self.config
    
    def reset_to_default(self) -> None:
        """重置为默认配置"""
        self.config = SystemConfig()
        logger.info("配置已重置为默认值")
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典（部分或完整）
        """
        current_dict = self.config.to_dict()
        current_dict.update(config_dict)
        
        try:
            self.config = SystemConfig.from_dict(current_dict)
            logger.info(f"配置已从字典更新")
        except Exception as e:
            logger.error(f"从字典更新配置失败: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取配置管理器信息
        
        Returns:
            包含配置路径和当前配置的字典
        """
        return {
            'config_path': self.config_path,
            'config': self.config.to_dict()
        }

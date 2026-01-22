"""
视频捕获模块

管理视频输入源，提供统一的帧访问接口。
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoCapture:
    """
    视频捕获类
    
    管理视频输入源（摄像头或视频文件），提供统一的帧读取接口。
    
    Attributes:
        source: 视频源（摄像头索引或文件路径）
        cap: OpenCV VideoCapture 对象
    """
    
    def __init__(self, source: Union[int, str] = 0):
        """
        初始化视频捕获
        
        Args:
            source: 视频源
                - int: 摄像头索引（0 表示默认摄像头）
                - str: 视频文件路径
        
        Raises:
            ValueError: 如果视频源无效
            FileNotFoundError: 如果视频文件不存在
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._is_camera = isinstance(source, int)
        
        # 验证文件路径
        if isinstance(source, str):
            if not Path(source).exists():
                raise FileNotFoundError(f"视频文件不存在: {source}")
        
        # 打开视频源
        self._open()
    
    def _open(self) -> None:
        """打开视频源"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频源: {self.source}")
            
            logger.info(f"成功打开视频源: {self.source}")
            
            # 记录视频信息
            if self._is_camera:
                logger.info(f"摄像头索引: {self.source}")
            else:
                logger.info(f"视频文件: {self.source}")
                logger.info(f"总帧数: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            
            logger.info(f"分辨率: {self.get_frame_size()}")
            logger.info(f"帧率: {self.get_fps():.2f} FPS")
            
        except Exception as e:
            logger.error(f"打开视频源失败: {e}")
            raise
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取下一帧
        
        Returns:
            (success, frame) 元组
            - success: 是否成功读取
            - frame: 帧图像（BGR格式），失败时为 None
        """
        if self.cap is None or not self.cap.isOpened():
            logger.warning("视频源未打开或已关闭")
            return False, None
        
        try:
            success, frame = self.cap.read()
            
            if success:
                self._frame_count += 1
                return True, frame
            else:
                if not self._is_camera:
                    logger.info("视频已结束")
                else:
                    logger.warning("读取帧失败")
                return False, None
                
        except Exception as e:
            logger.error(f"读取帧时发生错误: {e}")
            return False, None
    
    def get_fps(self) -> float:
        """
        获取视频帧率
        
        Returns:
            帧率（FPS）
        """
        if self.cap is None or not self.cap.isOpened():
            return 0.0
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 某些摄像头可能返回 0，使用默认值
        if fps == 0:
            fps = 30.0
            logger.warning(f"无法获取帧率，使用默认值: {fps} FPS")
        
        return fps
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        获取帧尺寸
        
        Returns:
            (width, height) 元组
        """
        if self.cap is None or not self.cap.isOpened():
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)
    
    def get_frame_count(self) -> int:
        """
        获取已读取的帧数
        
        Returns:
            已读取的帧数
        """
        return self._frame_count
    
    def get_total_frames(self) -> int:
        """
        获取视频总帧数（仅对视频文件有效）
        
        Returns:
            总帧数，摄像头返回 -1
        """
        if self._is_camera:
            return -1
        
        if self.cap is None or not self.cap.isOpened():
            return 0
        
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_position(self) -> float:
        """
        获取当前播放位置（仅对视频文件有效）
        
        Returns:
            当前位置（0.0 到 1.0），摄像头返回 0.0
        """
        if self._is_camera:
            return 0.0
        
        total = self.get_total_frames()
        if total <= 0:
            return 0.0
        
        return self._frame_count / total
    
    def is_opened(self) -> bool:
        """
        检查视频源是否打开
        
        Returns:
            是否打开
        """
        return self.cap is not None and self.cap.isOpened()
    
    def is_camera(self) -> bool:
        """
        检查是否为摄像头
        
        Returns:
            是否为摄像头
        """
        return self._is_camera
    
    def release(self) -> None:
        """释放视频源"""
        if self.cap is not None:
            self.cap.release()
            logger.info(f"释放视频源: {self.source}, 共读取 {self._frame_count} 帧")
            self.cap = None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()
    
    def __del__(self):
        """析构函数"""
        self.release()
    
    def __iter__(self):
        """迭代器接口"""
        return self
    
    def __next__(self) -> np.ndarray:
        """
        迭代器下一帧
        
        Returns:
            帧图像
        
        Raises:
            StopIteration: 当视频结束时
        """
        success, frame = self.read_frame()
        if not success or frame is None:
            raise StopIteration
        return frame


class VideoWriter:
    """
    视频写入类
    
    用于保存处理后的视频。
    
    Attributes:
        output_path: 输出文件路径
        fps: 帧率
        frame_size: 帧尺寸
        writer: OpenCV VideoWriter 对象
    """
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = 'mp4v'
    ):
        """
        初始化视频写入器
        
        Args:
            output_path: 输出文件路径
            fps: 帧率
            frame_size: 帧尺寸 (width, height)
            fourcc: 编解码器代码（默认 'mp4v'）
        
        Raises:
            ValueError: 如果参数无效
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self._frame_count = 0
        self.writer = None  # 初始化为 None
        
        # 验证参数
        if fps <= 0:
            raise ValueError(f"帧率必须大于 0，得到: {fps}")
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError(f"帧尺寸必须大于 0，得到: {frame_size}")
        
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 VideoWriter
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc_code,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"无法创建视频写入器: {output_path}")
        
        logger.info(f"创建视频写入器: {output_path}")
        logger.info(f"参数: {fps} FPS, {frame_size}")
    
    def write(self, frame: np.ndarray) -> bool:
        """
        写入一帧
        
        Args:
            frame: 帧图像（BGR格式）
        
        Returns:
            是否成功写入
        """
        if self.writer is None or not self.writer.isOpened():
            logger.error("视频写入器未打开")
            return False
        
        try:
            # 验证帧尺寸
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                logger.warning(
                    f"帧尺寸不匹配: 期望 {self.frame_size}, "
                    f"得到 ({frame.shape[1]}, {frame.shape[0]})"
                )
                # 调整尺寸
                frame = cv2.resize(frame, self.frame_size)
            
            self.writer.write(frame)
            self._frame_count += 1
            return True
            
        except Exception as e:
            logger.error(f"写入帧时发生错误: {e}")
            return False
    
    def get_frame_count(self) -> int:
        """
        获取已写入的帧数
        
        Returns:
            已写入的帧数
        """
        return self._frame_count
    
    def release(self) -> None:
        """释放视频写入器"""
        if self.writer is not None:
            self.writer.release()
            logger.info(f"释放视频写入器: {self.output_path}, 共写入 {self._frame_count} 帧")
            self.writer = None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()
    
    def __del__(self):
        """析构函数"""
        self.release()

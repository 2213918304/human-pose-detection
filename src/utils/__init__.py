"""工具模块"""

from .logger import setup_logger, log_with_details
from .video_capture import VideoCapture, VideoWriter

__all__ = [
    "setup_logger",
    "log_with_details",
    "VideoCapture",
    "VideoWriter",
]

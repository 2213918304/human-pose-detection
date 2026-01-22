"""
日志配置模块

提供统一的日志记录功能。
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON 格式的日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        
        # 添加额外的详细信息
        if hasattr(record, 'details'):
            log_data["details"] = record.details
        
        # 添加异常信息
        if record.exc_info:
            log_data["details"] = log_data.get("details", {})
            log_data["details"]["error_type"] = record.exc_info[0].__name__
            log_data["details"]["stack_trace"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_json: bool = False
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        use_json: 是否使用 JSON 格式
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器（设置UTF-8编码）
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    # 设置控制台编码为UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def log_with_details(logger: logging.Logger, level: int, message: str, details: Dict[str, Any]) -> None:
    """
    记录带有详细信息的日志
    
    Args:
        logger: 日志记录器
        level: 日志级别
        message: 日志消息
        details: 详细信息字典
    """
    extra = {'details': details}
    logger.log(level, message, extra=extra)

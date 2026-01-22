#!/usr/bin/env python
"""
人体姿态识别系统 - 命令行接口

使用方法:
    python main.py --source 0                    # 使用默认摄像头
    python main.py --source video.mp4            # 使用视频文件
    python main.py --config config.json          # 使用配置文件
    python main.py --source 0 --save-video       # 保存输出视频
    python main.py --source 0 --export-data      # 导出检测数据
"""

import argparse
import sys
from pathlib import Path

from src.pose_detection_system import PoseDetectionSystem
from src.config.config_manager import ConfigManager, SystemConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='人体姿态识别系统 - 基于 YOLOv8 + MediaPipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --source 0                           使用默认摄像头
  %(prog)s --source video.mp4                   处理视频文件
  %(prog)s --source 0 --save-video              保存输出视频
  %(prog)s --source 0 --export-data             导出检测数据
  %(prog)s --config config.json                 使用配置文件
  %(prog)s --source 0 --no-landmarks            不显示关键点
  %(prog)s --source 0 --fps 60                  设置目标帧率
        """
    )
    
    # 视频源
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='0',
        help='视频源：摄像头索引（如 0）或视频文件路径（默认: 0）'
    )
    
    # 配置文件
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径（JSON 格式）'
    )
    
    # 输出选项
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='保存输出视频'
    )
    
    parser.add_argument(
        '--output-video',
        type=str,
        default='output.mp4',
        help='输出视频路径（默认: output.mp4）'
    )
    
    parser.add_argument(
        '--export-data',
        action='store_true',
        help='导出检测数据为 JSON'
    )
    
    parser.add_argument(
        '--output-data',
        type=str,
        default='detections.json',
        help='输出数据路径（默认: detections.json）'
    )
    
    # 检测参数
    parser.add_argument(
        '--detection-confidence',
        type=float,
        default=None,
        help='最小检测置信度 [0.0-1.0]（默认: 0.5）'
    )
    
    parser.add_argument(
        '--tracking-confidence',
        type=float,
        default=None,
        help='最小跟踪置信度 [0.0-1.0]（默认: 0.5）'
    )
    
    parser.add_argument(
        '--classification-confidence',
        type=float,
        default=None,
        help='分类置信度阈值 [0.0-1.0]（默认: 0.6）'
    )
    
    # 可视化选项
    parser.add_argument(
        '--no-landmarks',
        action='store_true',
        help='不显示关键点'
    )
    
    parser.add_argument(
        '--no-skeleton',
        action='store_true',
        help='不显示骨架'
    )
    
    parser.add_argument(
        '--no-bbox',
        action='store_true',
        help='不显示边界框'
    )
    
    parser.add_argument(
        '--no-label',
        action='store_true',
        help='不显示姿态标签'
    )
    
    # 性能选项
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='目标帧率（默认: 30）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', '0', '1', '2', '3'],
        help='运行设备：cpu 或 cuda（GPU）（默认: cuda）'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='不显示实时画面（加快处理速度）'
    )
    
    # 其他选项
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> SystemConfig:
    """
    从命令行参数创建配置
    
    Args:
        args: 解析后的命令行参数
    
    Returns:
        SystemConfig 实例
    """
    # 如果提供了配置文件，先加载
    if args.config:
        try:
            config_manager = ConfigManager(args.config)
            config = config_manager.get_config()
            logger.info(f"从配置文件加载: {args.config}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            config = SystemConfig()
    else:
        config = SystemConfig()
    
    # 命令行参数覆盖配置文件
    
    # 视频源
    try:
        # 尝试转换为整数（摄像头索引）
        config.video_source = int(args.source)
    except ValueError:
        # 否则作为文件路径
        config.video_source = args.source
    
    # 输出选项
    if args.save_video:
        config.save_video = True
        config.output_video_path = args.output_video
    
    if args.export_data:
        config.export_data = True
        config.export_data_path = args.output_data
    
    # 检测参数
    if args.detection_confidence is not None:
        config.min_detection_confidence = args.detection_confidence
    
    if args.tracking_confidence is not None:
        config.min_tracking_confidence = args.tracking_confidence
    
    if args.classification_confidence is not None:
        config.classification_confidence_threshold = args.classification_confidence
    
    # 可视化选项
    if args.no_landmarks:
        config.show_landmarks = False
    
    if args.no_skeleton:
        config.show_skeleton = False
    
    if args.no_bbox:
        config.show_bbox = False
    
    if args.no_label:
        config.show_label = False
    
    # 性能选项
    if args.fps is not None:
        config.target_fps = args.fps
    
    # 设备选项
    if hasattr(args, 'device'):
        config.device = args.device
    
    # 显示选项
    if hasattr(args, 'no_display'):
        config.show_display = not args.no_display
    
    return config


def validate_config(config: SystemConfig) -> bool:
    """
    验证配置
    
    Args:
        config: 系统配置
    
    Returns:
        配置是否有效
    """
    # 检查视频源
    if isinstance(config.video_source, str):
        if not Path(config.video_source).exists():
            logger.error(f"视频文件不存在: {config.video_source}")
            return False
    
    # 检查置信度范围
    if not (0 <= config.min_detection_confidence <= 1):
        logger.error(f"检测置信度必须在 [0, 1] 范围内: {config.min_detection_confidence}")
        return False
    
    if not (0 <= config.min_tracking_confidence <= 1):
        logger.error(f"跟踪置信度必须在 [0, 1] 范围内: {config.min_tracking_confidence}")
        return False
    
    if not (0 <= config.classification_confidence_threshold <= 1):
        logger.error(f"分类置信度必须在 [0, 1] 范围内: {config.classification_confidence_threshold}")
        return False
    
    return True


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 验证配置
    if not validate_config(config):
        logger.error("配置验证失败")
        sys.exit(1)
    
    # 显示配置信息
    logger.info("=" * 60)
    logger.info("人体姿态识别系统")
    logger.info("=" * 60)
    logger.info(f"视频源: {config.video_source}")
    logger.info(f"检测置信度: {config.min_detection_confidence}")
    logger.info(f"跟踪置信度: {config.min_tracking_confidence}")
    logger.info(f"分类置信度: {config.classification_confidence_threshold}")
    logger.info(f"保存视频: {config.save_video}")
    if config.save_video:
        logger.info(f"  输出路径: {config.output_video_path}")
    logger.info(f"导出数据: {config.export_data}")
    if config.export_data:
        logger.info(f"  输出路径: {config.export_data_path}")
    logger.info(f"显示选项: 边界框={config.show_bbox}, 关键点={config.show_landmarks}, "
               f"骨架={config.show_skeleton}, 标签={config.show_label}")
    logger.info("=" * 60)
    logger.info("按 'q' 或 ESC 键退出")
    logger.info("=" * 60)
    
    # 创建并运行系统
    system = PoseDetectionSystem(config)
    
    try:
        system.run()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"系统运行失败: {e}", exc_info=True)
        sys.exit(1)
    finally:
        system.cleanup()
    
    logger.info("程序结束")


if __name__ == "__main__":
    main()

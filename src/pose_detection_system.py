"""
人体姿态识别系统主应用程序

集成所有模块，实现完整的姿态识别流程。
"""

import cv2
import time
from typing import Optional, List
from pathlib import Path

from src.utils.video_capture import VideoCapture, VideoWriter
from src.detection.yolo_detector import YOLOv8Detector
from src.pose.pose_estimator import PoseEstimator
from src.pose.pose_classifier import PoseClassifier
from src.visualization.visualizer import Visualizer
from src.config.config_manager import ConfigManager, SystemConfig
from src.export.data_exporter import DataExporter
from src.models import DetectionResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PoseDetectionSystem:
    """
    人体姿态识别系统
    
    集成 YOLOv8 检测、MediaPipe 姿态估计和姿态分类功能。
    
    Attributes:
        config: 系统配置
        video_capture: 视频捕获器
        detector: YOLOv8 检测器
        pose_estimator: 姿态估计器
        pose_classifier: 姿态分类器
        visualizer: 可视化器
        video_writer: 视频写入器（可选）
        data_exporter: 数据导出器（可选）
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化系统
        
        Args:
            config: 系统配置（可选，默认使用默认配置）
        """
        self.config = config or SystemConfig()
        
        # 初始化组件
        self.video_capture: Optional[VideoCapture] = None
        self.detector: Optional[YOLOv8Detector] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.pose_classifier: Optional[PoseClassifier] = None
        self.visualizer: Optional[Visualizer] = None
        self.video_writer: Optional[VideoWriter] = None
        self.data_exporter: Optional[DataExporter] = None
        
        # 统计信息
        self.frame_count = 0
        self.start_time = 0.0
        self.is_running = False
        
        logger.info("人体姿态识别系统初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            是否成功初始化
        """
        try:
            logger.info("开始初始化系统组件...")
            
            # 1. 初始化视频捕获
            logger.info(f"初始化视频源: {self.config.video_source}")
            self.video_capture = VideoCapture(self.config.video_source)
            
            if not self.video_capture.is_opened():
                logger.error("无法打开视频源")
                return False
            
            # 2. 初始化检测器
            logger.info("初始化 YOLOv8 检测器...")
            device = self.config.device if hasattr(self.config, 'device') else 'cpu'
            self.detector = YOLOv8Detector(
                confidence_threshold=self.config.min_detection_confidence,
                device=device
            )
            
            # 3. 初始化姿态估计器
            logger.info("初始化 MediaPipe 姿态估计器...")
            self.pose_estimator = PoseEstimator(
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            
            # 4. 初始化姿态分类器
            logger.info("初始化姿态分类器...")
            self.pose_classifier = PoseClassifier(
                confidence_threshold=self.config.classification_confidence_threshold
            )
            
            # 5. 初始化可视化器
            logger.info("初始化可视化器...")
            self.visualizer = Visualizer(
                show_landmarks=self.config.show_landmarks,
                show_skeleton=self.config.show_skeleton,
                show_bbox=self.config.show_bbox,
                show_label=self.config.show_label
            )
            
            # 6. 初始化视频写入器（如果需要）
            if self.config.save_video:
                logger.info(f"初始化视频写入器: {self.config.output_video_path}")
                frame_size = self.video_capture.get_frame_size()
                fps = self.video_capture.get_fps()
                
                self.video_writer = VideoWriter(
                    self.config.output_video_path,
                    fps=fps,
                    frame_size=frame_size
                )
            
            # 7. 初始化数据导出器（如果需要）
            if self.config.export_data:
                logger.info(f"初始化数据导出器: {self.config.export_data_path}")
                self.data_exporter = DataExporter(self.config.export_data_path)
                
                # 设置元数据
                video_source_str = str(self.config.video_source)
                fps = self.video_capture.get_fps()
                self.data_exporter.set_metadata(
                    video_source=video_source_str,
                    fps=fps
                )
            
            logger.info("所有组件初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            self.cleanup()
            return False
    
    def process_frame(self, frame) -> Optional[List[DetectionResult]]:
        """
        处理单帧
        
        Args:
            frame: 输入帧
        
        Returns:
            检测结果列表，如果处理失败返回 None
        """
        try:
            # 1. 人体检测
            person_detections = self.detector.detect(frame)
            
            if not person_detections:
                return []
            
            # 2. 姿态估计和分类
            detection_results = []
            
            for person_det in person_detections:
                try:
                    # 姿态估计
                    pose_estimation = self.pose_estimator.estimate(
                        frame,
                        person_det.bounding_box
                    )
                    
                    landmarks = None
                    pose_classification = None
                    
                    if pose_estimation is not None:
                        landmarks = pose_estimation.landmarks
                        
                        # 姿态分类
                        try:
                            pose_classification = self.pose_classifier.classify(landmarks)
                        except Exception as e:
                            logger.warning(f"姿态分类失败: {e}")
                    
                    # 创建检测结果
                    detection_result = DetectionResult(
                        person_id=person_det.person_id,
                        bounding_box=person_det.bounding_box,
                        confidence=person_det.confidence,
                        landmarks=landmarks,
                        pose_classification=pose_classification,
                        timestamp=time.time() - self.start_time
                    )
                    
                    detection_results.append(detection_result)
                    
                except Exception as e:
                    logger.warning(f"处理人员 {person_det.person_id} 失败: {e}")
                    continue
            
            return detection_results
            
        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            return None
    
    def run(self) -> None:
        """
        运行主处理循环
        """
        if not self.initialize():
            logger.error("系统初始化失败，无法运行")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        logger.info("开始处理视频...")
        
        try:
            while self.is_running:
                # 读取帧
                success, frame = self.video_capture.read_frame()
                
                if not success:
                    logger.info("视频处理完成或读取失败")
                    break
                
                # 处理帧
                detection_results = self.process_frame(frame)
                
                if detection_results is None:
                    logger.warning(f"帧 {self.frame_count} 处理失败，跳过")
                    continue
                
                # 可视化
                if self.visualizer is not None:
                    frame = self.visualizer.draw(frame, detection_results)
                
                # 保存视频
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                
                # 导出数据
                if self.data_exporter is not None:
                    timestamp = time.time() - self.start_time
                    self.data_exporter.add_frame_data(
                        self.frame_count,
                        timestamp,
                        detection_results
                    )
                
                # 显示
                show_display = self.config.show_display if hasattr(self.config, 'show_display') else True
                if show_display:
                    cv2.imshow('Pose Detection', frame)
                    
                    # 检查退出键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' 或 ESC
                        logger.info("用户请求退出")
                        break
                else:
                    # 不显示时也检查是否有中断
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        logger.info("用户请求退出")
                        break
                
                self.frame_count += 1
                
                # 显示进度
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"已处理 {self.frame_count} 帧，FPS: {fps:.2f}")
        
        except KeyboardInterrupt:
            logger.info("用户中断处理")
        
        except Exception as e:
            logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """停止系统并清理资源"""
        logger.info("停止系统...")
        self.is_running = False
        
        # 导出数据
        if self.data_exporter is not None:
            try:
                logger.info("导出数据...")
                self.data_exporter.export()
            except Exception as e:
                logger.error(f"导出数据失败: {e}")
        
        # 清理资源
        self.cleanup()
        
        # 显示统计信息
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"处理完成: 总帧数={self.frame_count}, "
                       f"总时间={elapsed:.2f}s, 平均FPS={fps:.2f}")
    
    def cleanup(self) -> None:
        """清理所有资源"""
        logger.info("清理资源...")
        
        try:
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            
            if self.detector is not None:
                self.detector.close()
                self.detector = None
            
            if self.pose_estimator is not None:
                self.pose_estimator.close()
                self.pose_estimator = None
            
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            cv2.destroyAllWindows()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")
    
    def get_stats(self) -> dict:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "fps": fps,
            "is_running": self.is_running
        }


def main():
    """主函数"""
    # 加载配置
    config_manager = ConfigManager("config.json")
    config = config_manager.get_config()
    
    # 创建系统
    system = PoseDetectionSystem(config)
    
    # 运行
    try:
        system.run()
    except Exception as e:
        logger.error(f"系统运行失败: {e}", exc_info=True)
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()

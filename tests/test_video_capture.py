"""
VideoCapture 模块测试

包含单元测试和属性测试。
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from src.utils import VideoCapture, VideoWriter


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_video(path: str, num_frames: int = 30, fps: float = 30.0, size: tuple = (640, 480)):
    """创建测试视频文件"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    
    for i in range(num_frames):
        # 创建渐变帧
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :] = (i * 255 // num_frames, 128, 255 - i * 255 // num_frames)
        writer.write(frame)
    
    writer.release()


@pytest.fixture
def test_video_file(tmp_path):
    """创建临时测试视频文件"""
    video_path = tmp_path / "test_video.mp4"
    create_test_video(str(video_path), num_frames=10, fps=30.0)
    return str(video_path)


@pytest.fixture
def test_output_path(tmp_path):
    """创建临时输出路径"""
    return str(tmp_path / "output.mp4")


# ============================================================================
# 单元测试 - VideoCapture
# ============================================================================

class TestVideoCapture:
    """VideoCapture 单元测试"""
    
    def test_create_with_video_file(self, test_video_file):
        """测试使用视频文件创建"""
        cap = VideoCapture(test_video_file)
        assert cap.is_opened()
        assert not cap.is_camera()
        assert cap.get_total_frames() == 10
        cap.release()
    
    def test_create_with_nonexistent_file(self):
        """测试使用不存在的文件"""
        with pytest.raises(FileNotFoundError):
            VideoCapture("nonexistent_video.mp4")
    
    def test_read_frame_from_video(self, test_video_file):
        """测试从视频读取帧"""
        cap = VideoCapture(test_video_file)
        success, frame = cap.read_frame()
        
        assert success
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # (height, width, channels)
        assert frame.shape[2] == 3  # BGR
        
        cap.release()
    
    def test_read_all_frames(self, test_video_file):
        """测试读取所有帧"""
        cap = VideoCapture(test_video_file)
        frame_count = 0
        
        while True:
            success, frame = cap.read_frame()
            if not success:
                break
            frame_count += 1
        
        assert frame_count == 10
        assert cap.get_frame_count() == 10
        cap.release()
    
    def test_get_fps(self, test_video_file):
        """测试获取帧率"""
        cap = VideoCapture(test_video_file)
        fps = cap.get_fps()
        assert fps > 0
        assert fps == pytest.approx(30.0, rel=0.1)
        cap.release()
    
    def test_get_frame_size(self, test_video_file):
        """测试获取帧尺寸"""
        cap = VideoCapture(test_video_file)
        width, height = cap.get_frame_size()
        assert width == 640
        assert height == 480
        cap.release()
    
    def test_get_position(self, test_video_file):
        """测试获取播放位置"""
        cap = VideoCapture(test_video_file)
        
        # 初始位置
        assert cap.get_position() == 0.0
        
        # 读取一半帧
        for _ in range(5):
            cap.read_frame()
        
        position = cap.get_position()
        assert 0.4 < position < 0.6  # 大约 50%
        
        cap.release()
    
    def test_is_opened(self, test_video_file):
        """测试检查是否打开"""
        cap = VideoCapture(test_video_file)
        assert cap.is_opened()
        
        cap.release()
        assert not cap.is_opened()
    
    def test_context_manager(self, test_video_file):
        """测试上下文管理器"""
        with VideoCapture(test_video_file) as cap:
            assert cap.is_opened()
            success, frame = cap.read_frame()
            assert success
        
        # 退出上下文后应该自动释放
        assert not cap.is_opened()
    
    def test_iterator_interface(self, test_video_file):
        """测试迭代器接口"""
        cap = VideoCapture(test_video_file)
        frames = list(cap)
        
        assert len(frames) == 10
        for frame in frames:
            assert isinstance(frame, np.ndarray)
    
    def test_read_after_release(self, test_video_file):
        """测试释放后读取"""
        cap = VideoCapture(test_video_file)
        cap.release()
        
        success, frame = cap.read_frame()
        assert not success
        assert frame is None


# ============================================================================
# 单元测试 - VideoWriter
# ============================================================================

class TestVideoWriter:
    """VideoWriter 单元测试"""
    
    def test_create_video_writer(self, test_output_path):
        """测试创建视频写入器"""
        writer = VideoWriter(test_output_path, fps=30.0, frame_size=(640, 480))
        assert writer.get_frame_count() == 0
        writer.release()
        
        # 验证文件已创建
        assert Path(test_output_path).exists()
    
    def test_write_frame(self, test_output_path):
        """测试写入帧"""
        writer = VideoWriter(test_output_path, fps=30.0, frame_size=(640, 480))
        
        # 创建测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # 蓝色
        
        success = writer.write(frame)
        assert success
        assert writer.get_frame_count() == 1
        
        writer.release()
    
    def test_write_multiple_frames(self, test_output_path):
        """测试写入多帧"""
        writer = VideoWriter(test_output_path, fps=30.0, frame_size=(640, 480))
        
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = (i * 25, 128, 255 - i * 25)
            writer.write(frame)
        
        assert writer.get_frame_count() == 10
        writer.release()
        
        # 验证视频可以读取
        cap = VideoCapture(test_output_path)
        assert cap.get_total_frames() == 10
        cap.release()
    
    def test_write_wrong_size_frame(self, test_output_path):
        """测试写入错误尺寸的帧"""
        writer = VideoWriter(test_output_path, fps=30.0, frame_size=(640, 480))
        
        # 创建不同尺寸的帧
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        
        # 应该自动调整尺寸
        success = writer.write(frame)
        assert success
        
        writer.release()
    
    def test_invalid_fps(self, test_output_path):
        """测试无效的帧率"""
        with pytest.raises(ValueError, match="帧率必须大于 0"):
            VideoWriter(test_output_path, fps=0, frame_size=(640, 480))
    
    def test_invalid_frame_size(self, test_output_path):
        """测试无效的帧尺寸"""
        with pytest.raises(ValueError, match="帧尺寸必须大于 0"):
            VideoWriter(test_output_path, fps=30.0, frame_size=(0, 480))
    
    def test_context_manager(self, test_output_path):
        """测试上下文管理器"""
        with VideoWriter(test_output_path, fps=30.0, frame_size=(640, 480)) as writer:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)
            assert writer.get_frame_count() == 1
        
        # 验证文件已创建
        assert Path(test_output_path).exists()


# ============================================================================
# 单元测试 - 集成测试
# ============================================================================

class TestVideoIntegration:
    """VideoCapture 和 VideoWriter 集成测试"""
    
    def test_read_and_write(self, test_video_file, test_output_path):
        """测试读取和写入视频"""
        # 读取视频
        cap = VideoCapture(test_video_file)
        fps = cap.get_fps()
        size = cap.get_frame_size()
        
        # 写入新视频
        writer = VideoWriter(test_output_path, fps=fps, frame_size=size)
        
        for frame in cap:
            writer.write(frame)
        
        cap.release()
        writer.release()
        
        # 验证输出视频
        output_cap = VideoCapture(test_output_path)
        assert output_cap.get_total_frames() == 10
        assert output_cap.get_frame_size() == size
        output_cap.release()


# ============================================================================
# 属性测试 - Feature: human-pose-detection, Property 1: 持续帧处理
# ============================================================================

@pytest.mark.property
class TestContinuousFrameProcessingProperty:
    """
    属性 1: 持续帧处理
    验证需求: 1.1
    
    对于任何有效的视频流，系统应该能够持续处理所有帧而不中断，直到视频流结束
    """
    
    def test_process_all_frames_without_interruption(self, test_video_file):
        """
        Feature: human-pose-detection, Property 1: 持续帧处理
        
        对于任何视频文件，应该能够读取所有帧而不中断
        """
        cap = VideoCapture(test_video_file)
        total_frames = cap.get_total_frames()
        processed_frames = 0
        
        # 持续处理所有帧
        while True:
            success, frame = cap.read_frame()
            if not success:
                break
            
            # 验证帧有效
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            processed_frames += 1
        
        # 验证处理了所有帧
        assert processed_frames == total_frames
        cap.release()
    
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_frames=st.integers(min_value=1, max_value=50),
        fps=st.floats(min_value=1.0, max_value=60.0),
    )
    def test_process_generated_video_continuously(self, tmp_path, num_frames, fps):
        """
        Feature: human-pose-detection, Property 1: 持续帧处理
        
        对于任何生成的视频，应该能够持续处理所有帧
        """
        # 创建测试视频
        video_path = tmp_path / f"test_{num_frames}_{fps}.mp4"
        create_test_video(str(video_path), num_frames=num_frames, fps=fps)
        
        # 处理视频
        cap = VideoCapture(str(video_path))
        processed = 0
        
        for frame in cap:
            assert frame is not None
            processed += 1
        
        # 验证处理了所有帧
        assert processed == num_frames


# ============================================================================
# 属性测试 - Feature: human-pose-detection, Property 2: 优雅终止
# ============================================================================

@pytest.mark.property
class TestGracefulTerminationProperty:
    """
    属性 2: 优雅终止
    验证需求: 1.3
    
    对于任何视频流中断或结束事件，系统应该优雅地处理终止而不崩溃
    """
    
    def test_graceful_end_of_video(self, test_video_file):
        """
        Feature: human-pose-detection, Property 2: 优雅终止
        
        视频结束时应该优雅地返回 False，而不是抛出异常
        """
        cap = VideoCapture(test_video_file)
        
        # 读取所有帧
        while True:
            success, frame = cap.read_frame()
            if not success:
                break
        
        # 尝试再次读取（视频已结束）
        success, frame = cap.read_frame()
        assert not success
        assert frame is None
        
        # 应该仍然可以安全释放
        cap.release()
        assert not cap.is_opened()
    
    def test_release_multiple_times(self, test_video_file):
        """
        Feature: human-pose-detection, Property 2: 优雅终止
        
        多次释放不应该导致错误
        """
        cap = VideoCapture(test_video_file)
        
        # 多次释放
        cap.release()
        cap.release()
        cap.release()
        
        # 不应该抛出异常
        assert not cap.is_opened()
    
    def test_context_manager_exception_handling(self, test_video_file):
        """
        Feature: human-pose-detection, Property 2: 优雅终止
        
        上下文管理器应该在异常时也能正确清理
        """
        try:
            with VideoCapture(test_video_file) as cap:
                cap.read_frame()
                raise ValueError("测试异常")
        except ValueError:
            pass
        
        # 即使发生异常，资源也应该被释放
        assert not cap.is_opened()

"""
DataExporter 模块测试

测试数据导出功能。
"""

import pytest
import json
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings

from src.export import DataExporter
from src.models import DetectionResult, BoundingBox, Landmark, PoseClassification, PoseState


# ============================================================================
# 辅助函数
# ============================================================================

def create_test_landmark(x: float = 0.5, y: float = 0.5) -> Landmark:
    """创建测试关键点"""
    return Landmark(x=x, y=y, z=0.0, visibility=1.0)


def create_test_detection(
    person_id: int = 1,
    with_landmarks: bool = False,
    with_classification: bool = False
) -> DetectionResult:
    """创建测试检测结果"""
    bbox = BoundingBox(x=100, y=50, width=200, height=400)
    
    landmarks = None
    if with_landmarks:
        landmarks = [create_test_landmark(i * 0.03, i * 0.03) for i in range(33)]
    
    classification = None
    if with_classification:
        classification = PoseClassification(
            pose_state=PoseState.STANDING,
            confidence=0.95,
            features={"torso_angle": 5.0}
        )
    
    return DetectionResult(
        person_id=person_id,
        bounding_box=bbox,
        confidence=0.9,
        landmarks=landmarks,
        pose_classification=classification,
        timestamp=0.0
    )


# ============================================================================
# 单元测试
# ============================================================================

class TestDataExporterInitialization:
    """测试 DataExporter 初始化"""
    
    def test_init(self):
        """测试初始化"""
        exporter = DataExporter("output.json")
        
        assert exporter.output_path == "output.json"
        assert exporter.metadata["video_source"] == ""
        assert exporter.metadata["total_frames"] == 0
        assert exporter.metadata["fps"] == 0.0
        assert len(exporter.frames_data) == 0
    
    def test_set_metadata(self):
        """测试设置元数据"""
        exporter = DataExporter("output.json")
        
        exporter.set_metadata(video_source="test.mp4", fps=30.0)
        
        assert exporter.metadata["video_source"] == "test.mp4"
        assert exporter.metadata["fps"] == 30.0


class TestDataExporterAddData:
    """测试添加数据"""
    
    def test_add_frame_data_empty(self):
        """测试添加空帧数据"""
        exporter = DataExporter("output.json")
        
        exporter.add_frame_data(0, 0.0, [])
        
        assert len(exporter.frames_data) == 1
        assert exporter.frames_data[0]["frame_number"] == 0
        assert exporter.frames_data[0]["timestamp"] == 0.0
        assert len(exporter.frames_data[0]["detections"]) == 0
    
    def test_add_frame_data_with_detection(self):
        """测试添加带检测的帧数据"""
        exporter = DataExporter("output.json")
        detection = create_test_detection(person_id=1)
        
        exporter.add_frame_data(0, 0.0, [detection])
        
        assert len(exporter.frames_data) == 1
        assert len(exporter.frames_data[0]["detections"]) == 1
        
        det_dict = exporter.frames_data[0]["detections"][0]
        assert det_dict["person_id"] == 1
        assert det_dict["confidence"] == 0.9
        assert "bounding_box" in det_dict
    
    def test_add_frame_data_with_landmarks(self):
        """测试添加带关键点的帧数据"""
        exporter = DataExporter("output.json")
        detection = create_test_detection(person_id=1, with_landmarks=True)
        
        exporter.add_frame_data(0, 0.0, [detection])
        
        det_dict = exporter.frames_data[0]["detections"][0]
        assert "landmarks" in det_dict
        assert len(det_dict["landmarks"]) == 33
        assert "x" in det_dict["landmarks"][0]
        assert "y" in det_dict["landmarks"][0]
        assert "z" in det_dict["landmarks"][0]
        assert "visibility" in det_dict["landmarks"][0]
    
    def test_add_frame_data_with_classification(self):
        """测试添加带分类的帧数据"""
        exporter = DataExporter("output.json")
        detection = create_test_detection(person_id=1, with_classification=True)
        
        exporter.add_frame_data(0, 0.0, [detection])
        
        det_dict = exporter.frames_data[0]["detections"][0]
        assert "pose_state" in det_dict
        assert det_dict["pose_state"] == "站立"
        assert "pose_confidence" in det_dict
        assert det_dict["pose_confidence"] == 0.95
        assert "features" in det_dict
    
    def test_add_multiple_frames(self):
        """测试添加多帧数据"""
        exporter = DataExporter("output.json")
        
        for i in range(5):
            detection = create_test_detection(person_id=i)
            exporter.add_frame_data(i, i * 0.033, [detection])
        
        assert len(exporter.frames_data) == 5
        assert exporter.frames_data[0]["frame_number"] == 0
        assert exporter.frames_data[4]["frame_number"] == 4
    
    def test_add_multiple_detections_per_frame(self):
        """测试每帧添加多个检测"""
        exporter = DataExporter("output.json")
        
        detections = [
            create_test_detection(person_id=1),
            create_test_detection(person_id=2),
            create_test_detection(person_id=3)
        ]
        
        exporter.add_frame_data(0, 0.0, detections)
        
        assert len(exporter.frames_data[0]["detections"]) == 3
        assert exporter.frames_data[0]["detections"][0]["person_id"] == 1
        assert exporter.frames_data[0]["detections"][2]["person_id"] == 3


class TestDataExporterExport:
    """测试导出功能"""
    
    def test_export_empty_data(self):
        """测试导出空数据"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            exporter = DataExporter(temp_path)
            exporter.set_metadata(video_source="test.mp4", fps=30.0)
            exporter.export()
            
            # 验证文件存在
            assert Path(temp_path).exists()
            
            # 验证内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "frames" in data
            assert data["metadata"]["video_source"] == "test.mp4"
            assert data["metadata"]["fps"] == 30.0
            assert data["metadata"]["total_frames"] == 0
            assert len(data["frames"]) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_export_with_data(self):
        """测试导出带数据"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            exporter = DataExporter(temp_path)
            exporter.set_metadata(video_source="test.mp4", fps=30.0)
            
            # 添加数据
            for i in range(3):
                detection = create_test_detection(
                    person_id=i,
                    with_landmarks=True,
                    with_classification=True
                )
                exporter.add_frame_data(i, i * 0.033, [detection])
            
            exporter.export()
            
            # 验证内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["metadata"]["total_frames"] == 3
            assert len(data["frames"]) == 3
            assert data["frames"][0]["frame_number"] == 0
            assert len(data["frames"][0]["detections"]) == 1
        finally:
            Path(temp_path).unlink()
    
    def test_export_creates_directory(self):
        """测试导出时创建目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "output.json"
            
            exporter = DataExporter(str(output_path))
            exporter.export()
            
            assert output_path.exists()
            assert output_path.parent.exists()


class TestDataExporterMethods:
    """测试其他方法"""
    
    def test_clear(self):
        """测试清空数据"""
        exporter = DataExporter("output.json")
        
        # 添加数据
        detection = create_test_detection()
        exporter.add_frame_data(0, 0.0, [detection])
        exporter.add_frame_data(1, 0.033, [detection])
        
        assert len(exporter.frames_data) == 2
        
        # 清空
        exporter.clear()
        
        assert len(exporter.frames_data) == 0
        assert exporter.metadata["total_frames"] == 0
    
    def test_get_frame_count(self):
        """测试获取帧数"""
        exporter = DataExporter("output.json")
        
        assert exporter.get_frame_count() == 0
        
        detection = create_test_detection()
        exporter.add_frame_data(0, 0.0, [detection])
        assert exporter.get_frame_count() == 1
        
        exporter.add_frame_data(1, 0.033, [detection])
        assert exporter.get_frame_count() == 2
    
    def test_get_info(self):
        """测试获取信息"""
        exporter = DataExporter("output.json")
        exporter.set_metadata(video_source="test.mp4", fps=30.0)
        
        detection = create_test_detection()
        exporter.add_frame_data(0, 0.0, [detection])
        
        info = exporter.get_info()
        
        assert info["output_path"] == "output.json"
        assert info["frame_count"] == 1
        assert info["metadata"]["video_source"] == "test.mp4"
        assert info["metadata"]["fps"] == 30.0


class TestDataExporterLoad:
    """测试加载功能"""
    
    def test_load_from_file(self):
        """测试从文件加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出数据
            exporter = DataExporter(temp_path)
            exporter.set_metadata(video_source="test.mp4", fps=30.0)
            detection = create_test_detection(with_landmarks=True, with_classification=True)
            exporter.add_frame_data(0, 0.0, [detection])
            exporter.export()
            
            # 加载数据
            loaded_data = DataExporter.load_from_file(temp_path)
            
            assert "metadata" in loaded_data
            assert "frames" in loaded_data
            assert loaded_data["metadata"]["video_source"] == "test.mp4"
            assert loaded_data["metadata"]["total_frames"] == 1
            assert len(loaded_data["frames"]) == 1
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            DataExporter.load_from_file("nonexistent.json")
    
    def test_load_invalid_json(self):
        """测试加载无效 JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="文件格式错误"):
                DataExporter.load_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


# ============================================================================
# 属性测试
# ============================================================================

class TestDataExporterProperties:
    """
    属性 10: 数据导出往返
    
    验证需求: 6.3
    
    数据导出系统应该：
    1. 导出的数据可以被重新加载
    2. 加载后的数据与原始数据等效
    3. JSON 格式正确且完整
    """
    
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(
        num_frames=st.integers(min_value=0, max_value=10),
        num_detections=st.integers(min_value=0, max_value=3)
    )
    def test_property_10_export_import_roundtrip(
        self,
        num_frames: int,
        num_detections: int
    ):
        """
        属性 10: 数据导出往返
        
        验证导出后重新加载的数据与原始数据等效。
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 创建导出器并添加数据
            exporter = DataExporter(temp_path)
            exporter.set_metadata(video_source="test.mp4", fps=30.0)
            
            for frame_num in range(num_frames):
                detections = []
                for det_id in range(num_detections):
                    detection = create_test_detection(
                        person_id=det_id,
                        with_landmarks=(det_id % 2 == 0),
                        with_classification=(det_id % 2 == 1)
                    )
                    detections.append(detection)
                
                exporter.add_frame_data(frame_num, frame_num * 0.033, detections)
            
            # 导出
            exporter.export()
            
            # 加载
            loaded_data = DataExporter.load_from_file(temp_path)
            
            # 验证元数据
            assert loaded_data["metadata"]["video_source"] == "test.mp4"
            assert loaded_data["metadata"]["fps"] == 30.0
            assert loaded_data["metadata"]["total_frames"] == num_frames
            
            # 验证帧数据
            assert len(loaded_data["frames"]) == num_frames
            
            for i, frame_data in enumerate(loaded_data["frames"]):
                assert frame_data["frame_number"] == i
                assert len(frame_data["detections"]) == num_detections
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        video_source=st.text(min_size=1, max_size=50),
        fps=st.floats(min_value=1.0, max_value=120.0)
    )
    def test_property_10_metadata_preservation(
        self,
        video_source: str,
        fps: float
    ):
        """
        属性 10: 元数据保存
        
        验证元数据在导出和加载后保持不变。
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            exporter = DataExporter(temp_path)
            exporter.set_metadata(video_source=video_source, fps=fps)
            exporter.export()
            
            loaded_data = DataExporter.load_from_file(temp_path)
            
            assert loaded_data["metadata"]["video_source"] == video_source
            assert abs(loaded_data["metadata"]["fps"] - fps) < 0.001
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        person_id=st.integers(min_value=0, max_value=100),
        confidence=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_property_10_detection_data_preservation(
        self,
        person_id: int,
        confidence: float
    ):
        """
        属性 10: 检测数据保存
        
        验证检测数据在导出和加载后保持不变。
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            exporter = DataExporter(temp_path)
            
            bbox = BoundingBox(x=100, y=50, width=200, height=400)
            detection = DetectionResult(
                person_id=person_id,
                bounding_box=bbox,
                confidence=confidence,
                timestamp=0.0
            )
            
            exporter.add_frame_data(0, 0.0, [detection])
            exporter.export()
            
            loaded_data = DataExporter.load_from_file(temp_path)
            
            det_data = loaded_data["frames"][0]["detections"][0]
            assert det_data["person_id"] == person_id
            assert abs(det_data["confidence"] - confidence) < 0.001
            assert det_data["bounding_box"]["x"] == 100
            assert det_data["bounding_box"]["y"] == 50
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

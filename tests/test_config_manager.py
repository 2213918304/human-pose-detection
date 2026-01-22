"""
ConfigManager 模块测试

测试配置管理器的加载、保存和访问功能。
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.config import ConfigManager, SystemConfig


# ============================================================================
# 辅助函数
# ============================================================================

def create_test_config_file(config_dict: dict) -> str:
    """创建临时配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(config_dict, f)
        return f.name


# ============================================================================
# SystemConfig 测试
# ============================================================================

class TestSystemConfig:
    """测试 SystemConfig 数据类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SystemConfig()
        
        assert config.video_source == 0
        assert config.target_fps == 30
        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
        assert config.classification_confidence_threshold == 0.6
        assert config.show_landmarks is True
        assert config.show_skeleton is True
        assert config.show_bbox is True
        assert config.show_label is True
        assert config.save_video is False
        assert config.output_video_path == "output.mp4"
        assert config.export_data is False
        assert config.export_data_path == "detections.json"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SystemConfig(
            video_source="test.mp4",
            target_fps=60,
            min_detection_confidence=0.7,
            show_landmarks=False,
            save_video=True
        )
        
        assert config.video_source == "test.mp4"
        assert config.target_fps == 60
        assert config.min_detection_confidence == 0.7
        assert config.show_landmarks is False
        assert config.save_video is True
    
    def test_invalid_target_fps(self):
        """测试无效的目标帧率"""
        with pytest.raises(ValueError, match="target_fps 必须大于 0"):
            SystemConfig(target_fps=0)
        
        with pytest.raises(ValueError, match="target_fps 必须大于 0"):
            SystemConfig(target_fps=-10)
    
    def test_invalid_detection_confidence(self):
        """测试无效的检测置信度"""
        with pytest.raises(ValueError, match="min_detection_confidence 必须在"):
            SystemConfig(min_detection_confidence=-0.1)
        
        with pytest.raises(ValueError, match="min_detection_confidence 必须在"):
            SystemConfig(min_detection_confidence=1.5)
    
    def test_invalid_tracking_confidence(self):
        """测试无效的跟踪置信度"""
        with pytest.raises(ValueError, match="min_tracking_confidence 必须在"):
            SystemConfig(min_tracking_confidence=-0.1)
        
        with pytest.raises(ValueError, match="min_tracking_confidence 必须在"):
            SystemConfig(min_tracking_confidence=1.5)
    
    def test_invalid_classification_threshold(self):
        """测试无效的分类阈值"""
        with pytest.raises(ValueError, match="classification_confidence_threshold 必须在"):
            SystemConfig(classification_confidence_threshold=-0.1)
        
        with pytest.raises(ValueError, match="classification_confidence_threshold 必须在"):
            SystemConfig(classification_confidence_threshold=1.5)
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = SystemConfig(
            video_source="test.mp4",
            target_fps=60
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['video_source'] == "test.mp4"
        assert config_dict['target_fps'] == 60
        assert 'min_detection_confidence' in config_dict
    
    def test_from_dict(self):
        """测试从字典创建"""
        config_dict = {
            'video_source': 'test.mp4',
            'target_fps': 60,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'classification_confidence_threshold': 0.6,
            'show_landmarks': False,
            'show_skeleton': True,
            'show_bbox': True,
            'show_label': True,
            'save_video': True,
            'output_video_path': 'output.mp4',
            'export_data': False,
            'export_data_path': 'detections.json'
        }
        
        config = SystemConfig.from_dict(config_dict)
        
        assert config.video_source == 'test.mp4'
        assert config.target_fps == 60
        assert config.min_detection_confidence == 0.7
        assert config.show_landmarks is False
        assert config.save_video is True


# ============================================================================
# ConfigManager 测试
# ============================================================================

class TestConfigManagerInitialization:
    """测试 ConfigManager 初始化"""
    
    def test_init_without_path(self):
        """测试无路径初始化"""
        manager = ConfigManager()
        
        assert isinstance(manager.config, SystemConfig)
        assert manager.config_path is None
    
    def test_init_with_valid_path(self):
        """测试有效路径初始化"""
        config_dict = {
            'video_source': 'test.mp4',
            'target_fps': 60,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'classification_confidence_threshold': 0.6,
            'show_landmarks': False,
            'show_skeleton': True,
            'show_bbox': True,
            'show_label': True,
            'save_video': True,
            'output_video_path': 'output.mp4',
            'export_data': False,
            'export_data_path': 'detections.json'
        }
        
        config_path = create_test_config_file(config_dict)
        
        try:
            manager = ConfigManager(config_path)
            
            assert manager.config.video_source == 'test.mp4'
            assert manager.config.target_fps == 60
            assert manager.config_path == config_path
        finally:
            Path(config_path).unlink()
    
    def test_init_with_nonexistent_path(self):
        """测试不存在的路径初始化（应使用默认配置）"""
        manager = ConfigManager("nonexistent_config.json")
        
        # 应该使用默认配置
        assert manager.config.video_source == 0
        assert manager.config.target_fps == 30


class TestConfigManagerGetSet:
    """测试配置获取和设置"""
    
    def test_get_existing_key(self):
        """测试获取存在的键"""
        manager = ConfigManager()
        
        value = manager.get('video_source')
        assert value == 0
        
        value = manager.get('target_fps')
        assert value == 30
    
    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        manager = ConfigManager()
        
        value = manager.get('nonexistent_key')
        assert value is None
        
        value = manager.get('nonexistent_key', 'default_value')
        assert value == 'default_value'
    
    def test_set_valid_value(self):
        """测试设置有效值"""
        manager = ConfigManager()
        
        manager.set('video_source', 'test.mp4')
        assert manager.config.video_source == 'test.mp4'
        
        manager.set('target_fps', 60)
        assert manager.config.target_fps == 60
        
        manager.set('show_landmarks', False)
        assert manager.config.show_landmarks is False
    
    def test_set_invalid_key(self):
        """测试设置不存在的键"""
        manager = ConfigManager()
        
        with pytest.raises(AttributeError, match="配置键 .* 不存在"):
            manager.set('nonexistent_key', 'value')
    
    def test_set_invalid_value(self):
        """测试设置无效值"""
        manager = ConfigManager()
        
        with pytest.raises(ValueError):
            manager.set('target_fps', -10)
        
        with pytest.raises(ValueError):
            manager.set('min_detection_confidence', 1.5)


class TestConfigManagerSaveLoad:
    """测试配置保存和加载"""
    
    def test_save_config(self):
        """测试保存配置"""
        manager = ConfigManager()
        manager.set('video_source', 'test.mp4')
        manager.set('target_fps', 60)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager.save(temp_path)
            
            # 验证文件存在
            assert Path(temp_path).exists()
            
            # 验证内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            
            assert saved_config['video_source'] == 'test.mp4'
            assert saved_config['target_fps'] == 60
        finally:
            Path(temp_path).unlink()
    
    def test_save_without_path(self):
        """测试无路径保存（应该失败）"""
        manager = ConfigManager()
        
        with pytest.raises(ValueError, match="必须提供保存路径"):
            manager.save()
    
    def test_save_with_initial_path(self):
        """测试使用初始路径保存"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 创建初始配置文件
            config_dict = SystemConfig().to_dict()
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f)
            
            manager = ConfigManager(temp_path)
            manager.set('video_source', 'updated.mp4')
            
            # 保存（不提供路径，应使用初始路径）
            manager.save()
            
            # 验证更新
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            
            assert saved_config['video_source'] == 'updated.mp4'
        finally:
            Path(temp_path).unlink()
    
    def test_load_valid_config(self):
        """测试加载有效配置"""
        config_dict = {
            'video_source': 'loaded.mp4',
            'target_fps': 45,
            'min_detection_confidence': 0.8,
            'min_tracking_confidence': 0.5,
            'classification_confidence_threshold': 0.6,
            'show_landmarks': False,
            'show_skeleton': True,
            'show_bbox': True,
            'show_label': True,
            'save_video': True,
            'output_video_path': 'output.mp4',
            'export_data': False,
            'export_data_path': 'detections.json'
        }
        
        config_path = create_test_config_file(config_dict)
        
        try:
            manager = ConfigManager()
            manager.load(config_path)
            
            assert manager.config.video_source == 'loaded.mp4'
            assert manager.config.target_fps == 45
            assert manager.config.min_detection_confidence == 0.8
            assert manager.config.show_landmarks is False
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        manager = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load('nonexistent_config.json')
    
    def test_load_invalid_json(self):
        """测试加载无效 JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            manager = ConfigManager()
            
            with pytest.raises(ValueError, match="配置文件格式错误"):
                manager.load(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_config_values(self):
        """测试加载无效配置值"""
        config_dict = {
            'video_source': 0,
            'target_fps': -10,  # 无效值
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'classification_confidence_threshold': 0.6,
            'show_landmarks': True,
            'show_skeleton': True,
            'show_bbox': True,
            'show_label': True,
            'save_video': False,
            'output_video_path': 'output.mp4',
            'export_data': False,
            'export_data_path': 'detections.json'
        }
        
        config_path = create_test_config_file(config_dict)
        
        try:
            manager = ConfigManager()
            
            with pytest.raises(Exception):  # 应该抛出验证错误
                manager.load(config_path)
        finally:
            Path(config_path).unlink()


class TestConfigManagerMethods:
    """测试配置管理器方法"""
    
    def test_get_config(self):
        """测试获取配置对象"""
        manager = ConfigManager()
        
        config = manager.get_config()
        
        assert isinstance(config, SystemConfig)
        assert config.video_source == 0
    
    def test_reset_to_default(self):
        """测试重置为默认配置"""
        manager = ConfigManager()
        manager.set('video_source', 'test.mp4')
        manager.set('target_fps', 60)
        
        manager.reset_to_default()
        
        assert manager.config.video_source == 0
        assert manager.config.target_fps == 30
    
    def test_update_from_dict(self):
        """测试从字典更新配置"""
        manager = ConfigManager()
        
        update_dict = {
            'video_source': 'updated.mp4',
            'target_fps': 45,
            'show_landmarks': False
        }
        
        manager.update_from_dict(update_dict)
        
        assert manager.config.video_source == 'updated.mp4'
        assert manager.config.target_fps == 45
        assert manager.config.show_landmarks is False
        # 其他值应保持默认
        assert manager.config.min_detection_confidence == 0.5
    
    def test_update_from_dict_invalid_values(self):
        """测试从字典更新无效值"""
        manager = ConfigManager()
        
        update_dict = {
            'target_fps': -10
        }
        
        with pytest.raises(ValueError):
            manager.update_from_dict(update_dict)
    
    def test_get_info(self):
        """测试获取配置信息"""
        manager = ConfigManager()
        manager.set('video_source', 'test.mp4')
        
        info = manager.get_info()
        
        assert 'config_path' in info
        assert 'config' in info
        assert isinstance(info['config'], dict)
        assert info['config']['video_source'] == 'test.mp4'


class TestConfigManagerIntegration:
    """测试配置管理器集成场景"""
    
    def test_full_workflow(self):
        """测试完整工作流：创建、修改、保存、加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 1. 创建配置管理器
            manager1 = ConfigManager()
            
            # 2. 修改配置
            manager1.set('video_source', 'workflow_test.mp4')
            manager1.set('target_fps', 50)
            manager1.set('show_landmarks', False)
            
            # 3. 保存配置
            manager1.save(temp_path)
            
            # 4. 创建新的管理器并加载
            manager2 = ConfigManager(temp_path)
            
            # 5. 验证配置正确加载
            assert manager2.config.video_source == 'workflow_test.mp4'
            assert manager2.config.target_fps == 50
            assert manager2.config.show_landmarks is False
        finally:
            Path(temp_path).unlink()
    
    def test_partial_config_update(self):
        """测试部分配置更新"""
        manager = ConfigManager()
        
        # 只更新部分配置
        manager.update_from_dict({
            'video_source': 'partial.mp4',
            'show_bbox': False
        })
        
        # 验证更新的值
        assert manager.config.video_source == 'partial.mp4'
        assert manager.config.show_bbox is False
        
        # 验证未更新的值保持默认
        assert manager.config.target_fps == 30
        assert manager.config.show_landmarks is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

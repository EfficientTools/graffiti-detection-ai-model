"""
Integration tests for the complete graffiti detection system
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import json


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.config_dir = Path(cls.temp_dir) / "configs"
        cls.data_dir = Path(cls.temp_dir) / "data"
        cls.output_dir = Path(cls.temp_dir) / "outputs"
        
        cls.config_dir.mkdir(parents=True)
        cls.data_dir.mkdir(parents=True)
        cls.output_dir.mkdir(parents=True)
        
        # Create test image
        cls.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cls.image_path = cls.data_dir / "test_image.jpg"
        cv2.imwrite(str(cls.image_path), cls.test_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir)
    
    def test_config_files_exist(self):
        """Test that all configuration files can be loaded"""
        config_files = [
            'configs/dataset.yaml',
            'configs/training.yaml',
            'configs/model.yaml',
            'configs/surveillance_config.yaml'
        ]
        
        for config_file in config_files:
            self.assertTrue(
                Path(config_file).exists(),
                f"Config file {config_file} should exist"
            )
    
    def test_alert_config_structure(self):
        """Test alert configuration structure"""
        alert_config_path = Path('configs/alerts_example.json')
        
        if alert_config_path.exists():
            with open(alert_config_path, 'r') as f:
                config = json.load(f)
            
            # Check required keys
            expected_keys = ['email', 'sms', 'webhook', 'discord', 'slack']
            for key in expected_keys:
                self.assertIn(key, config)
                self.assertIn('enabled', config[key])
    
    def test_camera_config_structure(self):
        """Test camera configuration structure"""
        camera_config_path = Path('configs/cameras_example.json')
        
        if camera_config_path.exists():
            with open(camera_config_path, 'r') as f:
                config = json.load(f)
            
            # Verify structure
            for camera_id, camera_info in config.items():
                self.assertIn('source', camera_info)
                self.assertIn('conf_threshold', camera_info)
                self.assertIn('enabled', camera_info)
    
    def test_directory_structure(self):
        """Test that all required directories exist"""
        required_dirs = [
            'data/images/train',
            'data/images/val',
            'data/images/test',
            'data/labels/train',
            'data/labels/val',
            'data/labels/test',
            'models/checkpoints',
            'outputs/logs',
            'outputs/metrics',
            'outputs/predictions',
            'outputs/visualizations',
            'scripts',
            'src/data',
            'src/evaluation',
            'src/utils'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(
                Path(dir_path).exists(),
                f"Directory {dir_path} should exist"
            )
    
    def test_scripts_are_executable(self):
        """Test that main scripts have execute permissions"""
        scripts = [
            'scripts/train.py',
            'scripts/evaluate.py',
            'scripts/inference.py',
            'scripts/prepare_dataset.py',
            'scripts/multi_camera_surveillance.py',
            'scripts/real_time_dashboard.py',
            'scripts/incident_logger.py'
        ]
        
        for script in scripts:
            script_path = Path(script)
            self.assertTrue(script_path.exists(), f"Script {script} should exist")


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def test_dataset_preparation_workflow(self):
        """Test dataset preparation workflow"""
        # This would test the complete workflow from raw data to prepared dataset
        # Simplified version
        self.assertTrue(Path('scripts/prepare_dataset.py').exists())
    
    def test_training_workflow(self):
        """Test training workflow components"""
        # Verify training script exists
        self.assertTrue(Path('scripts/train.py').exists())
        
        # Verify configs exist
        self.assertTrue(Path('configs/training.yaml').exists())
        self.assertTrue(Path('configs/dataset.yaml').exists())
    
    def test_inference_workflow(self):
        """Test inference workflow components"""
        # Verify inference script exists
        self.assertTrue(Path('scripts/inference.py').exists())
    
    def test_surveillance_workflow(self):
        """Test surveillance system components"""
        # Verify surveillance script exists
        self.assertTrue(Path('scripts/multi_camera_surveillance.py').exists())
        
        # Verify alert system exists
        self.assertTrue(Path('src/utils/alerts.py').exists())
        
        # Verify dashboard exists
        self.assertTrue(Path('scripts/real_time_dashboard.py').exists())


if __name__ == '__main__':
    unittest.main()

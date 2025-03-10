import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from pipeline.pipeline_manager import PipelineManager
from monitoring.monitor_manager import MonitorManager

class TestSystemIntegration(unittest.TestCase):
    """Full system integration tests"""
    
    def setUp(self):
        self.test_data = self._load_test_data()
        self.pipeline = PipelineManager(config={'test_mode': True})
        self.monitor = MonitorManager(settings={'enabled': True})
        
    def test_complete_workflow(self):
        """Test entire system workflow"""
        # Test configuration
        self.assertTrue(self._verify_config())
        
        # Test data pipeline
        data = self.pipeline._fetch_data('AAPL')
        self.assertIsNotNone(data)
        
        # Test monitoring
        metrics = self.monitor.get_metrics()
        self.assertTrue('execution_time' in metrics)
        
        # Test result storage
        results = self.pipeline.process_ticker('AAPL')
        self.assertTrue(Path('reports').exists())

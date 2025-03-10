import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
import numpy as np

class TestPlatformIntegration(unittest.TestCase):
    """Complete platform integration tests"""
    
    def setUp(self):
        self.test_data = self._generate_test_data()
        self.config = self._load_test_config()
    
    def test_full_pipeline(self):
        """Test complete prediction pipeline"""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = self.test_data
            
            # Test data layer
            self.assertTrue(self._verify_data_layer())
            
            # Test prediction generation
            predictions = self._run_predictions(['AAPL'])
            self.assertIsNotNone(predictions)
            
            # Verify reports
            report_file = Path('reports/latest_predictions.json')
            self.assertTrue(report_file.exists())
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        with patch('yfinance.download', side_effect=Exception("Test error")):
            with self.assertLogs(level='ERROR') as log:
                self._run_predictions(['INVALID'])
                self.assertIn("Error handling activated", log.output[0])

    def test_monitoring_integration(self):
        """Test monitoring system integration"""
        with self.assertLogs(level='INFO') as log:
            self._run_with_monitoring(['AAPL'])
            self.assertIn("Performance metrics collected", log.output)

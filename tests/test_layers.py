import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

class TestDataLayer(unittest.TestCase):
    def setUp(self):
        self.data_manager = DataManager()
        self.test_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })

    def test_data_fetching(self):
        """Test data fetching functionality"""
        data = self.data_manager.get_data('AAPL', days=30)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_data_preprocessing(self):
        """Test data preprocessing"""
        clean_data = self.data_manager.preprocess(self.test_data)
        self.assertFalse(clean_data.isnull().any().any())

class TestPredictionLayer(unittest.TestCase):
    def setUp(self):
        self.pipeline = PipelineManager()
        self.test_prices = np.random.randn(100)

    def test_indicator_calculation(self):
        """Test technical indicator calculation"""
        indicators = self.pipeline.calculate_indicators(self.test_prices)
        self.assertIsNotNone(indicators.get('sma'))
        self.assertIsNotNone(indicators.get('macd'))

class TestComplete(unittest.TestCase):
    """Complete system test suite"""
    
    def setUp(self):
        self.config = self._load_test_config()
        self.pipeline = PipelineManager(self.config)
        self.test_data = self._generate_test_data()

    def test_end_to_end(self):
        """Test complete prediction pipeline"""
        # Test data layer
        data = self.pipeline._fetch_data('AAPL')
        self.assertIsNotNone(data)
        self.assertTrue('Close' in data.columns)

        # Test prediction layer
        result = self.pipeline.process_ticker('AAPL')
        self.assertIsNotNone(result)
        self.assertTrue('prediction' in result)

        # Test indicators
        self.assertTrue('sma' in result['indicators'])
        self.assertTrue('macd' in result['indicators'])

        # Verify output format
        self.assertIsInstance(result['prediction'], float)
        
    def test_error_handling(self):
        """Test error handling and recovery"""
        with patch('yfinance.download', side_effect=Exception("Test error")):
            result = self.pipeline.process_ticker('INVALID')
            self.assertTrue('error' in result)

import unittest
import numpy as np
from predictors.sticker_fusionPredictor import sticker_fusionPredictor
from unittest.mock import Mock, patch

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.dataManager = Mock()
        self.predictor = sticker_fusionPredictor(self.dataManager, 'AAPL', 'test')
        self.sample_data = np.random.randn(100)
        
    def test_indicator_calculation(self):
        """Test individual technical indicators"""
        indicators = self.predictor._calculate_indicators(tuple(self.sample_data))
        self.assertIsNotNone(indicators.get('sma'))
        self.assertIsNotNone(indicators.get('rsi'))
        
    def test_caching_mechanism(self):
        """Test prediction caching"""
        self.dataManager.globalDatasource.get.return_value = self.sample_data
        
        # First call
        pred1 = self.predictor.getPrediction('2023-01-01', 'AAPL')
        # Second call should use cache
        pred2 = self.predictor.getPrediction('2023-01-01', 'AAPL')
        
        self.assertEqual(pred1, pred2)
        self.assertEqual(self.dataManager.globalDatasource.get.call_count, 1)
        
    @patch('numpy.random.randn')
    def test_prediction_stability(self, mock_randn):
        """Test prediction stability with controlled random data"""
        mock_randn.return_value = np.ones(100)
        self.dataManager.globalDatasource.get.return_value = np.ones(100)
        
        predictions = [self.predictor.getPrediction('2023-01-01', 'AAPL') 
                      for _ in range(10)]
        
        # Check stability
        self.assertTrue(all(p == predictions[0] for p in predictions))
        
    def test_prediction_buffer(self):
        """Test prediction buffer functionality"""
        self.dataManager.globalDatasource.get.return_value = self.sample_data
        
        # Fill buffer
        for i in range(self.predictor.max_buffer_size + 10):
            day = f"2023-01-{i:02d}"
            self.predictor.getPrediction(day, 'AAPL')
            
        # Check buffer size
        self.assertLessEqual(
            len(self.predictor.prediction_buffer), 
            self.predictor.max_buffer_size
        )

    def test_confidence_calculation(self):
        """Test prediction confidence calculation"""
        confidence = self.predictor._calculate_confidence({
            'sma': np.ones(100),
            'rsi': np.ones(100) * 70,  # Strong signal
            'macd': np.ones(100) * 0.5
        })
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

if __name__ == '__main__':
    unittest.main()

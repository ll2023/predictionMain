from typing import Dict, Any
import logging
from pathlib import Path
import unittest

class ValidationPlan:
    """Systematic validation of all system layers"""
    
    LAYERS = {
        'data': {
            'components': ['DataManager', 'DataLoader'],
            'dependencies': ['yfinance', 'pandas'],
            'test_files': ['test_data_manager.py', 'test_data_loader.py']
        },
        'prediction': {
            'components': ['PipelineManager', 'Indicators'],
            'dependencies': ['talib', 'numpy'],
            'test_files': ['test_pipeline.py', 'test_indicators.py']
        },
        'monitoring': {
            'components': ['PerformanceMonitor', 'HealthCheck'],
            'dependencies': ['psutil', 'logging'],
            'test_files': ['test_monitoring.py']
        },
        'reporting': {
            'components': ['ReportManager'],
            'dependencies': ['matplotlib', 'seaborn'],
            'test_files': ['test_reporting.py']
        }
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}

    def validate_layer(self, layer: str) -> bool:
        """Validate a specific layer"""
        try:
            config = self.LAYERS[layer]
            
            # Check dependencies
            for dep in config['dependencies']:
                if not self._verify_dependency(dep):
                    return False
                    
            # Run layer tests
            test_results = self._run_layer_tests(layer)
            self.results[layer] = test_results
            
            return all(test_results.values())
            
        except Exception as e:
            self.logger.error(f"Layer validation failed: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        return {
            layer: {
                'status': 'passed' if all(results.values()) else 'failed',
                'details': results
            }
            for layer, results in self.results.items()
        }

class LayerValidation:
    """Layer-specific validation tests"""
    
    def test_data_layer(self):
        """Test data layer functionality"""
        try:
            # Test data access
            result = self.data_manager.get_data('AAPL', days=30)
            assert len(result) > 0, "No data retrieved"
            
            # Test data preprocessing
            clean_data = self.data_manager.preprocess(result)
            assert not clean_data.isnull().any().any(), "Data contains null values"
            
            return True
        except Exception as e:
            self.logger.error(f"Data layer test failed: {e}")
            return False

    def test_prediction_layer(self):
        """Test prediction layer functionality"""
        try:
            # Test indicator calculation
            prices = self.get_test_data()
            indicators = self.pipeline.calculate_indicators(prices)
            assert all(ind is not None for ind in indicators.values()), "Invalid indicators"
            
            # Test prediction generation
            prediction = self.pipeline.generate_prediction(indicators)
            assert isinstance(prediction, (int, float)), "Invalid prediction type"
            
            return True
        except Exception as e:
            self.logger.error(f"Prediction layer test failed: {e}")
            return False

from typing import Dict, List, Any
import logging
from pathlib import Path
from dataclasses import dataclass
import yaml

@dataclass
class ValidationReport:
    layer: str
    status: bool
    messages: List[str]
    metrics: Dict[str, Any]

class PlatformValidator:
    """Enhanced platform validation with detailed reporting"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.validation_results = []
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("platform_validator")
        # ...existing code...
        return logger

    def validate_all(self) -> Dict[str, ValidationReport]:
        """Comprehensive platform validation"""
        layers = {
            'data': self._validate_data_layer,
            'prediction': self._validate_prediction_layer,
            'monitoring': self._validate_monitoring_layer,
            'reporting': self._validate_reporting_layer
        }
        
        results = {}
        for layer, validator in layers.items():
            try:
                self.logger.info(f"Validating {layer} layer...")
                status, messages, metrics = validator()
                results[layer] = ValidationReport(layer, status, messages, metrics)
            except Exception as e:
                self.logger.error(f"Validation failed for {layer}: {e}")
                results[layer] = ValidationReport(layer, False, [str(e)], {})
                
        self._generate_validation_report(results)
        return results

    def _validate_data_layer(self) -> tuple:
        """Validate data layer functionality"""
        messages = []
        metrics = {}
        
        # Test data access
        try:
            import yfinance as yf
            data = yf.download("AAPL", period="1d")
            messages.append("✓ Data source connection successful")
            metrics['data_points'] = len(data)
        except Exception as e:
            messages.append(f"✗ Data source error: {e}")
            return False, messages, metrics

        # ...more specific checks...
        return True, messages, metrics

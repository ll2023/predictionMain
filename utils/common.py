from typing import Dict, List, Any
import pandas as pd
import numpy as np
from functools import lru_cache

class PredictionUtils:
    @staticmethod
    @lru_cache(maxsize=100)
    def calculate_returns(prices: tuple) -> np.ndarray:
        """Calculate returns from price series"""
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def validate_prediction(pred: float, confidence: float, threshold: float = 0.6) -> bool:
        """Validate prediction based on confidence"""
        return abs(pred) > 0 and confidence >= threshold
    
    @staticmethod
    def format_prediction_output(predictions: Dict[str, float]) -> pd.DataFrame:
        """Format predictions for reporting"""
        return pd.DataFrame({
            'ticker': list(predictions.keys()),
            'prediction': list(predictions.values())
        })

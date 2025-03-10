from typing import Any, Optional
import pandas as pd
import numpy as np

class InputValidator:
    @staticmethod
    def validate_price_data(data: Any) -> bool:
        """Validate price data input"""
        if not isinstance(data, (np.ndarray, pd.Series)):
            return False
        if len(data) < 2:  # Minimum length check
            return False
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False
        return True

    @staticmethod
    def validate_indicators(indicators: Dict[str, Any]) -> bool:
        """Validate technical indicators"""
        required = {'sma', 'rsi', 'macd'}
        return all(k in indicators for k in required)

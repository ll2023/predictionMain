import numpy as np
import talib
from typing import Dict, Any, Optional

class AdvancedIndicators:
    """Advanced technical indicators for stock analysis"""
    
    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    @staticmethod
    def calculate_momentum_indicators(close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate momentum-based indicators"""
        return {
            'roc': talib.ROC(close, timeperiod=10),
            'mom': talib.MOM(close, timeperiod=10),
            'adx': talib.ADX(high, low, close, timeperiod=14),
            'cci': talib.CCI(high, low, close, timeperiod=14)
        }

    @staticmethod
    def calculate_volatility_indicators(close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volatility-based indicators"""
        return {
            'atr': talib.ATR(high, low, close, timeperiod=14),
            'natr': talib.NATR(high, low, close, timeperiod=14),
            'trange': talib.TRANGE(high, low, close)
        }

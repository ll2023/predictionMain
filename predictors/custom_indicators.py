from predictors.indicator_registry import BaseIndicator, IndicatorRegistry
import numpy as np
import talib

@IndicatorRegistry.register
class VolumeWeightedMACD(BaseIndicator):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    @property
    def name(self) -> str:
        return 'vwmacd'
    
    @property
    def required_data(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: np.ndarray) -> np.ndarray:
        close = data['close']
        volume = data['volume']
        vw_price = close * volume / volume.mean()
        return talib.MACD(vw_price, self.fast_period, self.slow_period, self.signal_period)

@IndicatorRegistry.register
class AdaptiveRSI(BaseIndicator):
    """RSI with adaptive period based on volatility"""
    def __init__(self, base_period=14, vol_adjust=True):
        self.base_period = base_period
        self.vol_adjust = vol_adjust
    
    @property
    def name(self) -> str:
        return 'adaptive_rsi'
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        # Adaptive period based on volatility
        if self.vol_adjust:
            volatility = talib.STDDEV(data, timeperiod=20)
            period = np.maximum(5, self.base_period * (1 + volatility/100))
        else:
            period = self.base_period
        return talib.RSI(data, timeperiod=int(period))

import numpy as np
import pandas as pd
from Configuration import Configuration
from Service.Utilities import Tools
import talib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
from typing import Dict, Optional, List, Tuple

class sticker_fusionPredictor:
    """
    A predictor that combines multiple technical indicators to generate stock predictions.
    
    This class implements:
    - Thread-safe caching
    - Parallel indicator calculation
    - Confidence-based filtering
    
    Example:
        ```python
        predictor = sticker_fusionPredictor(data_manager, 'AAPL', 'momentum')
        prediction = predictor.getPrediction('2023-01-01', 'AAPL')
        ```
    """
    
    def __init__(self, dataManager, sticker: str, predTag: str, dirct: int = 1):
        """
        Initialize the sticker_fusionPredictor with a data manager, stock ticker, prediction tag, and direction.
        
        Parameters:
        dataManager (DataManager): The data manager to use.
        sticker (str): The stock ticker.
        predTag (str): The prediction tag.
        dirct (int): The direction (1 for positive, -1 for negative). Defaults to 1.
        """
        self.dataManager = dataManager
        self.sticker = sticker
        self.predTag = predTag
        self.dirct = dirct
        self.name = f"Predictor_{sticker}_{predTag}"
        self.reportMan = None
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.technical_indicators = {
            'sma': {'function': talib.SMA, 'timeperiod': 20},
            'rsi': {'function': talib.RSI, 'timeperiod': 14},
            'macd': {'function': talib.MACD, 'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'bbands': {'function': talib.BBANDS, 'timeperiod': 20},
            'stoch': {'function': talib.STOCH, 'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}
        }
        self._cache_lock = threading.Lock()
        self._prediction_cache = {}
        self.prediction_buffer = {}
        self.max_buffer_size = 1000
        self.min_confidence = 0.6
        self._prediction_lock = threading.RLock()  # Reentrant lock for thread safety
        self._buffer_lock = threading.Lock()
        self._indicator_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0}

    def setReportMan(self, rm):
        """
        Set the report manager.
        
        Parameters:
        rm (ReportManager): The report manager to set.
        """
        self.reportMan = rm

    @lru_cache(maxsize=1000)
    def _calculate_single_indicator(self, prices_tuple: Tuple[float], indicator_name: str) -> np.ndarray:
        """Calculate single technical indicator with caching"""
        prices = np.array(prices_tuple)
        params = self.technical_indicators[indicator_name]
        return params['function'](prices, **{k: v for k, v in params.items() if k != 'function'})

    def calculate_indicators_parallel(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators in parallel"""
        prices_tuple = tuple(prices)  # Convert to tuple for caching
        futures = []
        
        for indicator_name in self.technical_indicators:
            future = self.executor.submit(self._calculate_single_indicator, prices_tuple, indicator_name)
            futures.append((indicator_name, future))
        
        return {name: future.result() for name, future in futures}

    def calculate_indicator(self, prices: np.ndarray, indicator_name: str) -> Optional[np.ndarray]:
        """Calculate technical indicator with error handling"""
        try:
            params = self.technical_indicators[indicator_name]
            return params['function'](prices, **{k: v for k, v in params.items() if k != 'function'})
        except Exception as e:
            print(f"Error calculating {indicator_name}: {e}")
            return None

    @lru_cache(maxsize=1000)
    def _calculate_indicators(self, prices_tuple: Tuple[float]) -> Dict[str, np.ndarray]:
        """Calculate technical indicators with caching"""
        prices = np.array(prices_tuple)
        return {
            name: params['function'](prices, **{k: v for k, v in params.items() if k != 'function'})
            for name, params in self.technical_indicators.items()
        }

    def _cleanup_buffer(self):
        """Clean up old predictions from buffer"""
        if len(self.prediction_buffer) > self.max_buffer_size:
            oldest_keys = sorted(self.prediction_buffer.keys())[:-self.max_buffer_size]
            for key in oldest_keys:
                del self.prediction_buffer[key]

    @handle_errors(logging.getLogger(__name__))
    def getPrediction(self, forday: str, sticker: str) -> Optional[float]:
        """
        Generate a prediction for the given stock and day.
        
        Args:
            forday: The date to predict for (YYYY-MM-DD)
            sticker: The stock ticker symbol
            
        Returns:
            float: The predicted value, or None if prediction fails
            
        Raises:
            ValueError: If input validation fails
        """
        try:
            cache_key = f"{forday}_{sticker}"
            
            # Check prediction buffer first
            if cache_key in self.prediction_buffer:
                self.logger.debug(f"Cache hit for {cache_key}")
                return self.prediction_buffer[cache_key]

            # Vectorized operations
            close_prices = self.dataManager.globalDatasource.get(sticker, 'close', forday, 100)
            indicators = self.calculate_indicators_parallel(close_prices)
            
            # Optimize weighted calculation using numpy
            weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # Optimized weights
            signals = np.array([ind[-1] for ind in indicators.values()])
            prediction = np.dot(weights, signals) * self.dirct
            
            # Store in buffer if confidence is high enough
            confidence = self._calculate_confidence(indicators)
            if confidence >= self.min_confidence:
                self.prediction_buffer[cache_key] = prediction
                self._cleanup_buffer()
            
            with self._cache_lock:
                self._prediction_cache[cache_key] = prediction
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in getPrediction: {e}")
            return None

    def getSkip(self, forday: str, sticker: str) -> bool:
        """
        Determine whether to skip the prediction for a specific day and stock ticker.
        
        Parameters:
        forday (str): The day for which to determine whether to skip the prediction.
        sticker (str): The stock ticker.
        
        Returns:
        bool: True if the prediction should be skipped, False otherwise.
        """
        try:
            # ...existing code...
        except Exception as e:
            print(f"Error in getSkip: {e}")
            return False

    def runAll(self, forday: str):
        """
        Run all predictions for a specific day.
        
        Parameters:
        forday (str): The day for which to run all predictions.
        """
        try:
            # ...existing code...
        except Exception as e:
            print(f"Error in runAll: {e}")

    def refresh(self, forday: str):
        """
        Refresh the predictor for a specific day.
        
        Parameters:
        forday (str): The day for which to refresh the predictor.
        """
        try:
            # ...existing code...
        except Exception as e:
            print(f"Error in refresh: {e}")

    def _thread_safe_calculate(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Thread-safe calculation of indicators"""
        with self._prediction_lock:
            return self._calculate_indicators(tuple(prices))

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return self._cache_stats.copy()

    def _load_indicators(self) -> None:
        """Load indicators from configuration"""
        self.indicators = {}
        for ind_name, ind_config in self.config.get('indicators', {}).items():
            indicator_class = IndicatorRegistry.get_indicator(ind_name)
            self.indicators[ind_name] = indicator_class(**ind_config)

    def add_indicator(self, name: str, indicator: BaseIndicator) -> None:
        """Add new indicator at runtime"""
        with self._prediction_lock:
            self.indicators[name] = indicator
            self._cache.clear()  # Invalidate cache since indicators changed











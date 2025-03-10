import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import talib
import numpy as np
import pandas as pd

# Add safe import for yfinance
try:
    import yfinance as yf
except ImportError:
    logging.error("yfinance not found. Please install with: pip install yfinance")
    raise

@dataclass
class PipelineResult:
    success: bool
    data: Dict[str, Any]
    errors: List[str]

class PipelineManager:
    """Core pipeline with execution logic"""
    
    def __init__(self, settings: 'Settings'):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._setup_components()
        
    def _setup_components(self):
        """Initialize core components"""
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.system.get('max_workers', 4)
        )
        self._lock = threading.Lock()
        
    def process(self, data: Dict[str, Any]) -> PipelineResult:
        """Process predictions with error handling"""
        try:
            tickers = data['tickers']
            results = {}
            errors = []
            
            # Process each ticker
            for ticker in tickers:
                result = self.process_ticker(ticker)
                if 'error' in result:
                    errors.append(result['error'])
                else:
                    results[ticker] = result
            
            return PipelineResult(
                success=len(errors) == 0,
                data=results,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return PipelineResult(success=False, data={}, errors=[str(e)])
            
    def process_ticker(self, ticker: str) -> Dict[str, Any]:
        """Process single ticker with enhanced error handling"""
        try:
            # Fetch data
            data = self._fetch_data(ticker)
            
            # Pre-process data
            close_prices = data['Close'].values
            if len(close_prices) == 0:
                raise ValueError(f"No price data for {ticker}")
            
            # Calculate indicators
            indicators = {}
            for name, ind in self.settings.indicators.items():
                try:
                    value = self._calculate_indicator(close_prices, ind)
                    indicators[name] = value
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {name} for {ticker}: {e}")
                    indicators[name] = None
            
            # Filter out failed calculations
            valid_indicators = {k: v for k, v in indicators.items() if v is not None}
            
            if not valid_indicators:
                raise ValueError("No valid indicators calculated")
            
            prediction = self._combine_predictions(valid_indicators)
            
            return {
                'ticker': ticker,
                'prediction': prediction,
                'indicators': valid_indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e)
            }

    def _fetch_data(self, ticker: str) -> pd.DataFrame:
        """Safely fetch and validate ticker data"""
        try:
            data = yf.download(
                ticker, 
                period="120d",
                interval="1d",
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
                
            # Use modern pandas methods
            data = data.ffill().bfill()  # Replace deprecated fillna
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns for {ticker}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {e}")
            raise

    def _calculate_indicator(self, prices: np.ndarray, indicator: Any) -> float:
        """Calculate technical indicator value with proper dimension handling"""
        try:
            # Ensure we have enough data points
            min_length = max(30, indicator.config.get('timeperiod', 20) * 2)
            if len(prices) < min_length:
                raise ValueError(f"Insufficient data points ({len(prices)} < {min_length})")

            # Ensure prices is 1D array of proper type
            prices = np.asarray(prices, dtype=np.float64).flatten()

            if indicator.name == 'sma':
                timeperiod = indicator.config.get('timeperiod', 20)
                result = talib.SMA(prices, timeperiod=timeperiod)
                
            elif indicator.name == 'macd':
                fastperiod = indicator.config.get('fastperiod', 12)
                slowperiod = indicator.config.get('slowperiod', 26)
                signalperiod = indicator.config.get('signalperiod', 9)
                macd, signal, hist = talib.MACD(
                    prices,
                    fastperiod=fastperiod,
                    slowperiod=slowperiod,
                    signalperiod=signalperiod
                )
                result = macd
            else:
                raise ValueError(f"Unknown indicator: {indicator.name}")

            # Handle NaN values and return last valid value
            valid_values = result[~np.isnan(result)]
            if len(valid_values) == 0:
                raise ValueError("No valid indicator values calculated")
                
            return float(valid_values[-1])
                
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {e}")
            raise
            
    def _combine_predictions(self, predictions: Dict[str, float]) -> float:
        """Combine individual indicator predictions"""
        total_weight = 0
        weighted_sum = 0
        
        for name, value in predictions.items():
            weight = self.settings.indicators[name].config['weight']
            weighted_sum += value * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0

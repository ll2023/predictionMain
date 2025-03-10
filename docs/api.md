# API Documentation

## Predictor Module

### sticker_fusionPredictor

The main predictor class that combines multiple technical indicators to generate predictions.

#### Methods

##### getPrediction(forday: str, sticker: str) -> float
Generates a prediction for a specific stock on a given day.

```python
predictor = sticker_fusionPredictor(dataManager, 'AAPL', 'test')
prediction = predictor.getPrediction('2023-01-01', 'AAPL')
```

##### calculate_indicators(prices: np.ndarray) -> Dict[str, np.ndarray]
Calculates technical indicators for given price data.

#### Technical Indicators

The predictor uses the following technical indicators:
- SMA (Simple Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator

## Performance Optimization

### Caching
The predictor implements two levels of caching:
1. Raw indicator calculations using `@lru_cache`
2. Final predictions using thread-safe dictionary cache

### Parallel Processing
Indicator calculations are performed in parallel using ThreadPoolExecutor

## Pipeline Components

### PipelineManager

The core pipeline orchestrator that manages the prediction workflow.

```python
manager = PipelineManager(config)
results = manager.run_pipeline(tickers)
```

#### Configuration Options

| Parameter | Type | Description |
|-----------|------|-------------|
| max_workers | int | Number of parallel workers |
| batch_size | int | Size of processing batches |
| cache_size | int | Maximum cache entries |

#### Error Handling

The pipeline implements smart error recovery:
1. Automatic retry for transient failures
2. Cache clearing for memory issues
3. Batch size adjustment for performance

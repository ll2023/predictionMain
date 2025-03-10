# Prediction Engine Documentation

## Overview
This system provides stock market predictions using multiple technical indicators and machine learning algorithms.

## Components
1. **Predictor**: Generates predictions using technical analysis
2. **Fuser**: Combines multiple predictions
3. **ReportManager**: Generates reports and visualizations

## Usage
```python
from Fuser import Fuser
from ReportManager import ReportManager

# Initialize
fuser = Fuser(dataManager)
reportManager = ReportManager(fuser)

# Run predictions
fuser.runseq()
```

## Configuration
All configurations are managed through `config.yaml`:
```yaml
technical_indicators:
  sma:
    timeperiod: 20
  rsi:
    timeperiod: 14
```

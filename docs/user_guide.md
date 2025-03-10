# User Guide

## Quick Start

```bash
# Run predictions
python cli.py run config.yaml -t AAPL -t GOOGL

# Generate reports
python cli.py report --format html
```

## Configuration

### Technical Indicators
```yaml
technical_indicators:
  sma:
    timeperiod: 20
    weight: 0.2
```

### Performance Tuning
- Adjust `max_workers` for parallel processing
- Configure cache sizes
- Set monitoring thresholds

## Troubleshooting

Common issues and solutions:
1. Memory usage too high
   - Reduce cache sizes
   - Decrease parallel workers
2. Prediction accuracy low
   - Adjust indicator weights
   - Tune confidence thresholds

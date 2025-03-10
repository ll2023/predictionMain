from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AlgorithmConfig:
    """Available prediction algorithms"""
    
    ALGORITHMS = {
        'technical': {
            'sma': {
                'timeperiod': 20,
                'weight': 0.3
            },
            'macd': {
                'fastperiod': 12,
                'slowperiod': 26,
                'signalperiod': 9,
                'weight': 0.3
            },
            'rsi': {
                'timeperiod': 14,
                'weight': 0.2
            },
            'bbands': {
                'timeperiod': 20,
                'weight': 0.2
            }
        },
        'ml': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1
            }
        },
        'deep_learning': {
            'lstm': {
                'units': 50,
                'layers': 2
            }
        }
    }

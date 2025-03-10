import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

class PerformanceAnalyzer:
    def __init__(self, predictions_file: str, results_file: str):
        self.predictions = pd.read_csv(predictions_file)
        self.results = pd.read_csv(results_file)
        
    def analyze_performance(self) -> PerformanceMetrics:
        """Analyze prediction performance"""
        # Calculate metrics
        metrics = PerformanceMetrics(
            accuracy=self._calculate_accuracy(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            max_drawdown=self._calculate_max_drawdown(),
            win_rate=self._calculate_win_rate(),
            profit_factor=self._calculate_profit_factor()
        )
        
        return metrics

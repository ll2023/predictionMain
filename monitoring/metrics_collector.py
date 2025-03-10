import time
import numpy as np
from typing import Dict, Any
import logging
from dataclasses import dataclass
from collections import deque

@dataclass
class PredictionMetrics:
    accuracy: float
    latency: float
    memory_usage: int
    cache_hits: int
    cache_misses: int

class MetricsCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = {}
        
    def record_prediction(self, prediction: float, actual: float, 
                         latency: float, cache_hit: bool) -> None:
        """Record metrics for a single prediction"""
        metrics = PredictionMetrics(
            accuracy=1.0 if np.sign(prediction) == np.sign(actual) else 0.0,
            latency=latency,
            memory_usage=0,  # Will be filled by monitor
            cache_hits=1 if cache_hit else 0,
            cache_misses=0 if cache_hit else 1
        )
        self.metrics_history.append(metrics)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics"""
        if not self.metrics_history:
            return {}
            
        metrics_array = np.array([(m.accuracy, m.latency) for m in self.metrics_history])
        return {
            'accuracy_mean': np.mean(metrics_array[:, 0]),
            'accuracy_std': np.std(metrics_array[:, 0]),
            'latency_mean': np.mean(metrics_array[:, 1]),
            'latency_p95': np.percentile(metrics_array[:, 1], 95),
            'cache_hit_rate': sum(m.cache_hits for m in self.metrics_history) / len(self.metrics_history)
        }

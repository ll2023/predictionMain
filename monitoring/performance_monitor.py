import time
import psutil
import logging
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}
        
    def start_monitoring(self, task_name: str) -> None:
        """Start monitoring a task"""
        self.metrics[task_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss
        }
        
    def end_monitoring(self, task_name: str) -> Dict[str, float]:
        """End monitoring and return metrics"""
        if task_name not in self.metrics:
            return {}
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = {
            'duration': end_time - self.metrics[task_name]['start_time'],
            'memory_usage': end_memory - self.metrics[task_name]['start_memory']
        }
        
        self.logger.info(f"Task {task_name} completed: {metrics}")
        return metrics

    def monitor_prediction_accuracy(self, predictions: Dict[str, float], actuals: Dict[str, float]) -> Dict[str, float]:
        """Monitor prediction accuracy metrics"""
        metrics = {}
        
        for ticker in predictions:
            if ticker in actuals:
                error = abs(predictions[ticker] - actuals[ticker])
                metrics[ticker] = {
                    'absolute_error': error,
                    'percentage_error': (error / actuals[ticker]) * 100 if actuals[ticker] != 0 else float('inf')
                }
        
        self.logger.info(f"Prediction accuracy metrics: {metrics}")
        return metrics
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor system health metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Alert if resources are running low
        if any(v > 90 for v in metrics.values()):
            self.logger.warning(f"System resources running high: {metrics}")
            
        return metrics

import logging
from typing import Dict, Any
import threading
import time
from dataclasses import dataclass

@dataclass
class MonitoringMetrics:
    execution_time: float
    memory_usage: float
    success_rate: float
    error_count: int

class MonitorManager:
    """Centralized monitoring with metrics collection"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._metrics = {}
        self._lock = threading.Lock()

    def start_monitoring(self, component: str) -> None:
        """Start monitoring a component"""
        with self._lock:
            self._metrics[component] = {
                'start_time': time.time(),
                'errors': 0,
                'successes': 0
            }

    def record_error(self, component: str, error: Exception) -> None:
        """Record component error"""
        with self._lock:
            if component in self._metrics:
                self._metrics[component]['errors'] += 1
                self.logger.error(f"{component} error: {error}")

    def get_metrics(self) -> Dict[str, MonitoringMetrics]:
        """Get current monitoring metrics"""
        with self._lock:
            return {
                component: MonitoringMetrics(
                    execution_time=time.time() - metrics['start_time'],
                    memory_usage=self._get_memory_usage(),
                    success_rate=self._calculate_success_rate(metrics),
                    error_count=metrics['errors']
                )
                for component, metrics in self._metrics.items()
            }

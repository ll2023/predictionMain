import logging
import psutil
import threading
from typing import Dict, Any

class RuntimeMonitor:
    """Monitor runtime execution and resource usage"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        
    def start_monitoring(self):
        """Start monitoring thread"""
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                if self._should_alert(metrics):
                    self._handle_threshold_exceeded(metrics)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }

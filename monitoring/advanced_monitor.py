import logging
from typing import Dict, Any
import time
from dataclasses import dataclass
import threading
import psutil

@dataclass
class PredictionStats:
    total_predictions: int
    successful_predictions: int
    cache_hits: int
    average_latency: float

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int

class AdvancedMonitor:
    """Enhanced monitoring with real-time metrics and alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._metrics_history = []
        self._stop_event = threading.Event()
        self._monitoring_thread = None
        self._stats_lock = threading.Lock()
        self._stats = PredictionStats(0, 0, 0, 0.0)
        
    def start_monitoring(self):
        """Start automated monitoring"""
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self._analyze_metrics(metrics)
                self._store_metrics(metrics)
                
                # Alert if thresholds exceeded
                if self._should_alert(metrics):
                    self._send_alert(metrics)
                    
                time.sleep(self.config.get('monitoring_interval', 60))
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                
    def record_prediction(self, success: bool, latency: float, cache_hit: bool):
        """Record prediction statistics thread-safely"""
        with self._stats_lock:
            self._stats.total_predictions += 1
            if success:
                self._stats.successful_predictions += 1
            if cache_hit:
                self._stats.cache_hits += 1
            self._stats.average_latency = (
                (self._stats.average_latency * (self._stats.total_predictions - 1) + latency)
                / self._stats.total_predictions
            )

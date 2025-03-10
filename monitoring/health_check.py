import psutil
import logging
from typing import Dict
import threading
import time

class SystemHealthCheck:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._health_thread = None
        self._stop_event = threading.Event()
        
    def start_monitoring(self):
        """Start continuous health monitoring"""
        def monitor_loop():
            while not self._stop_event.is_set():
                self.check_health()
                time.sleep(60)  # Check every minute
                
        self._health_thread = threading.Thread(target=monitor_loop)
        self._health_thread.start()
        
    def check_health(self) -> Dict[str, float]:
        """Check system health metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_used': psutil.virtual_memory().percent,
            'disk_used': psutil.disk_usage('/').percent,
            'swap_used': psutil.swap_memory().percent
        }
        
        # Alert on high resource usage
        if any(v > 90 for v in metrics.values()):
            self.logger.warning(f"High resource usage detected: {metrics}")
            
        return metrics

    def verify_system_state(self) -> Dict[str, bool]:
        """Comprehensive system verification"""
        try:
            checks = {
                'memory': self._check_memory_usage(),
                'disk': self._check_disk_space(),
                'cpu': self._check_cpu_usage(),
                'network': self._check_network_connectivity(),
                'required_processes': self._check_required_processes()
            }
            
            if not all(checks.values()):
                self.logger.error(f"System checks failed: {checks}")
                
            return checks
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {'error': False}

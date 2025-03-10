import cProfile
import pstats
from functools import wraps
from typing import Dict, Any
import time
import logging

class PerformanceProfiler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles = {}
        
    def profile(self, name: str):
        """Decorator for profiling functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                try:
                    return profiler.runcall(func, *args, **kwargs)
                finally:
                    stats = pstats.Stats(profiler)
                    self.profiles[name] = stats
                    self._log_performance_stats(name, stats)
            return wrapper
        return decorator
        
    def _log_performance_stats(self, name: str, stats: pstats.Stats):
        """Log detailed performance statistics"""
        summary = {
            'total_calls': stats.total_calls,
            'total_time': stats.total_tt,
            'calls_per_second': stats.total_calls / stats.total_tt if stats.total_tt > 0 else 0
        }
        self.logger.info(f"Performance profile for {name}: {summary}")

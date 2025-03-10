from typing import Dict, Any
import logging
import threading

class LogCoordinator:
    """Coordinates logging across all components"""
    
    def __init__(self):
        self.loggers = {}
        self._lock = threading.Lock()
        
    def setup_component_logger(self, component_name: str, log_level: int = logging.INFO) -> logging.Logger:
        """Setup logger for individual component with proper configuration"""
        with self._lock:
            if component_name not in self.loggers:
                logger = logging.getLogger(component_name)
                logger.setLevel(log_level)
                
                # Add file handler
                handler = logging.FileHandler(f"logs/{component_name}.log")
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                
                self.loggers[component_name] = logger
                
            return self.loggers[component_name]

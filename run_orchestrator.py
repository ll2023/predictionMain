import click
import logging
from pathlib import Path
from config.settings import Settings
from validation.validator import Validator
from monitoring.monitor_manager import MonitorManager
from monitoring.health_check import SystemHealthCheck
from monitoring.performance_monitor import PerformanceMonitor

class RunOrchestrator:
    """Enhanced orchestrator with comprehensive monitoring"""
    
    def __init__(self):
        self._setup_logging()
        self.monitor = MonitorManager(settings={'log_level': 'INFO'})
        self.health_check = SystemHealthCheck()
        self.perf_monitor = PerformanceMonitor()
        
    def _setup_logging(self):
        """Setup enhanced logging"""
        self.logger = logging.getLogger(__name__)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Add detailed formatting
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "orchestrator.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def execute(self, config_path: str, tickers: list) -> bool:
        """Execute pipeline with comprehensive monitoring"""
        try:
            with self.perf_monitor.track_execution("full_run"):
                # System health check
                if not self.health_check.verify_system_state():
                    raise SystemError("System health check failed")

                # Load and validate configuration
                settings = Settings.load(config_path)
                self.logger.info(f"Loaded configuration from {config_path}")

                # Start monitoring
                self.monitor.start_monitoring("pipeline")
                
                # Validate components
                validation = Validator.validate_config(settings.__dict__)
                if not validation.valid:
                    self.logger.error(f"Configuration invalid: {validation.errors}")
                    return False

                # Execute pipeline
                success = self._run_pipeline(settings, tickers)
                
                # Generate reports
                self._generate_execution_report()
                
                return success

        except Exception as e:
            self.logger.error(f"Execution failed: {e}", exc_info=True)
            return False

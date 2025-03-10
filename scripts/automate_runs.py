import schedule
import time
from pathlib import Path
import yaml
from typing import Dict
import subprocess
from datetime import datetime

class AutomatedRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.run_logger = RunLogger()
        self.schedule = self._setup_schedule()
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.backoff_factor = 2  # Exponential backoff

    def _setup_schedule(self) -> None:
        """Setup scheduled runs based on configuration"""
        schedules = self.config.get('schedules', {})
        for job_name, schedule_config in schedules.items():
            schedule.every().day.at(schedule_config['time']).do(
                self.run_prediction_job, job_name, schedule_config
            )

    def run_prediction_job(self, job_name: str, config: Dict) -> None:
        try:
            # System health check
            if not self._verify_system_health():
                raise SystemHealthError("System health check failed")
                
            # Resource check
            if not self._verify_resources():
                self._cleanup_resources()
                
            self.run_logger.logger.info(f"Starting job: {job_name}")
            
            result = subprocess.run([
                'python', 'run.py',
                '--config', config['config_file'],
                '--tickers'] + config['tickers'],
                capture_output=True, text=True
            )
            
            self.run_logger.log_run_metrics({
                'job_name': job_name,
                'status': 'success' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'errors': result.stderr,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self._handle_job_failure(job_name, e)

    def _handle_job_failure(self, job_name: str, error: Exception) -> None:
        attempts = self.recovery_attempts.get(job_name, 0)
        if attempts < self.max_recovery_attempts:
            delay = self.backoff_factor ** attempts
            self.logger.warning(f"Scheduling retry for {job_name} in {delay} seconds")
            schedule.every(delay).seconds.do(
                self.run_prediction_job, job_name, self.config
            ).tag(f"recovery_{job_name}")
            self.recovery_attempts[job_name] = attempts + 1
        else:
            self.logger.error(f"Job {job_name} failed permanently after {attempts} retries")

    def verify_and_run(self, config_path: str) -> bool:
        """Verify environment and run predictions"""
        try:
            # Pre-run checks
            verifier = DeploymentVerifier(config_path)
            if not verifier.verify_all():
                self.logger.error("Pre-run verification failed")
                return False
                
            # Initialize monitoring
            self.health_check = SystemHealthCheck()
            self.health_check.start_monitoring()
            
            # Run predictions
            with PerformanceMonitor().track_execution("full_run"):
                self.run_prediction_job("scheduled_run", self.config)
                
            # Verify results
            return self._verify_run_results()
            
        except Exception as e:
            self.logger.error(f"Run failed: {e}")
            return False

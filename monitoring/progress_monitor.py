from typing import Dict, Any
import time
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class ProgressStats:
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    start_time: datetime
    estimated_completion: datetime

class ProgressMonitor:
    def __init__(self, total_tasks: int):
        self.logger = logging.getLogger(__name__)
        self.total_tasks = total_tasks
        self.completed = 0
        self.failed = 0
        self.start_time = datetime.now()
        self.task_times = []

    def update(self, success: bool = True) -> None:
        """Update progress and log status"""
        if success:
            self.completed += 1
        else:
            self.failed += 1
            
        self.task_times.append(time.time())
        self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress and estimates"""
        progress = (self.completed + self.failed) / self.total_tasks
        avg_time = self._calculate_avg_time()
        eta = self._estimate_completion()
        
        self.logger.info(
            f"Progress: {progress:.1%} | "
            f"Completed: {self.completed} | "
            f"Failed: {self.failed} | "
            f"ETA: {eta.strftime('%H:%M:%S')}"
        )

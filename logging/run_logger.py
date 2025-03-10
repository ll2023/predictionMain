import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any

class RunLogger:
    def __init__(self, log_dir: str = "logs/runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"run_{self.run_id}")
        handler = logging.FileHandler(self.log_dir / f"run_{self.run_id}.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def log_run_metrics(self, metrics: Dict[str, Any]):
        """Log run metrics to JSON file"""
        metrics_file = self.log_dir / f"metrics_{self.run_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def log_pipeline_execution(self, pipeline_data: Dict[str, Any]):
        """Enhanced pipeline logging with metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': pipeline_data.get('execution_time'),
            'processed_tickers': len(pipeline_data.get('tickers', [])),
            'success_rate': pipeline_data.get('success_rate'),
            'memory_usage': pipeline_data.get('memory_usage'),
            'predictions': pipeline_data.get('predictions'),
            'errors': pipeline_data.get('errors')
        }
        
        # Save detailed metrics
        self._save_metrics(metrics)
        
        # Generate summary report
        self._generate_summary_report(metrics)
        
        # Alert on issues
        if metrics['success_rate'] < 0.9:
            self._send_alert(metrics)

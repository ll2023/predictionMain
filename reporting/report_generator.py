class ReportGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._load_templates()
        
    def generate_reports(self, data: Dict[str, Any]):
        """Generate comprehensive reports"""
        reports = {
            'execution': self._generate_execution_report(data),
            'performance': self._generate_performance_report(data),
            'predictions': self._generate_prediction_report(data),
            'errors': self._generate_error_report(data)
        }
        
        # Save reports
        self._save_reports(reports)
        
        # Generate alerts if needed
        self._check_and_generate_alerts(reports)

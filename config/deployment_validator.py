from typing import Dict, List
import yaml
import json
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ValidationResult:
    passed: bool
    messages: List[str]

class DeploymentValidator:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.results = []

    def validate_all(self) -> ValidationResult:
        """Run all deployment validations"""
        checks = [
            self._validate_configs(),
            self._validate_permissions(),
            self._validate_dependencies(),
            self._validate_data_sources()
        ]
        
        passed = all(check.passed for check in checks)
        messages = [msg for check in checks for msg in check.messages]
        
        return ValidationResult(passed=passed, messages=messages)

    def generate_validation_report(self) -> str:
        """Generate detailed validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': [{'check': r.check_name, 'passed': r.passed, 
                        'messages': r.messages} for r in self.results]
        }
        
        report_path = Path("logs/validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return str(report_path)

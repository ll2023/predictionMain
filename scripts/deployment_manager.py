import logging
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import yaml
import importlib
import subprocess
import psutil

@dataclass
class DeploymentStatus:
    success: bool
    stage: str
    message: str
    details: Dict[str, Any]

@dataclass
class DeploymentStep:
    name: str
    command: str
    validation: callable
    rollback: callable = None

class DeploymentManager:
    def __init__(self, config_path: str):
        self.logger = self._setup_logging()
        self.config_path = config_path
        self.validation_results = []
        self.health_check = SystemHealthCheck()
        self.verification_steps = [
            self.verify_environment,
            self.verify_dependencies,
            self.verify_configuration,
            self.verify_data_access,
            self.verify_permissions
        ]
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("deployment")
        logger.setLevel(logging.INFO)
        log_file = Path("logs/deployment.log")
        log_file.parent.mkdir(exist_ok=True)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def validate_environment(self) -> DeploymentStatus:
        """Pre-deployment environment validation"""
        try:
            # Verify Python version
            if sys.version_info < (3, 8):
                return DeploymentStatus(False, "env_check", "Python 3.8+ required", {})

            # Verify critical packages
            required_packages = ['numpy', 'pandas', 'talib', 'yaml']
            missing = []
            for package in required_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing.append(package)
            
            if missing:
                return DeploymentStatus(False, "package_check", 
                                     f"Missing packages: {', '.join(missing)}", 
                                     {"missing": missing})

            return DeploymentStatus(True, "env_check", "Environment valid", {})
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            return DeploymentStatus(False, "env_check", str(e), {})

    def validate_configuration(self) -> DeploymentStatus:
        """Validate configuration files"""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            # Validate structure
            required_sections = ['technical_indicators', 'monitoring', 'system']
            missing = [s for s in required_sections if s not in config]
            if missing:
                return DeploymentStatus(False, "config_check", 
                                     f"Missing sections: {missing}", 
                                     {"missing": missing})

            # Validate indicators
            for name, params in config['technical_indicators'].items():
                if 'weight' not in params:
                    return DeploymentStatus(False, "config_check",
                                         f"Missing weight in {name}", 
                                         {"indicator": name})

            return DeploymentStatus(True, "config_check", "Configuration valid", {})
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return DeploymentStatus(False, "config_check", str(e), {})

    def deploy(self) -> bool:
        """Enhanced deployment with health checks"""
        try:
            # Verify Python environment
            if not self._verify_python_env():
                return False
                
            # Verify configuration
            if not self._verify_config():
                return False
                
            # Verify system resources
            if not self._verify_resources():
                return False
                
            # Setup directories
            if not self._setup_directories():
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}", exc_info=True)
            return False

    def _verify_resources(self) -> bool:
        """Verify system resources"""
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            disk = psutil.disk_usage('/')
            
            if mem.percent > 90 or cpu > 90 or disk.percent > 90:
                self.logger.error("Insufficient system resources")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            return False

    def generate_report(self) -> str:
        """Generate deployment report"""
        report_path = Path("logs/deployment_report.yml")
        report = {
            "deployment_status": all(r.success for r in self.validation_results),
            "stages": [
                {
                    "name": r.stage,
                    "success": r.success,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.validation_results
            ]
        }
        
        with open(report_path, 'w') as f:
            yaml.dump(report, f)
            
        return str(report_path)

class EnhancedDeploymentManager:
    """Automated deployment with validation and rollback"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.deployment_steps = self._define_deployment_steps()
        
    def deploy(self) -> bool:
        """Run automated deployment with validation"""
        try:
            # Verify prerequisites
            if not self._verify_prerequisites():
                return False

            # Execute deployment steps
            for step in self.deployment_steps:
                self.logger.info(f"Executing step: {step.name}")
                
                try:
                    # Execute step
                    result = subprocess.run(
                        step.command, 
                        shell=True, 
                        capture_output=True,
                        text=True
                    )
                    
                    # Validate step
                    if not step.validation(result):
                        raise ValueError(f"Validation failed for {step.name}")
                        
                    self.logger.info(f"Successfully completed: {step.name}")
                    
                except Exception as e:
                    self.logger.error(f"Step failed: {step.name}")
                    if step.rollback:
                        self._execute_rollback(step)
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False

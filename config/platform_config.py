from typing import Dict, Any
import yaml
import os
from pathlib import Path
from abc import ABC, abstractmethod

class DataSourceConfig:
    def __init__(self, config: Dict[str, Any]):
        self.type = config.get('type', 'yahoo')
        self.cache_enabled = config.get('cache_enabled', True)
        self.batch_size = config.get('batch_size', 100)
        self.retry_attempts = config.get('retry_attempts', 3)

class PlatformConfig:
    def __init__(self, env: str = 'dev'):
        self.env = env
        self.config = self._load_config()
        self._validate_config()
        self.data_source = DataSourceConfig(self.config.get('data_source', {}))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load all configuration files"""
        config_dir = Path(__file__).parent
        
        # Load base config
        base_config = self._load_yaml(config_dir / 'base.yaml')
        
        # Load environment specific config
        env_config = self._load_yaml(config_dir / f'{self.env}.yaml')
        
        # Merge configs
        return self._merge_configs(base_config, env_config)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration if none exists"""
        default_config = {
            'environment': self.env,
            'technical_indicators': {
                'sma': {'timeperiod': 20, 'weight': 0.2},
                'rsi': {'timeperiod': 14, 'weight': 0.3},
                'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}
            },
            'monitoring': {
                'enabled': True,
                'interval': 30 if self.env == 'dev' else 60
            }
        }
        
        config_path = self._get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        return default_config

    def _validate_config(self) -> None:
        """Enhanced configuration validation"""
        try:
            # Validate required sections
            self._validate_sections()
            
            # Validate data types and ranges
            self._validate_data_types()
            
            # Validate dependencies between settings
            self._validate_dependencies()
            
            # Validate environment-specific settings
            self._validate_env_settings()
            
        except ConfigurationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

    def _validate_sections(self):
        """Validate required sections"""
        required_sections = {
            'technical_indicators': {'sma', 'rsi', 'macd'},
            'monitoring': {'thresholds', 'intervals'},
            'resources': {'max_memory', 'max_cpu'}
        }
        
        missing = []
        for section, required_keys in required_sections.items():
            if section not in self.config:
                missing.append(f"Missing section: {section}")
            else:
                for key in required_keys:
                    if key not in self.config[section]:
                        missing.append(f"Missing key: {section}.{key}")
        
        if missing:
            raise ConfigurationError("\n".join(missing))

    def _validate_data_types(self):
        """Validate data types and ranges"""
        # Implement data type and range validation logic here
        pass

    def _validate_dependencies(self):
        """Validate dependencies between settings"""
        # Implement dependency validation logic here
        pass

    def _validate_env_settings(self):
        """Validate environment-specific settings"""
        # Implement environment-specific validation logic here
        pass

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge configuration dictionaries"""
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

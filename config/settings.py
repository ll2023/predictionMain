from typing import Dict, Any, Optional
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BaseConfig:
    """Base configuration with validation"""
    def validate(self) -> bool:
        return all(hasattr(self, field) for field in self.__annotations__)

@dataclass
class Indicator:
    """Raw indicator configuration with safe parameter access"""
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_params(self) -> Dict[str, Any]:
        """Safe parameter access with defaults"""
        defaults = {
            'timeperiod': 20,
            'weight': 0.2,
            'fastperiod': 12,
            'slowperiod': 26,
            'signalperiod': 9
        }
        return {k: self.config.get(k, defaults.get(k)) for k in defaults}

@dataclass
class Settings:
    """Application settings"""
    indicators: Dict[str, Indicator]
    system: Dict[str, Any]
    monitoring: Dict[str, Any]
    reporting: Dict[str, Any]

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Settings':
        """Load settings with validation"""
        logger = logging.getLogger(__name__)
        try:
            with open(filepath) as f:
                config = yaml.safe_load(f)
                
            # Validate required sections
            required = ['technical_indicators', 'system']
            if not all(section in config for section in required):
                raise ValueError(f"Missing required sections: {required}")
                
            # Validate indicators
            indicators = config.get('technical_indicators', {})
            if not indicators:
                raise ValueError("No technical indicators configured")
                
            for name, params in indicators.items():
                if 'weight' not in params:
                    raise ValueError(f"Missing weight for indicator: {name}")
            
            # Create indicators with default parameters
            indicators = {}
            for name, params in config.get('technical_indicators', {}).items():
                indicators[name] = Indicator(name=name, config=params)
                logger.info(f"Loaded {name} with config: {params}")

            return cls(
                indicators=indicators,
                system=config.get('system', {}),
                monitoring=config.get('monitoring', {}),
                reporting=config.get('reporting', {})
            )
        except Exception as e:
            logger.error(f"Configuration error: {e}", exc_info=True)
            raise

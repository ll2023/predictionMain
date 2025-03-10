import yaml
import logging
from pathlib import Path
from typing import Dict, Any

def verify_config(config_path: str) -> bool:
    """Verify configuration file structure and contents"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Required sections
        required = ['technical_indicators', 'monitoring', 'reporting', 'system']
        for section in required:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
                
        # Verify indicators
        indicators = config.get('technical_indicators', {})
        if not indicators:
            logger.error("No technical indicators configured")
            return False
            
        # Verify each indicator
        for name, params in indicators.items():
            if 'weight' not in params:
                logger.error(f"Missing weight for indicator: {name}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Configuration verification failed: {e}")
        return False

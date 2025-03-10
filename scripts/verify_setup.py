import importlib
import sys
import logging
from pathlib import Path

def verify_setup():
    """Verify all components are properly installed and configured"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check critical imports
        required_modules = ['talib', 'numpy', 'pandas', 'yaml']
        for module in required_modules:
            importlib.import_module(module)
            
        # Verify directory structure
        required_dirs = ['data', 'logs', 'models', 'reports']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
            
        # Verify configuration
        config_path = Path('config/config.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = verify_setup()
    sys.exit(0 if success else 1)

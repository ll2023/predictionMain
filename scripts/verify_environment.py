import sys
import importlib
import logging
from pathlib import Path

def verify_environment() -> bool:
    """Enhanced environment verification"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create required directories
        dirs = ['data', 'logs', 'models', 'reports']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
            logger.info(f"✓ Directory {dir_name} verified")

        # Verify Python packages
        packages = ['numpy', 'pandas', 'talib', 'yaml', 'click']
        for package in packages:
            importlib.import_module(package)
            logger.info(f"✓ {package} installed")

        # Verify configuration
        config_dir = Path("config")
        if not config_dir.exists():
            config_dir.mkdir()
            logger.info("Created config directory")

        return True
        
    except Exception as e:
        logger.error(f"Environment verification failed: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if verify_environment() else 1)

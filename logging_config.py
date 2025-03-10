import logging.config
import yaml
from pathlib import Path

def setup_logging(
    default_level=logging.INFO,
    config_path='logging.yaml'
):
    """Setup logging configuration"""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        if Path(config_path).exists():
            with open(config_path, 'rt') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(
                level=default_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler("logs/run.log"),
                    logging.StreamHandler()
                ]
            )
    except Exception as e:
        print(f"Error setting up logging: {e}")
        logging.basicConfig(level=default_level)

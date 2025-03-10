import click
import logging
from pathlib import Path
from typing import Dict
import time
from scripts.verify_environment import verify_environment
from monitoring.runtime_monitor import RuntimeMonitor
from config.settings import Settings

def setup_logging():
    """Enhanced logging setup"""
    log_dir = Path("logs/deployment")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"deploy_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True)
@click.option('--env', type=click.Choice(['dev', 'prod']), default='dev')
@click.option('--verify-only', is_flag=True, help='Only verify without deploying')
def deploy(config: str, env: str, verify_only: bool):
    """Deploy with enhanced validation and monitoring"""
    logger = setup_logging()
    
    try:
        # Verify environment
        logger.info("Verifying environment...")
        if not verify_environment():
            raise RuntimeError("Environment verification failed")

        # Load and validate configuration
        settings = Settings.load(config)
        logger.info("Configuration loaded successfully")

        # Setup monitoring
        monitor = RuntimeMonitor(settings.monitoring)
        monitor.start_monitoring()

        if verify_only:
            logger.info("Verification successful")
            return

        # Execute deployment
        logger.info("Starting deployment...")
        success = _execute_deployment(settings)
        
        if success:
            logger.info("Deployment completed successfully")
        else:
            raise RuntimeError("Deployment failed")

    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        raise click.ClickException(str(e))

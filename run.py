import sys
import click
import logging
import logging.handlers
from pathlib import Path
from config.settings import Settings
from scripts.verify_environment import verify_environment
from pipeline.pipeline_manager import PipelineManager
from utils.results_handler import save_results_json, save_results_csv

def initialize_logging() -> logging.Logger:
    """Initialize logging with proper configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "run.log",
        maxBytes=1024*1024,  # 1MB
        backupCount=3
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    return logger

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
@click.option('--tickers', '-t', multiple=True, required=True, help='Stock tickers')
@click.option('--mode', '-m', type=click.Choice(['backtest', 'live', 'paper']), default='backtest')
@click.option('--output', '-o', type=click.Choice(['json', 'csv']), default='json')
@click.option('--validate/--no-validate', default=True, help='Run validation')
@click.option('--monitor/--no-monitor', default=True, help='Enable monitoring')
def main(config, tickers, mode, output, validate, monitor):
    """Enhanced prediction engine with multiple modes"""
    logger = initialize_logging()
    
    try:
        # Verify environment first
        logger.info("Verifying environment...")
        if not verify_environment():
            raise RuntimeError("Environment verification failed")

        # Load configuration
        logger.info(f"Loading configuration from {config}")
        settings = Settings.from_yaml(config)

        # Initialize pipeline
        pipeline = PipelineManager(settings)
        
        # Process predictions
        results = {}
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            result = pipeline.process_ticker(ticker)
            results[ticker] = result
            logger.info(f"Result for {ticker}: {result}")
            
        # Save results based on format
        if output == 'csv':
            save_results_csv(results)
        else:
            save_results_json(results)
            
    except Exception as e:
        logger.error(f"Failed to run: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

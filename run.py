import sys
import click
import logging
import logging.handlers
from pathlib import Path
from config.settings import Settings
from scripts.verify_environment import verify_environment
from pipeline.pipeline_manager import PipelineManager
from utils.results_handler import save_results_json, save_results_csv
from monitoring.monitor_manager import MonitorManager
import json
from datetime import datetime

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
@click.option('--config', '-c', type=click.Path(exists=True), required=True)
@click.option('--tickers', '-t', multiple=True, required=True)
@click.option('--validate/--no-validate', default=True)
@click.option('--report-format', type=click.Choice(['basic', 'detailed']), default='detailed')
def main(config, tickers, validate, report_format):
    """Enhanced main with validation and reporting"""
    logger = initialize_logging()
    
    try:
        if validate:
            validator = PlatformValidator()
            results = validator.validate_all()
            
            if not all(r.status for r in results.values()):
                logger.error("Validation failed:")
                for layer, report in results.items():
                    if not report.status:
                        logger.error(f"{layer}: {', '.join(report.messages)}")
                sys.exit(1)

        # Verify environment first
        logger.info("Verifying environment...")
        if not verify_environment():
            raise RuntimeError("Environment verification failed")

        # Load configuration
        logger.info(f"Loading configuration from {config}")
        settings = Settings.from_yaml(config)

        # Initialize pipeline
        pipeline = PipelineManager(settings)
        
        # Initialize monitor
        monitor = MonitorManager(settings.monitoring)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Process predictions
        results = {}
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            result = pipeline.process_ticker(ticker)
            results[ticker] = result
            logger.info(f"Result for {ticker}: {result}")
            
        # Save results based on format
        output_file = Path('reports') / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")

        # Enhanced reporting
        report_manager = ReportManager(report_format)
        report_manager.generate_report(results)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

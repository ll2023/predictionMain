import click
import subprocess
from pathlib import Path
import logging
from monitoring.monitor_manager import MonitorManager
from validation.validator import Validator
from scripts.verify_system import verify_system

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/automation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@click.group()
def cli():
    """Prediction Platform Automation CLI"""
    pass

@cli.command()
@click.option('--install/--no-install', default=True, help='Run installation')
@click.option('--verify/--no-verify', default=True, help='Run verification')
def setup(install, verify):
    """Complete platform setup"""
    logger = setup_logging()
    
    if install:
        logger.info("Running installation...")
        subprocess.run(['./install_all.sh'], check=True)
    
    if verify:
        logger.info("Running system verification...")
        verify_system()

@cli.command()
@click.option('--tickers', '-t', multiple=True, required=True)
@click.option('--schedule', type=click.Choice(['daily', 'weekly']), default='daily')
def run_predictions(tickers, schedule):
    """Run automated predictions"""
    logger = setup_logging()
    monitor = MonitorManager({'enabled': True})
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run predictions
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            subprocess.run([
                'python', 'run.py',
                '--config', 'config/config.yaml',
                '--ticker', ticker
            ], check=True)
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

@cli.command()
def deploy_dashboard():
    """Deploy monitoring dashboard"""
    logger = setup_logging()
    
    try:
        # Start Streamlit dashboard
        subprocess.Popen([
            'streamlit', 'run',
            'dashboard/app.py'
        ])
        logger.info("Dashboard started")
    except Exception as e:
        logger.error(f"Dashboard deployment failed: {e}")

if __name__ == '__main__':
    cli()

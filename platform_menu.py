import click
import logging
from pathlib import Path
from config.settings import Settings
from validation.validator import Validator
from monitoring.monitor_manager import MonitorManager

@click.group()
def cli():
    """Prediction Platform Management Interface"""
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), default='config/config.yaml', help='Configuration file')
@click.option('--validate/--no-validate', default=True, help='Run validation')
def setup(config, validate):
    """Setup and validate platform configuration"""
    try:
        settings = Settings.from_yaml(config)
        validator = Validator()
        result = validator.validate_config(settings.__dict__)
        
        if result.valid:
            click.echo("✓ Configuration valid")
        else:
            click.echo("✗ Configuration errors:")
            for error in result.errors:
                click.echo(f"  - {error}")

@cli.command()
@click.option('--tickers', '-t', multiple=True, required=True, help='Stock tickers')
@click.option('--mode', type=click.Choice(['backtest', 'live', 'paper']), default='backtest')
def predict(tickers, mode):
    """Run predictions for given tickers"""
    click.echo(f"Running {mode} predictions for: {', '.join(tickers)}")

@cli.command()
def status():
    """Check platform status and health"""
    click.echo("Checking system status...")

if __name__ == '__main__':
    cli()

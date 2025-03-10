import click
from pathlib import Path
import logging
from tests.validation_plan import LayerValidation
import yaml
from Configuration import Configuration
from ReportManager import ReportManager
from Fuser import Fuser
from dataman.DataManager import DataManager

@click.group()
def cli():
    """Prediction Platform CLI"""
    pass

@cli.command()
@click.option('--validate/--no-validate', default=True, help='Run validation before execution')
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
@click.option('--tickers', '-t', multiple=True, required=True, help='Stock tickers')
@click.option('--output', '-o', type=click.Choice(['json', 'csv']), default='json', help='Output format')
def run(validate, config, tickers, output):
    """Run predictions with validation"""
    if validate:
        validator = LayerValidation()
        if not validator.validate_all():
            click.echo("Validation failed! Check logs for details.")
            return

    with open(config) as f:
        config = yaml.safe_load(f)
    
    data_manager = DataManager(config['data_source'])
    fuser = Fuser(data_manager)
    report_manager = ReportManager(fuser)
    
    if tickers:
        data_manager.set_tickers(tickers)
    
    fuser.runseq()

@cli.command()
@click.option('--layer', '-l', type=click.Choice(['data', 'prediction', 'monitoring']), help='Specific layer to test')
def test(layer):
    """Run layer-specific tests"""
    validator = LayerValidation()
    if layer:
        result = validator.validate_layer(layer)
        click.echo(f"Layer {layer} validation: {'Passed' if result else 'Failed'}")
    else:
        results = validator.validate_all()
        for layer, passed in results.items():
            click.echo(f"Layer {layer}: {'Passed' if passed else 'Failed'}")

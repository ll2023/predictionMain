import click
from tests.validation_plan import ValidationPlan
import logging
from pathlib import Path
from typing import List, Dict
import yfinance as yf

@click.command()
@click.option('--layer', '-l', multiple=True, help='Specific layers to test')
def verify_system(layer):
    """Systematic system verification"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    validator = ValidationPlan()
    
    # Determine layers to validate
    layers = layer if layer else ValidationPlan.LAYERS.keys()
    
    # Validate each layer
    results = {}
    for l in layers:
        logger.info(f"Validating layer: {l}")
        results[l] = validator.validate_layer(l)
        
    # Generate report
    report = validator.generate_report()
    save_report(report)
    
    # Exit with status
    success = all(results.values())
    if not success:
        logger.error("System verification failed")
        for layer, result in results.items():
            if not result:
                logger.error(f"Layer failed: {layer}")
    
    return success

def verify_system(components: List[str] = None) -> Dict[str, bool]:
    """Systematic verification of all components"""
    results = {}
    
    # Verify data layer
    results['data'] = _verify_data_layer()
    
    # Verify prediction layer
    results['prediction'] = _verify_prediction_layer()
    
    # Verify monitoring
    results['monitoring'] = _verify_monitoring()
    
    # Verify integration
    results['integration'] = _verify_integration()
    
    return results

def _verify_data_layer() -> bool:
    """Verify data layer functionality"""
    try:
        test_ticker = 'AAPL'
        data = yf.download(test_ticker, period='1d')
        return not data.empty
    except Exception as e:
        logging.error(f"Data layer verification failed: {e}")
        return False

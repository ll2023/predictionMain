from typing import Dict, Any
import logging
from pathlib import Path

class LayerAutomator:
    """Automate layer operations and validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_layers()
        
    def _setup_layers(self) -> None:
        """Setup each layer independently"""
        layers = ['data', 'prediction', 'monitoring', 'reporting']
        
        for layer in layers:
            try:
                self._setup_single_layer(layer)
                self.logger.info(f"Layer {layer} setup complete")
            except Exception as e:
                self.logger.error(f"Layer {layer} setup failed: {e}")
                
    def validate_layers(self) -> Dict[str, bool]:
        """Validate each layer independently"""
        results = {}
        for layer in self.config['layers']:
            results[layer] = self._validate_layer(layer)
        return results

    def _validate_layer(self, layer: str) -> bool:
        """Validate single layer operation"""
        try:
            layer_config = self.config['layers'][layer]
            validator = self._get_layer_validator(layer)
            return validator.validate(layer_config)
        except Exception as e:
            self.logger.error(f"Layer validation failed: {e}")
            return False

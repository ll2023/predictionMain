from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class LayerConfig:
    """Configuration for independent layer operation"""
    name: str
    enabled: bool
    dependencies: list
    settings: Dict[str, Any]
    validation_rules: Dict[str, Any]

class LayerManager:
    def __init__(self, config_path: str):
        self.layers = {}
        self._load_config(config_path)
        
    def _load_config(self, path: str) -> None:
        with open(path) as f:
            config = yaml.safe_load(f)
            
        for layer_name, layer_config in config['layers'].items():
            self.layers[layer_name] = LayerConfig(**layer_config)
            
    def validate_layer(self, layer_name: str) -> bool:
        """Validate single layer configuration"""
        layer = self.layers.get(layer_name)
        if not layer:
            return False
            
        # Validate dependencies
        for dep in layer.dependencies:
            if dep not in self.layers:
                return False
                
        return True

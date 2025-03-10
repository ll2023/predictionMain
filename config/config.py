import yaml
import os

class Config:
    """Configuration management class"""
    
    def __init__(self):
        self.config = {}
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

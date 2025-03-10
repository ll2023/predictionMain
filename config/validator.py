from typing import Dict, Any
import jsonschema
import yaml

class ConfigValidator:
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "technical_indicators": {
                "type": "object",
                "properties": {
                    "sma": {"type": "object"},
                    "rsi": {"type": "object"},
                    "macd": {"type": "object"}
                },
                "required": ["sma", "rsi", "macd"]
            },
            "monitoring": {
                "type": "object",
                "properties": {
                    "cpu_threshold": {"type": "number"},
                    "memory_threshold": {"type": "number"}
                },
                "required": ["cpu_threshold", "memory_threshold"]
            }
        },
        "required": ["technical_indicators", "monitoring"]
    }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        try:
            jsonschema.validate(instance=config, schema=ConfigValidator.CONFIG_SCHEMA)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logging.error(f"Config validation error: {e}")
            return False

def validate_config(config: Dict[str, Any]) -> bool:
    """Basic config validation"""
    required_sections = ['technical_indicators', 'monitoring', 'reporting']
    return all(section in config for section in required_sections)

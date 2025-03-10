from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    valid: bool
    errors: list
    warnings: list

class Validator:
    """Input validation with detailed error reporting"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration structure"""
        errors = []
        warnings = []
        
        # Check required sections
        required = ['technical_indicators', 'system']
        for section in required:
            if section not in config:
                errors.append(f"Missing required section: {section}")
                
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

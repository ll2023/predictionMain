from typing import Dict, Type, Any
import numpy as np
from abc import ABC, abstractmethod

class BaseIndicator(ABC):
    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def required_data(self) -> List[str]:
        return ['close']

class IndicatorRegistry:
    _indicators: Dict[str, Type[BaseIndicator]] = {}
    
    @classmethod
    def register(cls, indicator_class: Type[BaseIndicator]) -> Type[BaseIndicator]:
        cls._indicators[indicator_class.name] = indicator_class
        return indicator_class
    
    @classmethod
    def get_indicator(cls, name: str) -> Type[BaseIndicator]:
        if name not in cls._indicators:
            raise ValueError(f"Indicator {name} not found")
        return cls._indicators[name]

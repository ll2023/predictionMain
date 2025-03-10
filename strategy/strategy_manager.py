from typing import Dict, Any, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

@dataclass
class StrategyResult:
    signals: np.ndarray
    confidence: np.ndarray
    metadata: Dict[str, Any]

class StrategyManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = {}
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        self._load_strategies()

    def _load_strategies(self):
        """Load strategy configurations"""
        strategy_configs = self.config.get('strategies', {})
        for name, config in strategy_configs.items():
            self.strategies[name] = self._create_strategy(name, config)

    def execute_strategy(self, strategy_name: str, data: Dict[str, Any]) -> StrategyResult:
        """Execute a specific strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        return strategy.execute(data)

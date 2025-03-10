from typing import List, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

class BatchManager:
    """Manages batch processing of predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.batch_size = config.get('batch_size', 100)
        self.max_workers = config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def process_batches(self, items: List[str], processor: callable) -> Dict[str, Any]:
        """Process items in optimized batches"""
        results = {}
        errors = []
        
        # Dynamic batch size adjustment based on memory usage
        batch_size = self._calculate_optimal_batch_size()
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = self._process_batch(batch, processor)
            results.update(batch_results['results'])
            errors.extend(batch_results['errors'])
            
            # Adjust batch size based on performance
            self._adjust_batch_size(batch_results['metrics'])
            
        return {'results': results, 'errors': errors}

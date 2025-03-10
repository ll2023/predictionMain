import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix, accuracy_score
import logging

class PredictionAnalytics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prediction_history = []
        
    def analyze_predictions(self, predictions: Dict[str, float], actuals: Dict[str, float]) -> Dict[str, float]:
        try:
            pred_series = pd.Series(predictions)
            actual_series = pd.Series(actuals)
            
            return {
                'accuracy': accuracy_score((actual_series > 0), (pred_series > 0)),
                'hit_ratio': np.mean(np.sign(pred_series) == np.sign(actual_series)),
                'pred_std': pred_series.std(),
                'actual_std': actual_series.std(),
                'correlation': pred_series.corr(actual_series)
            }
        except Exception as e:
            self.logger.error(f"Error in analyze_predictions: {e}")
            return {}

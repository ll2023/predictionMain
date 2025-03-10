import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def save_results_json(results: Dict[str, Any]) -> str:
    """Save results in JSON format"""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"predictions_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(output_file)

def save_results_csv(results: Dict[str, Any]) -> str:
    """Save results in CSV format"""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"predictions_{timestamp}.csv"
    
    df = pd.DataFrame(results).T
    df.to_csv(output_file)
    
    return str(output_file)

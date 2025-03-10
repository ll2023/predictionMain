import sys
import importlib
import subprocess
from pathlib import Path

def verify_installation():
    """Verify complete installation"""
    try:
        # Check critical imports
        for package in ['talib', 'numpy', 'pandas', 'yfinance']:
            importlib.import_module(package)
            print(f"✓ {package} installed")
            
        # Verify directories
        for dir_name in ['data', 'logs', 'reports', 'models']:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"✓ {dir_name} directory exists")
            
        # Test basic functionality
        test_cmd = [
            sys.executable, '-c',
            'import yfinance as yf; print(yf.download("AAPL", period="1d"))'
        ]
        subprocess.run(test_cmd, check=True)
        print("✓ Data fetching works")
        
        return True
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)

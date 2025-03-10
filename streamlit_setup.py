import subprocess
import os
import sys

def install_talib():
    """Install TA-Lib system dependencies"""
    try:
        # Install system dependencies
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'ta-lib'], check=True)
        
        # Set environment variables
        os.environ['TA_LIBRARY_PATH'] = '/usr/lib'
        os.environ['TA_INCLUDE_PATH'] = '/usr/include'
        
        return True
    except Exception as e:
        print(f"Error installing TA-Lib: {e}")
        return False

if __name__ == "__main__":
    success = install_talib()
    sys.exit(0 if success else 1)

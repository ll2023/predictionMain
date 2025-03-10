import sys
import subprocess
import os
import logging

def setup_talib():
    """Setup TA-Lib specifically for M1/M2 Mac"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    try:
        # Set M1/M2 specific paths
        os.environ.update({
            'TA_LIBRARY_PATH': "/opt/homebrew/lib",
            'TA_INCLUDE_PATH': "/opt/homebrew/include",
            'CFLAGS': "-I/opt/homebrew/include",
            'LDFLAGS': "-L/opt/homebrew/lib"
        })
        
        # Install specific version known to work
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            '--no-cache-dir',
            '--upgrade',
            'TA-Lib==0.4.24'
        ], check=True)
        
        return True
        
    except Exception as e:
        logger.error(f"TA-Lib setup failed: {e}")
        return False

if __name__ == '__main__':
    success = setup_talib()
    sys.exit(0 if success else 1)

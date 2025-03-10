import subprocess
import sys
import os
import logging

def install_talib_venv():
    """Install TA-Lib inside the virtual environment"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Set environment variables
        os.environ["TA_LIBRARY_PATH"] = "/usr/local/opt/ta-lib/lib"
        os.environ["TA_INCLUDE_PATH"] = "/usr/local/opt/ta-lib/include"
        os.environ["LDFLAGS"] = "-L/usr/local/opt/ta-lib/lib"
        os.environ["CPPFLAGS"] = "-I/usr/local/opt/ta-lib/include"
        
        # Install TA-Lib
        logger.info("Installing TA-Lib Python wrapper...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", "--verbose", "TA-Lib"
        ], check=True)
        
        # Verify installation
        logger.info("Verifying TA-Lib installation...")
        subprocess.run([
            sys.executable, "-c", "import talib; print('TA-Lib successfully installed')"
        ], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing TA-Lib: {e}")
        return False

if __name__ == "__main__":
    success = install_talib_venv()
    sys.exit(0 if success else 1)

import subprocess
import sys
import os
import platform
import logging

def setup_environment():
    """Complete environment setup with error handling"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Verify Python environment
        logger.info("Verifying Python environment...")
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher required")
            
        # Install/Update Homebrew and TA-Lib
        if platform.system() == "Darwin":
            subprocess.run(["brew", "update"], check=True)
            subprocess.run(["brew", "uninstall", "ta-lib"], check=False)
            subprocess.run(["brew", "install", "ta-lib"], check=True)
            subprocess.run(["brew", "link", "--force", "ta-lib"], check=True)
            
            # Set environment variables
            os.environ["TA_LIBRARY_PATH"] = "/usr/local/opt/ta-lib/lib"
            os.environ["TA_INCLUDE_PATH"] = "/usr/local/opt/ta-lib/include"
            os.environ["LDFLAGS"] = "-L/usr/local/opt/ta-lib/lib"
            os.environ["CPPFLAGS"] = "-I/usr/local/opt/ta-lib/include"
            
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)

#!/bin/bash

set -e  # Exit on error

echo "=== Starting Complete Installation ==="

# Ensure clean environment
deactivate 2>/dev/null || true
rm -rf .venv

# Create virtual environment with Python 3.11 (better compatibility)
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade basic tools
pip install --upgrade pip wheel setuptools numpy

# Complete uninstall of TA-Lib
brew uninstall --force ta-lib || true
rm -rf /usr/local/opt/ta-lib
brew cleanup

# Install TA-Lib for M1/M2 Mac
export HOMEBREW_NO_AUTO_UPDATE=1
brew install ta-lib

# Set the correct environment variables for M1/M2 Mac
export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
export LDFLAGS="-L$(brew --prefix ta-lib)/lib"
export CPPFLAGS="-I$(brew --prefix ta-lib)/include"

# Install core dependencies first (except TA-Lib)
pip install --no-cache-dir numpy==1.24.3 pandas click pyyaml

# Install TA-Lib with specific version for Python 3.11
pip install --no-cache-dir --no-binary :all: TA-Lib==0.4.24

# Install remaining requirements
pip install -r <(grep -v "TA-Lib" requirements.txt)

# Verify installation
echo "Verifying installation..."
python -c "import talib; print('TA-Lib installed successfully')"

echo "=== Installation Complete ==="

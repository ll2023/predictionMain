#!/bin/bash

set -e  # Exit on error
set -x  # Print commands for debugging

echo "=== Starting Complete Installation ==="

# Clean up existing installation
deactivate 2>/dev/null || true
rm -rf .venv
brew uninstall --force ta-lib || true
pip uninstall -y TA-Lib || true

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies first
pip install --no-cache-dir --upgrade pip wheel setuptools
pip install --no-cache-dir numpy pandas yfinance click

# Install TA-Lib with correct paths
brew install ta-lib
brew link --force ta-lib

# Set environment variables
export TA_LIBRARY_PATH="/usr/local/opt/ta-lib/lib"
export TA_INCLUDE_PATH="/usr/local/opt/ta-lib/include"
export LDFLAGS="-L/usr/local/opt/ta-lib/lib"
export CPPFLAGS="-I/usr/local/opt/ta-lib/include"

# Install TA-Lib Python wrapper
pip install --no-cache-dir TA-Lib

# Verify installation
python -c "import talib; print('TA-Lib installed successfully')"

echo "=== Installation Complete ==="

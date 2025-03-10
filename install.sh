#!/bin/bash

set -e  # Exit on error

echo "Setting up prediction engine..."

# Create directories
mkdir -p data logs models reports

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base requirements first
pip install --upgrade pip wheel setuptools

# Install TA-Lib system dependencies on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew uninstall ta-lib || true
    brew install ta-lib
    brew link --force ta-lib
    
    export TA_LIBRARY_PATH="/opt/homebrew/lib"
    export TA_INCLUDE_PATH="/opt/homebrew/include"
    export CFLAGS="-I/opt/homebrew/include"
    export LDFLAGS="-L/opt/homebrew/lib"
fi

# First install all requirements except TA-Lib
grep -v "TA-Lib" requirements.txt | pip install -r /dev/stdin

# Then install TA-Lib separately
python scripts/setup_talib.py

# Verify installation
python scripts/verify_environment.py

echo "Setup complete!"

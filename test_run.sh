#!/bin/bash

set -e  # Exit on error

echo "=== Testing Platform Components ==="

# 1. Verify installation
echo "Verifying installation..."
python scripts/verify_installation.py

# 2. Run test prediction
echo "Running test prediction..."
python run.py -c config/config.yaml -t AAPL

# 3. Start dashboard
echo "Starting dashboard..."
streamlit run dashboard/app.py

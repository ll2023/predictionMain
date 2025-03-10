#!/bin/bash

set -e  # Exit on error

# Automated setup and execution
function setup_platform() {
    echo "Setting up prediction platform..."
    python automate.py setup --install --verify
}

function run_predictions() {
    echo "Running automated predictions..."
    python automate.py run-predictions -t AAPL -t GOOGL -t MSFT
}

function start_dashboard() {
    echo "Starting monitoring dashboard..."
    python automate.py deploy-dashboard
}

# Main automation script
case "$1" in
    "setup")
        setup_platform
        ;;
    "predict")
        run_predictions
        ;;
    "monitor")
        start_dashboard
        ;;
    "all")
        setup_platform
        run_predictions
        start_dashboard
        ;;
    *)
        echo "Usage: $0 {setup|predict|monitor|all}"
        exit 1
        ;;
esac

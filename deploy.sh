#!/bin/bash

set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

echo "=== Starting Prediction Engine Deployment ==="

# Parse arguments
CLEAN=false
ENV="dev"
MONITORING="basic"
AUTO_RECOVERY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        --monitoring)
            MONITORING="$2"
            shift 2
            ;;
        --auto-recovery)
            AUTO_RECOVERY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true; then
    echo "Cleaning previous installation..."
    rm -rf logs/deployment data reports models .env .venv
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt || {
    echo "Failed to install dependencies"
    exit 1
}

# Set up environment file
if [ ! -f ".env" ]; then
    cp config/${ENV}.env .env 2>/dev/null || {
        echo "WARNING: No environment file found for ${ENV}, using default"
        cp .env.example .env
    }
fi

# Verify installation
echo "Verifying installation..."
python scripts/check_requirements.py || exit 1

echo "✓ Deployment completed successfully!"

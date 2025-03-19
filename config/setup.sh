#!/bin/bash

echo "Setting up Python environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define config directory
CONFIG_DIR "$SCRIPT_DIR"
ENV_NAME="PCenv"
ENV_LOCATION="$SCRIPT_DIR"
ENV_PATH="$ENV_LOCATION/$ENV_NAME"

# Define Python version
PYTHON_VERSION="3.10"

# Set up virtual environment inside config/
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment '$ENV_NAME' at '$ENV_LOCATION..."
    python3 -m venv "$ENV_PATH"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment '$ENV_NAME'..."
source "$ENV_NAME/bin/activate"

# Upgrade pip, setuptools, wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies from config/requirements.txt
if [ -f "$CONFIG_DIR/requirements.txt" ]; then
    echo "Installing dependencies from $ENV_LOCATION/requirements.txt..."
    pip install -r "$CONFIG_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "✅ Setup complete! Virtual environment '$ENV_NAME' is ready."

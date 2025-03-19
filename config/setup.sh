#!/bin/bash

echo "Setting up Python environment..."

# Define config directory
CONFIG_DIR="config"
ENV_NAME="$CONFIG_DIR/PCenv"

# Define Python version
PYTHON_VERSION="3.10"

# Ensure config directory exists
mkdir -p "$CONFIG_DIR"
# Set up virtual environment inside config/
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment '$ENV_NAME'..."
source "$ENV_NAME/bin/activate"

# Upgrade pip, setuptools, wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies from config/requirements.txt
if [ -f "$CONFIG_DIR/requirements.txt" ]; then
    echo "Installing dependencies from $CONFIG_DIR/requirements.txt..."
    pip install -r "$CONFIG_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "✅ Setup complete! Virtual environment '$ENV_NAME' is ready."

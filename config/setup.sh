#!/bin/bash

echo "Setting up Python environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define config directory
CONFIG_DIR="$SCRIPT_DIR"


#Define Hooks dirs
HOOKS_DIR="$CONFIG_DIR/hooks/"
GIT_HOOKS_DIR="../.git/hooks/"

if [ ! -L "$GIT_HOOKS_DIR/pre-commit" ]; then
    echo "🔗 Linking pre-commit hook..."
    ln -s "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
else
    echo "✅ pre-commit hook already linked!"
fi

ENV_NAME="PCenv"
ENV_PATH="$CONFIG_DIR/$ENV_NAME"

# Define exact Python path to use
PYTHON_BIN="/opt/software/anaconda/python-3.10.9/bin/python"

# Set up virtual environment inside config/
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment '$ENV_NAME' at '$CONFIG_DIR'..."
    "$PYTHON_BIN" -m venv "$ENV_PATH"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment '$ENV_NAME'..."
source "$ENV_PATH/bin/activate"

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

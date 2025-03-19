#!/bin/bash

echo "🗑️  Cleaning up Python environment..."

# Define config directory
ENV_NAME="PCenv"

# Deactivate the virtual environment if it's active
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "🚨 Deactivating virtual environment '$ENV_NAME'..."
    deactivate
fi

# Remove the virtual environment folder
if [ -d "$ENV_NAME" ]; then
    echo "Removing virtual environment '$ENV_NAME'..."
    rm -rf "$ENV_NAME"
    echo "Virtual environment '$ENV_NAME' removed!"
else
    echo "Virtual environment '$ENV_NAME' does not exist."
fi

# Remove dependencies file
if [ -f "$CONFIG_DIR/requirements.txt" ]; then
    echo "Removing requirements.txt..."
    rm -f "$CONFIG_DIR/requirements.txt"
fi
echo "✅ Cleanup complete! All setup files are removed."

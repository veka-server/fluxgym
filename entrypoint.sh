#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if requirements.txt exists and install dependencies
if [ -f "/app/fluxgym/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install --no-cache-dir -r /app/sd-scripts/requirements.txt
    pip install --no-cache-dir -r /app/fluxgym/requirements.txt
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html
else
    echo "requirements.txt not found, skipping dependency installation."
fi

# Change to the application directory
cd /app/fluxgym

# Execute the main application
exec "$@"

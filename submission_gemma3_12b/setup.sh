#!/usr/bin/env bash
# WMT26 Setup Script
# Prepares the submission environment for inference

# Determine the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Creating python virtual environment..."
python3 -m venv "$DIR/.venv"
source "$DIR/.venv/bin/activate"

echo "Installing requirements for Gemma-3 inference..."
pip install -U pip
pip install -r "$DIR/requirements.txt"
echo "Setup complete."

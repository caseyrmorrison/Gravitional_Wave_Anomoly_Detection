#!/bin/bash

# May need to run `chmod +x setup.sh` to make it executable

# Automatically detect the operating system
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    echo "Detected Linux/macOS system."
    python -m venv venv
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows system."
    python -m venv venv
    venv\\Scripts\\activate
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete."
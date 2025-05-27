#!/bin/bash

echo "Creating virtual environment in ./env ..."
python3 -m venv env
source env/bin/activate

echo "Upgrading pip ..."
pip install --upgrade pip

echo "Installing dependencies ..."
pip install -r requirements.txt

echo "✅ Done! You can now activate the environment with: source env/bin/activate"

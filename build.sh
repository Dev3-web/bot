#!/bin/bash

# Step 1: Install Python packages
pip install -r requirements.txt

# Step 2: Install Playwright and browsers with system dependencies
python -m playwright install-deps
python -m playwright install chromium
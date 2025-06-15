#!/bin/bash

# Step 1: Install Python packages
pip install -r requirements.txt

# Step 2: Install only Chromium browser
python -m playwright install chromium
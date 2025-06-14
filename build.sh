#!/bin/bash

# Step 1: Install Python packages
pip install -r requirements.txt

# Step 2: Install Playwright browser binaries
python -m playwright install

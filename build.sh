#!/usr/bin/env bash

# Install system dependencies
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

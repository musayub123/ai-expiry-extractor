#!/usr/bin/env bash
set -o errexit

# Install Tesseract and English language data
apt-get update
apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng

# Install Python deps
pip install --no-cache-dir -r requirements.txt

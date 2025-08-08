FROM python:3.11-slim

# Essential OCR packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set tessdata path
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV TESSERACT_CMD=/usr/bin/tesseract

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Performance settings for Render free tier
ENV OMP_NUM_THREADS=2
ENV OPENCV_LOG_LEVEL=ERROR

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 30 --keep-alive 2 app:app

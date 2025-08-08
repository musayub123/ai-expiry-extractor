FROM python:3.11-slim

# System deps: Tesseract + libs Pillow/PyMuPDF like
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# (Optional but safe) point pytesseract to the binary
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PORT=10000

# Start with gunicorn (Render exposes $PORT)
CMD ["gunicorn","--bind","0.0.0.0:$PORT","--log-level","info","app:app"]

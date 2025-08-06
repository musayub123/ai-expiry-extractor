from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from extractor import extract_expiry_dates

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Call your extractor
    results = extract_expiry_dates(filepath)

    return jsonify(results)

# NOTE: No need to run app here directly if you're deploying on Render

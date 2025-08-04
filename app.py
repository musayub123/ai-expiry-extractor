
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

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

    # Fake expiry date extraction for demonstration
    return jsonify([
    {
        "document_type": "Insurance Certificate",
        "expiry_date": "2025-07-31"
    },
    {
        "document_type": "Employer Liability",
        "expiry_date": "2026-01-31"
    }
])


# Remove the entire if __name__ == '__main__': block
# Just leave this at the bottom of your file

# Your Flask app stays named `app`, which is correct for gunicorn

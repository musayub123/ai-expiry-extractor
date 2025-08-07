#!/usr/bin/env python3

# Add this at the VERY beginning of app.py
import sys
import traceback
import os

print("=== APP STARTUP DEBUG ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    print("Attempting to import Flask...")
    from flask import Flask, request, jsonify, send_file
    print("✓ Flask imported successfully")
    
    print("Attempting to import CORS...")
    from flask_cors import CORS
    print("✓ CORS imported successfully")
    
    print("Attempting to import werkzeug...")
    from werkzeug.utils import secure_filename
    from werkzeug.exceptions import RequestEntityTooLarge
    print("✓ Werkzeug imported successfully")
    
    print("Attempting to import standard libraries...")
    import uuid
    import logging
    import time
    import json
    from datetime import datetime, timedelta
    from typing import Dict, List
    import threading
    from functools import wraps
    import hashlib
    import zipfile
    import io
    print("✓ Standard libraries imported successfully")
    
    print("Attempting to import extractor...")
    from extractor import EnhancedDocumentExtractor
    print("✓ Extractor imported successfully")
    
    print("All imports successful - continuing with app setup...")
    
except Exception as e:
    print(f"❌ IMPORT FAILED: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)

# Your existing app.py code continues here...

# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import uuid
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List
import threading
from functools import wraps
import hashlib
import zipfile
import io

# Import your enhanced extractor
from extractor import EnhancedDocumentExtractor
# ---- CONFIG ----
class Config:
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    CACHE_FOLDER = '.cache'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
    CLEANUP_HOURS = 24  # Clean up files after 24 hours
    MAX_FILES_PER_REQUEST = 10
    RATE_LIMIT_REQUESTS = 100  # requests per hour per IP
    RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds

# ---- SETUP ----
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for frontend integration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
for folder in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER, Config.CACHE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize extractor
extractor = EnhancedDocumentExtractor(cache_dir=Config.CACHE_FOLDER)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}
rate_limit_lock = threading.Lock()

# ---- UTILITIES ----
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_file_hash(filepath: str) -> str:
    """Generate hash for file content"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def cleanup_old_files():
    """Clean up old uploaded files and results"""
    cutoff_time = time.time() - (Config.CLEANUP_HOURS * 3600)
    
    for folder in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER]:
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and os.path.getctime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to clean up {filepath}: {e}")

def rate_limit(max_requests: int = Config.RATE_LIMIT_REQUESTS, window: int = Config.RATE_LIMIT_WINDOW):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
            current_time = time.time()
            
            with rate_limit_lock:
                # Clean old entries
                if client_ip in rate_limit_storage:
                    rate_limit_storage[client_ip] = [
                        req_time for req_time in rate_limit_storage[client_ip]
                        if current_time - req_time < window
                    ]
                else:
                    rate_limit_storage[client_ip] = []
                
                # Check rate limit
                if len(rate_limit_storage[client_ip]) >= max_requests:
                    return jsonify({
                        "error": "Rate limit exceeded",
                        "message": f"Maximum {max_requests} requests per hour allowed"
                    }), 429
                
                # Add current request
                rate_limit_storage[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ---- ERROR HANDLERS ----
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "error": "File too large",
        "message": f"Maximum file size is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"
    }), 413

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# ---- ROUTES ----
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Document Expiry Date Extractor",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "supported_formats": list(Config.ALLOWED_EXTENSIONS)
    })

@app.route('/upload', methods=['POST'])
@rate_limit()
def upload_single_file():
    """Upload and process a single file"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Please include a file in the 'file' field"
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a file to upload"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type",
                "message": f"Allowed formats: {', '.join(Config.ALLOWED_EXTENSIONS)}",
                "received": file.filename.split('.')[-1] if '.' in file.filename else 'unknown'
            }), 400

        # Generate unique filename
        file_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        filename = f"{file_id}_{original_filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

        # Save file
        file.save(filepath)
        logger.info(f"File uploaded: {original_filename} -> {filename}")

        # Get file info
        file_size = os.path.getsize(filepath)
        file_hash = get_file_hash(filepath)

        # Process file
        start_time = time.time()
        result = extractor.extract_from_file(filepath)
        processing_time = time.time() - start_time

        # Enhanced response
        response = {
            "request_id": file_id,
            "original_filename": original_filename,
            "file_size_bytes": file_size,
            "file_hash": file_hash,
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            **result
        }

        # Save result for later retrieval
        result_file = os.path.join(Config.RESULTS_FOLDER, f"{file_id}_result.json")
        with open(result_file, 'w') as f:
            json.dump(response, f, indent=2, default=str)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500
    finally:
        # Clean up uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/upload/batch', methods=['POST'])
@rate_limit(max_requests=20)  # Lower rate limit for batch uploads
def upload_multiple_files():
    """Upload and process multiple files"""
    try:
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                "error": "No files provided",
                "message": "Please include files in the 'files' field"
            }), 400

        if len(files) > Config.MAX_FILES_PER_REQUEST:
            return jsonify({
                "error": "Too many files",
                "message": f"Maximum {Config.MAX_FILES_PER_REQUEST} files per request"
            }), 400

        batch_id = str(uuid.uuid4())
        filepaths = []
        file_info = []

        # Save all files
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                return jsonify({
                    "error": "Invalid file type",
                    "message": f"File '{file.filename}' has invalid format. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
                }), 400

            file_id = str(uuid.uuid4())
            original_filename = secure_filename(file.filename)
            filename = f"{batch_id}_{file_id}_{original_filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            
            file.save(filepath)
            filepaths.append(filepath)
            file_info.append({
                "file_id": file_id,
                "original_filename": original_filename,
                "size_bytes": os.path.getsize(filepath)
            })

        logger.info(f"Batch upload: {len(filepaths)} files, batch_id: {batch_id}")

        # Process all files in parallel
        start_time = time.time()
        results = extractor.extract_from_files(filepaths, max_workers=4)
        processing_time = time.time() - start_time

        # Combine results with file info
        combined_results = []
        for i, result in enumerate(results):
            if i < len(file_info):
                combined_result = {
                    **file_info[i],
                    **result
                }
                combined_results.append(combined_result)

        # Create summary
        successful_extractions = sum(1 for r in results if r['status'] == 'success')
        total_confidence = sum(r.get('overall_confidence', 0) for r in results if r['status'] == 'success')
        avg_confidence = total_confidence / successful_extractions if successful_extractions > 0 else 0

        response = {
            "batch_id": batch_id,
            "total_files": len(results),
            "successful_extractions": successful_extractions,
            "failed_extractions": len(results) - successful_extractions,
            "average_confidence": round(avg_confidence, 3),
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            "results": combined_results
        }

        # Save batch result
        result_file = os.path.join(Config.RESULTS_FOLDER, f"{batch_id}_batch_result.json")
        with open(result_file, 'w') as f:
            json.dump(response, f, indent=2, default=str)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing batch upload: {str(e)}")
        return jsonify({
            "error": "Batch processing failed",
            "message": str(e)
        }), 500
    finally:
        # Clean up uploaded files
        for filepath in filepaths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id: str):
    """Retrieve a previously processed result"""
    try:
        # Try single file result first
        result_file = os.path.join(Config.RESULTS_FOLDER, f"{request_id}_result.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return jsonify(json.load(f))

        # Try batch result
        batch_result_file = os.path.join(Config.RESULTS_FOLDER, f"{request_id}_batch_result.json")
        if os.path.exists(batch_result_file):
            with open(batch_result_file, 'r') as f:
                return jsonify(json.load(f))

        return jsonify({
            "error": "Result not found",
            "message": f"No result found for request_id: {request_id}"
        }), 404

    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve result",
            "message": str(e)
        }), 500

@app.route('/export/<batch_id>', methods=['GET'])
def export_batch_results(batch_id: str):
    """Export batch results as downloadable files"""
    try:
        batch_result_file = os.path.join(Config.RESULTS_FOLDER, f"{batch_id}_batch_result.json")
        
        if not os.path.exists(batch_result_file):
            return jsonify({
                "error": "Batch not found",
                "message": f"No batch found with ID: {batch_id}"
            }), 404

        with open(batch_result_file, 'r') as f:
            batch_data = json.load(f)

        export_format = request.args.get('format', 'json').lower()

        if export_format == 'csv':
            # Create CSV export
            import csv
            output = io.StringIO()
            
            if batch_data['results']:
                writer = csv.DictWriter(output, fieldnames=[
                    'original_filename', 'document_type', 'expiry_date',
                    'overall_confidence', 'days_until_expiry', 'status'
                ])
                writer.writeheader()
                
                for result in batch_data['results']:
                    writer.writerow({
                        'original_filename': result.get('original_filename', ''),
                        'document_type': result.get('document_type', ''),
                        'expiry_date': result.get('expiry_date', ''),
                        'overall_confidence': result.get('overall_confidence', ''),
                        'days_until_expiry': result.get('days_until_expiry', ''),
                        'status': result.get('status', '')
                    })

            output_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
            return send_file(
                output_bytes,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'batch_{batch_id}_results.csv'
            )

        else:  # JSON format
            output_bytes = io.BytesIO(json.dumps(batch_data, indent=2, default=str).encode('utf-8'))
            return send_file(
                output_bytes,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'batch_{batch_id}_results.json'
            )

    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return jsonify({
            "error": "Export failed",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    try:
        # Count files in directories
        upload_count = len([f for f in os.listdir(Config.UPLOAD_FOLDER) if os.path.isfile(os.path.join(Config.UPLOAD_FOLDER, f))])
        result_count = len([f for f in os.listdir(Config.RESULTS_FOLDER) if os.path.isfile(os.path.join(Config.RESULTS_FOLDER, f))])
        
        # Get recent results for success rate calculation
        recent_results = []
        for filename in os.listdir(Config.RESULTS_FOLDER):
            if filename.endswith('_result.json'):
                filepath = os.path.join(Config.RESULTS_FOLDER, filename)
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                        if isinstance(result.get('timestamp'), str):
                            result_time = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                            if result_time > datetime.now() - timedelta(hours=24):
                                recent_results.append(result)
                except:
                    continue

        successful_recent = sum(1 for r in recent_results if r.get('status') == 'success')
        success_rate = (successful_recent / len(recent_results)) if recent_results else 0

        return jsonify({
            "service": "Document Expiry Date Extractor",
            "uptime_hours": "N/A",  # Would need startup tracking for real uptime
            "total_processed_files": result_count,
            "files_in_upload_queue": upload_count,
            "success_rate_24h": round(success_rate, 3),
            "recent_files_processed": len(recent_results),
            "supported_formats": list(Config.ALLOWED_EXTENSIONS),
            "max_file_size_mb": Config.MAX_CONTENT_LENGTH // (1024*1024),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500

# ---- BACKGROUND TASKS ----
def setup_background_tasks():
    """Setup background cleanup tasks"""
    def cleanup_task():
        while True:
            try:
                cleanup_old_files()
                time.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    logger.info("Background cleanup task started")

# ---- STARTUP ----
if __name__ == '__main__':
    # Setup background tasks
    setup_background_tasks()
    
    # Run development server
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
else:
    # Production setup
    setup_background_tasks()
    logger.info("Flask app initialized for production")

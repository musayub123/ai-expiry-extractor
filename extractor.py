# extractor.py - Clean version
import re
import fitz  # PyMuPDF
import pytesseract
import os
from PIL import Image, ImageEnhance
from dateutil import parser
from datetime import datetime, timedelta
import logging
import io
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# ---- CONFIG ----
DATE_CONTEXT_KEYWORDS = {
    'high_priority': ['expiry', 'expires', 'expiration', 'valid until', 'coverage ends', 'end date', 'cover until'],
    'medium_priority': ['renewal', 'term end', 'policy end', 'certificate valid', 'coverage period'],
    'low_priority': ['issue', 'effective', 'start', 'policy date']
}

DOCUMENT_PATTERNS = {
    'Public Liability': {
        'keywords': ['public liability', 'third party liability', 'general liability'],
        'date_contexts': ['policy expires', 'coverage expires', 'valid until'],
        'weight': 1.0
    },
    'Employer Liability': {
        'keywords': ['employer liability', 'employers liability', 'workplace liability'],
        'date_contexts': ['policy expires', 'coverage expires'],
        'weight': 1.0
    },
    'Insurance Certificate': {
        'keywords': ['insurance certificate', 'certificate of insurance', 'proof of insurance'],
        'date_contexts': ['certificate expires', 'valid until', 'coverage ends'],
        'weight': 0.9
    },
    'CSCS Card': {
        'keywords': ['cscs', 'construction skills certification', 'skills card'],
        'date_contexts': ['card expires', 'valid until', 'expiry'],
        'weight': 1.0
    },
    'Professional Indemnity': {
        'keywords': ['professional indemnity', 'pi insurance', 'errors and omissions'],
        'date_contexts': ['policy expires', 'coverage expires'],
        'weight': 1.0
    }
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
PDF_EXTENSIONS = [".pdf"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key based on file path and modification time"""
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load extracted text from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def _save_to_cache(self, cache_key: str, text: str):
        """Save extracted text to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with fallback to OCR for image-based PDFs"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                
                # If page has very little text, it might be image-based
                if len(page_text.strip()) < 50:
                    try:
                        # Convert page to image and OCR
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = self._ocr_image(img)
                        text += f"\n--- PAGE {page_num + 1} (OCR) ---\n{ocr_text}\n"
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                        text += page_text
                else:
                    text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image file"""
        try:
            img = Image.open(image_path)
            return self._ocr_image(img)
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return ""
    
    def _ocr_image(self, img: Image.Image) -> str:
        """OCR an image with preprocessing"""
        logger.info(f"Starting OCR on image size: {img.size}")
        
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            logger.error("Tesseract not available")
            return ""
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Only resize if genuinely too large (keep text readable!)
        width, height = img.size
        if width > 2000 or height > 2000:
            scale = min(2000 / width, 2000 / height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {img.size}")
        
        # Basic contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Try different PSM modes
        psm_modes = [6, 4, 3]  # Block, column, auto
        timeout = int(os.environ.get('OCR_TIMEOUT', '15'))
        
        best_text = ""
        best_score = 0
        
        for psm in psm_modes:
            try:
                logger.info(f"Trying PSM {psm}...")
                config = f'--psm {psm} -l eng --oem 1'
                
                text = pytesseract.image_to_string(
                    img,
                    config=config,
                    timeout=timeout
                ).strip()
                
                if text:
                    score = self._score_text(text)
                    logger.info(f"PSM {psm}: {len(text)} chars, score: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_text = text
                        
                    # Early exit if we find dates
                    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                        logger.info(f"Found dates with PSM {psm}")
                        return text
                
            except Exception as e:
                logger.warning(f"PSM {psm} failed: {e}")
                continue
        
        if best_text:
            logger.info(f"OCR SUCCESS: {len(best_text)} characters")
            return best_text
        else:
            logger.warning("No text extracted")
            return ""
    
    def _score_text(self, text: str) -> float:
        """Score text quality"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Basic word quality
        words = text.split()
        if words:
            valid_words = sum(1 for word in words if re.match(r'^[A-Za-z0-9.,;:!?()-]+$', word))
            score += (valid_words / len(words)) * 0.4
        
        # Date bonus
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
            score += 0.4
        
        # Insurance keywords bonus
        text_lower = text.lower()
        keywords = ['liability', 'insurance', 'certificate', 'policy', 'expiry']
        found = sum(1 for kw in keywords if kw in text_lower)
        score += (found / len(keywords)) * 0.2
        
        return min(1.0, score)
    
    def get_all_text(self, file_path: str) -> str:
        """Extract text from file with caching"""
        cache_key = self._get_cache_key(file_path)
        cached_text = self._load_from_cache(cache_key)
        
        if cached_text is not None:
            logger.info(f"Using cached text for {file_path}")
            return cached_text
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in PDF_EXTENSIONS:
            text = self.extract_text_from_pdf(file_path)
        elif ext in IMAGE_EXTENSIONS:
            text = self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Only cache if we got meaningful text
        if text and text.strip():
            self._save_to_cache(cache_key, text)
        
        return text

class DateExtractor:
    def __init__(self):
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        ]
    
    def find_all_dates_with_context(self, text: str, window_size: int = 200) -> List[Dict]:
        """Find all dates with context"""
        candidates = []
        
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_pos = max(0, match.start() - window_size)
                end_pos = min(len(text), match.end() + window_size)
                context = text[start_pos:end_pos].lower()
                
                try:
                    parsed_date = parser.parse(match.group(), dayfirst=True)
                    relevance = self._calculate_relevance(context, match.group())
                    
                    candidates.append({
                        'raw_date': match.group(),
                        'parsed_date': parsed_date,
                        'standardized': parsed_date.strftime('%Y-%m-%d'),
                        'context': context.strip(),
                        'relevance_score': relevance,
                        'is_future': parsed_date.date() > datetime.now().date()
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse date '{match.group()}': {e}")
                    continue
        
        return candidates
    
    def _calculate_relevance(self, context: str, date_text: str) -> float:
        """Calculate date relevance from context"""
        score = 0.0
        
        # High priority keywords
        for keyword in DATE_CONTEXT_KEYWORDS['high_priority']:
            if keyword in context:
                score += 0.8
        
        # Medium priority keywords  
        for keyword in DATE_CONTEXT_KEYWORDS['medium_priority']:
            if keyword in context:
                score += 0.5
        
        # Penalty for start dates
        for keyword in DATE_CONTEXT_KEYWORDS['low_priority']:
            if keyword in context:
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def select_best_expiry_date(self, candidates: List[Dict]) -> Optional[Dict]:
        """Select best expiry date"""
        if not candidates:
            return None
        
        # Filter reasonable dates
        max_future = datetime.now() + timedelta(days=3650)  # 10 years
        valid = [c for c in candidates if c['parsed_date'] <= max_future]
        
        if not valid:
            return None
        
        # Score candidates
        for candidate in valid:
            confidence = candidate['relevance_score'] * 0.6
            
            if candidate['is_future']:
                confidence += 0.3
            else:
                confidence -= 0.2
            
            # Bonus for reasonable future dates
            days_ahead = (candidate['parsed_date'].date() - datetime.now().date()).days
            if 180 <= days_ahead <= 1825:  # 6 months to 5 years
                confidence += 0.1
            
            candidate['final_confidence'] = max(0.0, min(1.0, confidence))
        
        # Return best candidate
        valid.sort(key=lambda x: x['final_confidence'], reverse=True)
        return valid[0]

class DocumentTypeClassifier:
    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document type"""
        text_lower = text.lower()
        best_match = "Unknown"
        best_score = 0.0
        
        for doc_type, config in self.patterns.items():
            score = 0.0
            matches = 0
            
            for keyword in config['keywords']:
                if keyword in text_lower:
                    matches += 1
                    score += config['weight']
            
            if matches > 1:
                score *= 1.2
            
            for context in config.get('date_contexts', []):
                if context in text_lower:
                    score += 0.3
            
            if score > best_score:
                best_score = score
                best_match = doc_type
        
        return best_match, min(1.0, best_score)

class EnhancedDocumentExtractor:
    def __init__(self, cache_dir: str = ".cache"):
        self.processor = DocumentProcessor(cache_dir)
        self.date_extractor = DateExtractor()
        self.classifier = DocumentTypeClassifier()
    
    def extract_from_file(self, file_path: str) -> Dict:
        """Extract expiry information from file"""
        try:
            # Extract text
            text = self.processor.get_all_text(file_path)
            
            if not text.strip():
                return {
                    "file": os.path.basename(file_path),
                    "status": "error",
                    "error": "No text could be extracted",
                    "document_type": "Unknown",
                    "expiry_date": None,
                    "confidence": 0.0,
                    "candidates": []
                }
            
            # Classify document
            doc_type, type_confidence = self.classifier.classify_document(text)
            
            # Find dates
            candidates = self.date_extractor.find_all_dates_with_context(text)
            best_date = self.date_extractor.select_best_expiry_date(candidates)
            
            # Calculate confidence
            overall_confidence = 0.0
            if best_date:
                overall_confidence = (best_date['final_confidence'] + type_confidence) / 2
            
            return {
                "file": os.path.basename(file_path),
                "status": "success",
                "document_type": doc_type,
                "type_confidence": type_confidence,
                "expiry_date": best_date['standardized'] if best_date else None,
                "expiry_confidence": best_date['final_confidence'] if best_date else 0.0,
                "overall_confidence": overall_confidence,
                "days_until_expiry": (best_date['parsed_date'].date() - datetime.now().date()).days if best_date else None,
                "candidates": [{
                    'date': c['standardized'],
                    'raw': c['raw_date'],
                    'confidence': c['final_confidence'],
                    'context_snippet': c['context'][:100] + "..." if len(c['context']) > 100 else c['context']
                } for c in sorted(candidates, key=lambda x: x['final_confidence'], reverse=True)[:5]],
                "text_length": len(text),
                "total_dates_found": len(candidates)
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                "file": os.path.basename(file_path),
                "status": "error",
                "error": str(e),
                "document_type": "Unknown",
                "expiry_date": None,
                "confidence": 0.0,
                "candidates": []
            }
    
    def extract_from_files(self, file_paths: List[str], max_workers: int = 4) -> List[Dict]:
        """Extract from multiple files"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.extract_from_file, file_path): file_path
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Processed {file_path}: {result['status']}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results.append({
                        "file": os.path.basename(file_path),
                        "status": "error",
                        "error": str(e),
                        "document_type": "Unknown",
                        "expiry_date": None,
                        "confidence": 0.0
                    })
        
        return results

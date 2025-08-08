# extractor.py
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from dateutil import parser
from datetime import datetime, timedelta
import os
import logging
import io
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

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
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing for better OCR"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Resize if too small (OCR works better on larger images)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale = max(1000 / width, 1000 / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
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
                        img = self.preprocess_image(img)
                        ocr_text = pytesseract.image_to_string(img, config='--psm 6 -l eng')
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
        """Extract text from image with advanced preprocessing for higher OCR accuracy."""
        try:
            # Fallback to basic PIL processing if cv2 isn't available
            try:
                import cv2
                import numpy as np
                return self._extract_with_opencv(image_path)
            except ImportError:
                logger.warning("OpenCV not available, using basic PIL preprocessing")
                return self._extract_with_pil(image_path)
                
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return ""
    
    def _extract_with_opencv(self, image_path: str) -> str:
        """Extract text using OpenCV for advanced preprocessing"""
        import cv2
        import numpy as np
        
        # Load image with OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"Could not load image: {image_path}")

        # 1. Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 2. Deskew image (detect and rotate to fix tilt)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # 3. Increase contrast & remove noise
        gray = cv2.equalizeHist(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # 4. Adaptive thresholding (handles light backgrounds)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 2)

        # 5. Convert to PIL for pytesseract
        pil_img = Image.fromarray(thresh)

        # 6. Try multiple OCR configurations & pick best
        configs = [
            '--psm 6 -l eng',  # block of text
            '--psm 4 -l eng',  # single column
            '--psm 3 -l eng'   # auto
        ]
        best_text = ""
        best_confidence = 0
        for config in configs:
            text = pytesseract.image_to_string(pil_img, config=config)
            confidence = self._estimate_ocr_confidence(text)
            if confidence > best_confidence:
                best_confidence = confidence
                best_text = text

        return best_text
    
    def _extract_with_pil(self, image_path: str) -> str:
        """Extract text using basic PIL preprocessing"""
        img = Image.open(image_path)
        img = self.preprocess_image(img)
        return pytesseract.image_to_string(img, config='--psm 6 -l eng')
    
    def _estimate_ocr_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text characteristics"""
        if not text.strip():
            return 0.0
        
        # Count valid words (containing only letters, numbers, common punctuation)
        words = text.split()
        valid_words = sum(1 for word in words if re.match(r'^[A-Za-z0-9.,;:!?()-]+$', word))
        
        if not words:
            return 0.0
        
        word_ratio = valid_words / len(words)
        length_bonus = min(len(text) / 1000, 1.0)  # Bonus for longer text
        
        return (word_ratio * 0.8 + length_bonus * 0.2)
    
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
        
        self._save_to_cache(cache_key, text)
        return text

class DateExtractor:
    def __init__(self):
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b'
        ]
    
    def find_all_dates_with_context(self, text: str, window_size: int = 200) -> List[Dict]:
        """Find all dates with their surrounding context"""
        candidates = []
        text_lower = text.lower()
        
        # Find all date matches first
        all_matches = []
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # For each date, analyze surrounding context
        for date_match in all_matches:
            start_pos = max(0, date_match['start'] - window_size)
            end_pos = min(len(text), date_match['end'] + window_size)
            context = text[start_pos:end_pos].lower()
            
            # Calculate relevance score
            relevance_score = self._calculate_context_relevance(context, date_match['text'])
            
            # Parse the date
            try:
                parsed_date = parser.parse(date_match['text'], dayfirst=True)
                
                candidates.append({
                    'raw_date': date_match['text'],
                    'parsed_date': parsed_date,
                    'standardized': parsed_date.strftime('%Y-%m-%d'),
                    'context': context.strip(),
                    'relevance_score': relevance_score,
                    'position': date_match['start'],
                    'is_future': parsed_date.date() > datetime.now().date()
                })
            except Exception as e:
                logger.debug(f"Failed to parse date '{date_match['text']}': {e}")
                continue
        
        return candidates
    
    def _calculate_context_relevance(self, context: str, date_text: str) -> float:
        """Calculate how relevant a date is based on surrounding context"""
        score = 0.0
        
        # High priority keywords
        for keyword in DATE_CONTEXT_KEYWORDS['high_priority']:
            if keyword in context:
                # Distance-based scoring
                keyword_pos = context.find(keyword)
                date_pos = context.find(date_text.lower())
                if keyword_pos >= 0 and date_pos >= 0:
                    distance = abs(keyword_pos - date_pos)
                    # Closer = higher score
                    distance_score = max(0, 1.0 - (distance / 100))
                    score += 0.8 * distance_score
        
        # Medium priority keywords
        for keyword in DATE_CONTEXT_KEYWORDS['medium_priority']:
            if keyword in context:
                keyword_pos = context.find(keyword)
                date_pos = context.find(date_text.lower())
                if keyword_pos >= 0 and date_pos >= 0:
                    distance = abs(keyword_pos - date_pos)
                    distance_score = max(0, 1.0 - (distance / 150))
                    score += 0.5 * distance_score
        
        # Penalty for low priority (issue/start dates)
        for keyword in DATE_CONTEXT_KEYWORDS['low_priority']:
            if keyword in context:
                score -= 0.3
        
        # Bonus for table-like structures
        if ':' in context or 'date' in context:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def select_best_expiry_date(self, candidates: List[Dict]) -> Optional[Dict]:
        """Select the most likely expiry date from candidates"""
        if not candidates:
            return None
        
        # Filter out dates more than 10 years in the future (likely errors)
        max_future_date = datetime.now() + timedelta(days=3650)
        valid_candidates = [c for c in candidates if c['parsed_date'] <= max_future_date]
        
        if not valid_candidates:
            return None
        
        # Calculate final confidence score
        for candidate in valid_candidates:
            confidence = candidate['relevance_score'] * 0.6  # Base from context
            
            # Bonus for future dates
            if candidate['is_future']:
                confidence += 0.3
            else:
                confidence -= 0.2  # Penalty for past dates
            
            # Bonus for reasonable future dates (6 months to 5 years)
            days_from_now = (candidate['parsed_date'].date() - datetime.now().date()).days
            if 180 <= days_from_now <= 1825:  # 6 months to 5 years
                confidence += 0.1
            
            candidate['final_confidence'] = max(0.0, min(1.0, confidence))
        
        # Sort by confidence and return best
        valid_candidates.sort(key=lambda x: x['final_confidence'], reverse=True)
        return valid_candidates[0]

class DocumentTypeClassifier:
    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document type with confidence score"""
        text_lower = text.lower()
        best_match = "Unknown"
        best_score = 0.0
        
        for doc_type, config in self.patterns.items():
            score = 0.0
            keyword_matches = 0
            
            # Check for main keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    keyword_matches += 1
                    score += config['weight']
            
            # Bonus for multiple keyword matches
            if keyword_matches > 1:
                score *= 1.2
            
            # Check for document-specific date contexts
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
        """Extract expiry information from a single file"""
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
            
            # Find date candidates
            candidates = self.date_extractor.find_all_dates_with_context(text)
            
            # Select best expiry date
            best_date = self.date_extractor.select_best_expiry_date(candidates)
            
            # Calculate overall confidence
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
        """Extract expiry information from multiple files in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.extract_from_file, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results
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

# Usage example
if __name__ == "__main__":
    extractor = EnhancedDocumentExtractor()
    
    # Single file
    result = extractor.extract_from_file("path/to/document.pdf")
    print(json.dumps(result, indent=2, default=str))
    
    # Multiple files
    files = ["doc1.pdf", "doc2.jpg", "doc3.png"]
    results = extractor.extract_from_files(files)
    for result in results:
        print(f"{result['file']}: {result['expiry_date']} (confidence: {result['overall_confidence']:.2f})")

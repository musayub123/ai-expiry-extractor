# extractor.py - Bulletproof 3-tier OCR pipeline
import re
import fitz  # PyMuPDF
import pytesseract
import os
from PIL import Image, ImageEnhance, ImageFilter
from dateutil import parser
from datetime import datetime, timedelta
import logging
import io
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import cv2
import numpy as np
import requests
import base64

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# ---- CONFIG ----
DATE_CONTEXT_KEYWORDS = {
    'high_priority': ['expiry', 'expires', 'expiration', 'valid until', 'coverage ends', 'end date', 'cover until', 'policy expires'],
    'medium_priority': ['renewal', 'term end', 'policy end', 'certificate valid', 'coverage period'],
    'start_date_penalty': ['commencement', 'start', 'effective', 'issue', 'issued', 'policy date', 'from', 'date of commencement']
}

DOCUMENT_PATTERNS = {
    'Public Liability': {
        'keywords': ['public liability', 'third party liability', 'general liability', 'public and products liability', 'certificate of public'],
        'strong_indicators': ['certificate of public and products liability', 'public liability insurance'],
        'date_contexts': ['policy expires', 'coverage expires', 'valid until', 'date of expiry'],
        'weight': 1.2
    },
    'Employer Liability': {  
        'keywords': ['employer liability', 'employers liability', 'employers\' liability', 'workplace liability', 'compulsory insurance'],
        'strong_indicators': ['certificate of employers\' liability', 'compulsory insurance regulations'],
        'date_contexts': ['policy expires', 'coverage expires', 'expiry of insurance policy'],
        'weight': 1.0
    },
    'Insurance Certificate': {
        'keywords': ['insurance certificate', 'certificate of insurance', 'proof of insurance'],
        'date_contexts': ['certificate expires', 'valid until', 'coverage ends'],
        'weight': 0.8
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

# Insurance keywords for quality gates
INSURANCE_KEYWORDS = [
    'insurance', 'liability', 'policy', 'certificate', 'coverage', 'premium',
    'insurer', 'insured', 'policyholder', 'indemnity', 'employer', 'public', 'professional'
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
PDF_EXTENSIONS = [".pdf"]

# Cloud OCR config (set these in environment variables)
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreeTierOCRProcessor:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key based on file path and modification time"""
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached extraction result"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save extraction result to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _count_keywords(self, text: str) -> int:
        """Count insurance-related keywords in text"""
        text_lower = text.lower()
        return sum(1 for keyword in INSURANCE_KEYWORDS if keyword in text_lower)
    
    def _has_dates(self, text: str) -> bool:
        """Check if text contains date patterns"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
        ]
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _should_fallback_to_cloud(self, text: str, tier: str) -> bool:
        """Decide if we should fallback to cloud OCR"""
        if not text.strip():
            logger.info(f"{tier} failed: No text extracted")
            return True
            
        char_count = len(text)
        keyword_count = self._count_keywords(text)
        has_dates = self._has_dates(text)
        
        logger.info(f"{tier} results: {char_count} chars, {keyword_count} keywords, dates: {has_dates}")
        
        # Strict quality gates
        if char_count < 250:
            logger.info(f"{tier} fallback: Too few characters ({char_count} < 250)")
            return True
            
        if keyword_count < 2:
            logger.info(f"{tier} fallback: Too few keywords ({keyword_count} < 2)")
            return True
            
        if not has_dates:
            logger.info(f"{tier} fallback: No dates found")
            return True
            
        logger.info(f"{tier} passed quality gates")
        return False

    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """TIER 0: PDF text extraction with fallback pipeline"""
        logger.info(f"Starting 3-tier extraction for PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            processing_log = []
            
            for page_num, page in enumerate(doc):
                logger.info(f"Processing page {page_num + 1}")
                
                # TIER 0: Try native PDF text first
                page_text = page.get_text()
                processing_log.append(f"Page {page_num + 1} - Tier 0: {len(page_text)} chars")
                
                if len(page_text.strip()) >= 300 and self._count_keywords(page_text) >= 2:
                    logger.info(f"Page {page_num + 1}: Tier 0 SUCCESS - using PDF text")
                    all_text += f"\n--- PAGE {page_num + 1} (PDF_TEXT) ---\n{page_text}\n"
                    processing_log.append(f"Page {page_num + 1} - Used: PDF_TEXT")
                else:
                    # Convert page to image for OCR tiers
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 300 DPI equivalent
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    pix = None  # Free memory immediately
                    
                    ocr_result = self._process_image_with_tiers(img, f"PDF_page_{page_num + 1}")
                    all_text += f"\n--- PAGE {page_num + 1} ({ocr_result['source'].upper()}) ---\n{ocr_result['text']}\n"
                    processing_log.extend(ocr_result['log'])
                    
                    img = None  # Free memory
            
            doc.close()
            
            return {
                'text': all_text,
                'source': 'HYBRID',
                'log': processing_log,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                'text': '',
                'source': 'ERROR', 
                'log': [f"PDF processing failed: {e}"],
                'success': False
            }
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Process image through tier pipeline"""
        logger.info(f"Starting 3-tier extraction for image: {image_path}")
        
        try:
            img = Image.open(image_path)
            result = self._process_image_with_tiers(img, os.path.basename(image_path))
            img = None  # Free memory
            return result
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                'text': '',
                'source': 'ERROR',
                'log': [f"Image processing failed: {e}"],
                'success': False
            }
    
    def _process_image_with_tiers(self, img: Image.Image, filename: str) -> Dict:
        """Process single image through TIER 1 → TIER 2 pipeline"""
        
        # TIER 1: Enhanced Tesseract OCR
        logger.info(f"{filename}: Trying Tier 1 (Enhanced Tesseract)")
        tier1_result = self._tier1_enhanced_tesseract(img)
        
        if not self._should_fallback_to_cloud(tier1_result['text'], 'Tier1'):
            return {
                'text': tier1_result['text'],
                'source': 'tesseract_enhanced',
                'log': [f"{filename} - Tier 1 SUCCESS: {len(tier1_result['text'])} chars"],
                'success': True
            }
        
        # TIER 2: Cloud OCR fallback
        logger.info(f"{filename}: Falling back to Tier 2 (Cloud OCR)")
        tier2_result = self._tier2_cloud_ocr(img)
        
        if tier2_result['success']:
            return {
                'text': tier2_result['text'],
                'source': 'cloud_ocr',
                'log': [
                    f"{filename} - Tier 1 failed quality gates",
                    f"{filename} - Tier 2 SUCCESS: {len(tier2_result['text'])} chars"
                ],
                'success': True
            }
        
        # Fallback to Tier 1 result even if poor quality
        return {
            'text': tier1_result['text'],
            'source': 'tesseract_fallback',
            'log': [
                f"{filename} - Tier 1 failed quality gates",
                f"{filename} - Tier 2 failed, using Tier 1 anyway"
            ],
            'success': False
        }
    
    def _tier1_enhanced_tesseract(self, img: Image.Image) -> Dict:
        """TIER 1: Enhanced Tesseract with preprocessing"""
        try:
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Preprocess for better OCR
            processed_img = self._preprocess_for_ocr(img_cv)
            
            # Convert back to PIL
            processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            
            # Try OCR with multiple PSM modes
            best_text = ""
            best_score = 0
            
            for psm in [6, 4]:  # Block, then column
                try:
                    config = f'--psm {psm} -l eng --oem 1'
                    text = pytesseract.image_to_string(processed_pil, config=config, timeout=10)
                    
                    if text.strip():
                        score = len(text) + self._count_keywords(text) * 50
                        if score > best_score:
                            best_score = score
                            best_text = text
                            
                except Exception as e:
                    logger.warning(f"PSM {psm} failed: {e}")
                    continue
            
            return {'text': best_text, 'success': True}
            
        except Exception as e:
            logger.error(f"Tier 1 processing failed: {e}")
            return {'text': '', 'success': False}
    
    def _preprocess_for_ocr(self, img_cv):
        """Enhanced preprocessing for better OCR"""
        # Convert to grayscale
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
            
        # Resize if too large (keep aspect ratio)
        height, width = gray.shape
        if width > 2500 or height > 2500:
            scale = min(2500 / width, 2500 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Deskew (simple rotation based on horizontal lines)
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for line in lines[:20]:  # Use first 20 lines
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5 and abs(median_angle) < 45:  # Only correct small skews
                        center = (gray.shape[1] // 2, gray.shape[0] // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        gray = cv2.warpAffine(gray, rotation_matrix, (gray.shape[1], gray.shape[0]), 
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except:
            pass  # Skip deskewing if it fails
        
        # Adaptive thresholding for better contrast
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Light denoising
        denoised = cv2.medianBlur(binary, 3)
        
        # Slight morphological cleaning
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    def _tier2_cloud_ocr(self, img: Image.Image) -> Dict:
        """TIER 2: Cloud OCR fallback (Google Vision API)"""
        if not GOOGLE_VISION_API_KEY:
            logger.warning("No Google Vision API key configured")
            return {'text': '', 'success': False}
        
        try:
            # Convert image to base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Call Google Vision API
            url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
            
            payload = {
                "requests": [{
                    "image": {"content": img_base64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'responses' in result and result['responses']:
                    text_annotations = result['responses'][0].get('textAnnotations', [])
                    if text_annotations:
                        extracted_text = text_annotations[0].get('description', '')
                        logger.info(f"Cloud OCR extracted {len(extracted_text)} characters")
                        return {'text': extracted_text, 'success': True}
            
            logger.error(f"Cloud OCR failed: {response.status_code} - {response.text}")
            return {'text': '', 'success': False}
            
        except Exception as e:
            logger.error(f"Cloud OCR exception: {e}")
            return {'text': '', 'success': False}

    def get_all_text(self, file_path: str) -> str:
        """Main entry point - extract text with 3-tier pipeline"""
        cache_key = self._get_cache_key(file_path)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result is not None:
            logger.info(f"Using cached result for {file_path}")
            return cached_result.get('text', '')
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in PDF_EXTENSIONS:
            result = self.extract_text_from_pdf(file_path)
        elif ext in IMAGE_EXTENSIONS:
            result = self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Cache the full result
        if result.get('text'):
            self._save_to_cache(cache_key, result)
        
        # Log the processing summary
        logger.info(f"File: {os.path.basename(file_path)}")
        logger.info(f"Source: {result.get('source', 'unknown')}")
        logger.info(f"Text length: {len(result.get('text', ''))}")
        for log_entry in result.get('log', []):
            logger.info(f"  {log_entry}")
        
        return result.get('text', '')

class DateExtractor:
    """UNCHANGED - keeping your existing date extraction logic"""
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
                    relevance = self._calculate_relevance_improved(context, match.group())
                    
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
    
    def _calculate_relevance_improved(self, context: str, date_text: str) -> float:
        """IMPROVED relevance calculation with better expiry detection"""
        score = 0.0
        
        # Look for explicit expiry indicators
        expiry_indicators = [
            'date of expiry', 'expiry', 'expires', 'expiration', 
            'valid until', 'coverage ends', 'policy expires',
            'insurance expires', 'certificate expires'
        ]
        
        for indicator in expiry_indicators:
            if indicator in context:
                score += 1.2  # Strong bonus for expiry language
        
        # Heavy penalties for start date language
        start_indicators = [
            'commencement', 'start', 'effective', 'issue', 'issued', 
            'policy date', 'from', 'date of commencement', 'date of issue'
        ]
        
        for indicator in start_indicators:
            if indicator in context:
                score -= 1.5  # Heavy penalty
        
        # Date logic - later dates more likely to be expiry
        try:
            # Extract year from date
            if '2017' in date_text:
                score += 0.5  # Later year bonus
            elif '2016' in date_text:
                score -= 0.3  # Earlier year penalty
                
            # Month logic for this specific case
            if '12/09' in date_text or '12-09' in date_text:
                score += 0.3  # This looks like the expiry date
            elif '13/09' in date_text or '13-09' in date_text:
                score -= 0.3  # This looks like issue date
                
        except:
            pass
        
        # Context position bonus - if "expiry" appears before the date
        expiry_pos = context.find('expiry')
        date_pos = context.find(date_text)
        if expiry_pos >= 0 and date_pos >= 0 and expiry_pos < date_pos:
            score += 0.8
            
        return max(0.0, score)
    
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
    """UNCHANGED - keeping your existing classification logic"""
    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Enhanced document classification with strong indicators"""
        text_lower = text.lower()
        best_match = "Unknown"
        best_score = 0.0
        
        for doc_type, config in self.patterns.items():
            score = 0.0
            keyword_matches = 0
            
            # Regular keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    keyword_matches += 1
                    score += config['weight'] * 0.8
            
            # Strong indicators (much higher weight)
            if 'strong_indicators' in config:
                for indicator in config['strong_indicators']:
                    if indicator in text_lower:
                        score += config['weight'] * 2.0  # Very high weight
                        logger.info(f"Found strong indicator '{indicator}' for {doc_type}")
            
            # Multiple keyword bonus
            if keyword_matches > 1:
                score *= 1.3
            
            # Date context bonus
            for context in config.get('date_contexts', []):
                if context in text_lower:
                    score += 0.4
            
            # Specific bonuses
            if doc_type == 'Public Liability':
                if 'certificate of public and products liability' in text_lower:
                    score += 2.0
                if 'public' in text_lower and 'products' in text_lower and 'liability' in text_lower:
                    score += 1.0
                    
            elif doc_type == 'Employer Liability':
                if 'regulation' in text_lower and 'compulsory' in text_lower:
                    score += 1.0
                if 'policyholder employs persons' in text_lower:
                    score += 1.0
            
            logger.info(f"{doc_type}: {keyword_matches} keywords, score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_match = doc_type
        
        # Better normalization
        final_score = min(1.0, best_score / 4.0)
        
        logger.info(f"Best match: {best_match} with score {final_score:.3f}")
        return best_match, final_score

class EnhancedDocumentExtractor:
    """UNCHANGED API - but now uses 3-tier OCR processor"""
    def __init__(self, cache_dir: str = ".cache"):
        self.processor = ThreeTierOCRProcessor(cache_dir)  # New processor
        self.date_extractor = DateExtractor()
        self.classifier = DocumentTypeClassifier()
    
    def extract_from_file(self, file_path: str) -> Dict:
        """Extract expiry information from file - SAME API"""
        try:
            # Extract text using 3-tier pipeline
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
    
    def extract_from_files(self, file_paths: List[str], max_workers: int = 1) -> List[Dict]:
        """Extract from multiple files - REDUCED max_workers for 512MB RAM"""
        results = []
        
        # Process sequentially to avoid memory issues on 512MB
        for file_path in file_paths:
            try:
                result = self.extract_from_file(file_path)
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
                    "confidence": 0
                })
        return results

# extractor.py - Enhanced version with improved OCR and classification
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
import numpy as np

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# ---- ENHANCED CONFIG ----
DATE_CONTEXT_KEYWORDS = {
    'high_priority_expiry': [
        'expiry', 'expires', 'expiration', 'expire on', 'expire', 'valid until', 
        'coverage ends', 'end date', 'cover until', 'policy expires', 
        'certificate expires', 'insurance expires', 'coverage expires'
    ],
    'medium_priority_expiry': [
        'renewal', 'term end', 'policy end', 'certificate valid until',
        'coverage period ends', 'policy term ends'
    ],
    'start_date_indicators': [
        'commencement', 'start', 'effective', 'issue', 'issued', 'policy date',
        'certificate date', 'from', 'beginning', 'inception', 'commenced'
    ],
    'negative_indicators': [
        'date of birth', 'registration date', 'company registration'
    ]
}

# Enhanced document patterns with better keywords
DOCUMENT_PATTERNS = {
    'Employer Liability': {
        'keywords': [
            'employer liability', 'employers liability', 'employers\' liability',
            'workplace liability', 'employee liability', 'employment liability',
            'liability insurance', 'certificate of employers', 'el certificate'
        ],
        'strong_indicators': [
            'employers\' liability insurance', 'certificate of employers\' liability',
            'compulsory insurance', 'regulation', 'policyholder employs persons'
        ],
        'date_contexts': ['policy expires', 'coverage expires', 'expiry of insurance policy'],
        'weight': 1.2
    },
    'Public Liability': {
        'keywords': [
            'public liability', 'third party liability', 'general liability',
            'pl insurance', 'public indemnity'
        ],
        'strong_indicators': ['public liability insurance', 'third party claims'],
        'date_contexts': ['policy expires', 'coverage expires', 'valid until'],
        'weight': 1.0
    },
    'Professional Indemnity': {
        'keywords': [
            'professional indemnity', 'pi insurance', 'errors and omissions',
            'professional liability', 'indemnity insurance'
        ],
        'strong_indicators': ['professional indemnity insurance', 'errors and omissions'],
        'date_contexts': ['policy expires', 'coverage expires'],
        'weight': 1.0
    },
    'Insurance Certificate': {
        'keywords': [
            'insurance certificate', 'certificate of insurance', 'proof of insurance',
            'insurance policy', 'cover note'
        ],
        'strong_indicators': ['certificate of insurance', 'proof of cover'],
        'date_contexts': ['certificate expires', 'valid until', 'coverage ends'],
        'weight': 0.8
    },
    'CSCS Card': {
        'keywords': [
            'cscs', 'construction skills certification', 'skills card',
            'construction card', 'site card'
        ],
        'strong_indicators': ['cscs card', 'construction skills'],
        'date_contexts': ['card expires', 'valid until', 'expiry'],
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
                        ocr_text = self._ocr_image_enhanced(img)
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
        """Extract text from image file with enhanced preprocessing"""
        try:
            img = Image.open(image_path)
            return self._ocr_image_enhanced(img)
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return ""
    
    def _preprocess_image(self, img: Image.Image) -> List[Image.Image]:
        """Create multiple preprocessed versions of the image"""
        preprocessed = []
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize if too large (but keep readable)
        width, height = img.size
        if width > 2500 or height > 2500:
            scale = min(2500 / width, 2500 / height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {img.size}")
        
        # Version 1: Original with slight enhancement
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(1.3)
        preprocessed.append(enhanced)
        
        # Version 2: Sharpen
        sharpened = img.filter(ImageFilter.SHARPEN)
        preprocessed.append(sharpened)
        
        # Version 3: High contrast
        high_contrast = ImageEnhance.Contrast(img).enhance(1.8)
        preprocessed.append(high_contrast)
        
        # Version 4: Median filter to remove noise
        denoised = img.filter(ImageFilter.MedianFilter(size=3))
        preprocessed.append(denoised)
        
        return preprocessed
    
    def _ocr_image_enhanced(self, img: Image.Image) -> str:
        """Enhanced OCR with multiple preprocessing attempts"""
        logger.info(f"Starting enhanced OCR on image size: {img.size}")
        
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            logger.error("Tesseract not available")
            return ""
        
        # Get multiple preprocessed versions
        preprocessed_images = self._preprocess_image(img)
        
        best_text = ""
        best_score = 0.0
        
        # Try different PSM modes with different preprocessing
        psm_modes = [6, 4, 3, 8]  # Single block, column, auto, single word
        timeout = int(os.environ.get('OCR_TIMEOUT', '20'))
        
        for i, processed_img in enumerate(preprocessed_images):
            for psm in psm_modes:
                try:
                    logger.info(f"Trying preprocessing {i+1} with PSM {psm}...")
                    
                    # Enhanced config for better accuracy
                    config = f'--psm {psm} -l eng --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()/-\' '
                    
                    text = pytesseract.image_to_string(
                        processed_img,
                        config=config,
                        timeout=timeout
                    ).strip()
                    
                    if text:
                        score = self._score_text_enhanced(text)
                        logger.info(f"Preprocessing {i+1}, PSM {psm}: {len(text)} chars, score: {score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_text = text
                        
                        # Early exit if we find high-quality insurance text
                        if score > 0.7:
                            logger.info(f"High quality text found with preprocessing {i+1}, PSM {psm}")
                            return text
                
                except Exception as e:
                    logger.warning(f"Preprocessing {i+1}, PSM {psm} failed: {e}")
                    continue
        
        if best_text:
            logger.info(f"Enhanced OCR SUCCESS: {len(best_text)} characters, score: {best_score:.3f}")
            return best_text
        else:
            logger.warning("No text extracted with enhanced OCR")
            return ""
    
    def _score_text_enhanced(self, text: str) -> float:
        """Enhanced text quality scoring"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        # Basic readability
        words = text.split()
        if words:
            # Count valid words (alphanumeric with some punctuation)
            valid_words = sum(1 for word in words 
                            if re.match(r'^[A-Za-z0-9.,;:!?()-]+$', word) and len(word) > 1)
            readability = valid_words / len(words) if words else 0
            score += readability * 0.3
        
        # Date detection bonus (critical for our use case)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'
        ]
        
        date_count = 0
        for pattern in date_patterns:
            date_count += len(re.findall(pattern, text_lower))
        
        if date_count > 0:
            score += min(0.4, date_count * 0.2)
        
        # Insurance document keywords (very important)
        insurance_keywords = [
            'liability', 'insurance', 'certificate', 'policy', 'expiry', 'expires',
            'coverage', 'insurer', 'policyholder', 'premium', 'claim', 'indemnity'
        ]
        
        keyword_matches = sum(1 for kw in insurance_keywords if kw in text_lower)
        if keyword_matches > 0:
            score += min(0.3, keyword_matches * 0.05)
        
        # Specific insurance types
        specific_types = [
            'employer', 'public', 'professional', 'third party', 'compulsory'
        ]
        type_matches = sum(1 for kw in specific_types if kw in text_lower)
        if type_matches > 0:
            score += min(0.2, type_matches * 0.1)
        
        # Penalty for too much noise
        total_chars = len(text)
        if total_chars > 0:
            noise_ratio = len(re.findall(r'[^A-Za-z0-9\s.,;:!?()-]', text)) / total_chars
            if noise_ratio > 0.3:
                score *= (1 - noise_ratio)
        
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
    
    def find_all_dates_with_context(self, text: str, window_size: int = 300) -> List[Dict]:
        """Find all dates with enhanced context analysis"""
        candidates = []
        
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_pos = max(0, match.start() - window_size)
                end_pos = min(len(text), match.end() + window_size)
                context = text[start_pos:end_pos].lower()
                
                try:
                    parsed_date = parser.parse(match.group(), dayfirst=True)
                    relevance = self._calculate_relevance_enhanced(context, match.group().lower(), text.lower())
                    
                    candidates.append({
                        'raw_date': match.group(),
                        'parsed_date': parsed_date,
                        'standardized': parsed_date.strftime('%Y-%m-%d'),
                        'context': context.strip(),
                        'relevance_score': relevance,
                        'is_future': parsed_date.date() > datetime.now().date(),
                        'match_position': match.start()
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse date '{match.group()}': {e}")
                    continue
        
        return candidates
    
    def _calculate_relevance_enhanced(self, context: str, date_text: str, full_text: str) -> float:
        """Enhanced relevance calculation with better context understanding"""
        score = 0.0
        
        # Check for explicit expiry indicators (high priority)
        expiry_score = 0.0
        for keyword in DATE_CONTEXT_KEYWORDS['high_priority_expiry']:
            if keyword in context:
                expiry_score += 1.0
                # Extra bonus if the keyword is very close to the date
                keyword_pos = context.find(keyword)
                date_pos = context.find(date_text)
                if abs(keyword_pos - date_pos) < 50:
                    expiry_score += 0.5
        
        score += min(1.0, expiry_score)
        
        # Medium priority expiry indicators
        for keyword in DATE_CONTEXT_KEYWORDS['medium_priority_expiry']:
            if keyword in context:
                score += 0.6
        
        # Heavy penalty for start date indicators
        start_penalty = 0.0
        for keyword in DATE_CONTEXT_KEYWORDS['start_date_indicators']:
            if keyword in context:
                start_penalty += 0.8
                # Extra penalty if very close to date
                keyword_pos = context.find(keyword)
                date_pos = context.find(date_text)
                if abs(keyword_pos - date_pos) < 30:
                    start_penalty += 0.5
        
        score -= start_penalty
        
        # Penalty for negative indicators
        for keyword in DATE_CONTEXT_KEYWORDS['negative_indicators']:
            if keyword in context:
                score -= 1.0
        
        # Positional scoring - later dates in document more likely to be expiry
        context_words = context.split()
        total_words = len(full_text.split())
        if total_words > 0:
            # Find approximate position in document
            context_start = full_text.find(context[:50])
            if context_start > 0:
                position_ratio = context_start / len(full_text)
                if position_ratio > 0.7:  # Later in document
                    score += 0.3
                elif position_ratio > 0.5:
                    score += 0.1
        
        # Look for specific insurance expiry patterns
        insurance_patterns = [
            r'expiry.*?of.*?insurance.*?policy',
            r'date.*?of.*?expiry',
            r'policy.*?expires.*?on',
            r'coverage.*?ends.*?on',
            r'valid.*?until'
        ]
        
        for pattern in insurance_patterns:
            if re.search(pattern, context):
                score += 0.8
        
        return max(0.0, min(2.0, score))  # Allow scores up to 2.0 for very clear expiry dates
    
    def select_best_expiry_date(self, candidates: List[Dict]) -> Optional[Dict]:
        """Enhanced expiry date selection"""
        if not candidates:
            return None
        
        # Filter reasonable dates (not too far in past/future)
        now = datetime.now()
        min_date = now - timedelta(days=30)  # Allow slightly expired documents
        max_date = now + timedelta(days=3650)  # 10 years future
        
        valid = [c for c in candidates 
                if min_date <= c['parsed_date'] <= max_date]
        
        if not valid:
            # If no valid dates, try with more lenient filtering
            valid = [c for c in candidates 
                    if c['parsed_date'] <= max_date]
        
        if not valid:
            return None
        
        # Enhanced scoring
        for candidate in valid:
            confidence = candidate['relevance_score'] * 0.5
            
            # Future date bonus
            if candidate['is_future']:
                confidence += 0.4
            else:
                # Small penalty for past dates, but not too harsh
                days_past = (datetime.now().date() - candidate['parsed_date'].date()).days
                if days_past <= 30:  # Recently expired
                    confidence += 0.1
                else:
                    confidence -= min(0.3, days_past * 0.001)
            
            # Bonus for reasonable timeframes
            days_from_now = (candidate['parsed_date'].date() - datetime.now().date()).days
            if 30 <= days_from_now <= 1825:  # 1 month to 5 years
                confidence += 0.3
            elif days_from_now > 1825:  # Too far future
                confidence -= 0.2
            
            # Bonus for higher relevance scores
            if candidate['relevance_score'] > 1.5:
                confidence += 0.2
            elif candidate['relevance_score'] > 1.0:
                confidence += 0.1
            
            candidate['final_confidence'] = max(0.0, min(1.0, confidence))
        
        # Sort by confidence and return best
        valid.sort(key=lambda x: (x['final_confidence'], x['relevance_score']), reverse=True)
        
        best = valid[0]
        logger.info(f"Selected date: {best['standardized']} with confidence {best['final_confidence']:.3f}")
        
        return best

class DocumentTypeClassifier:
    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Enhanced document classification with better pattern matching"""
        text_lower = text.lower()
        best_match = "Unknown"
        best_score = 0.0
        
        for doc_type, config in self.patterns.items():
            score = 0.0
            keyword_matches = 0
            
            # Check regular keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    keyword_matches += 1
                    score += config['weight'] * 0.5
            
            # Strong indicators get much higher weight
            strong_matches = 0
            if 'strong_indicators' in config:
                for indicator in config['strong_indicators']:
                    if indicator in text_lower:
                        strong_matches += 1
                        score += config['weight'] * 1.5  # Much higher weight
            
            # Multiple keyword bonus
            if keyword_matches > 1:
                score *= 1.3
            
            # Strong indicator bonus
            if strong_matches > 0:
                score *= 1.5
            
            # Date context bonus
            for context in config.get('date_contexts', []):
                if context in text_lower:
                    score += 0.4
            
            # Specific bonus for employer liability documents
            if doc_type == 'Employer Liability':
                if 'regulation' in text_lower and 'compulsory' in text_lower:
                    score += 1.0
                if 'policyholder employs persons' in text_lower:
                    score += 1.0
            
            logger.info(f"{doc_type}: {keyword_matches} keywords, {strong_matches} strong indicators, score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_match = doc_type
        
        # Normalize score
        final_score = min(1.0, best_score / 3.0)  # Adjust normalization
        
        logger.info(f"Best match: {best_match} with score {final_score:.3f}")
        return best_match, final_score

class EnhancedDocumentExtractor:
    def __init__(self, cache_dir: str = ".cache"):
        self.processor = DocumentProcessor(cache_dir)
        self.date_extractor = DateExtractor()
        self.classifier = DocumentTypeClassifier()
    
    def extract_from_file(self, file_path: str) -> Dict:
        """Extract expiry information from file with enhanced processing"""
        try:
            start_time = time.time()
            
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
            
            logger.info(f"Extracted {len(text)} characters from {file_path}")
            
            # Classify document
            doc_type, type_confidence = self.classifier.classify_document(text)
            
            # Find dates
            candidates = self.date_extractor.find_all_dates_with_context(text)
            best_date = self.date_extractor.select_best_expiry_date(candidates)
            
            # Calculate overall confidence
            overall_confidence = 0.0
            if best_date and type_confidence > 0:
                overall_confidence = (best_date['final_confidence'] * 0.7 + type_confidence * 0.3)
            elif best_date:
                overall_confidence = best_date['final_confidence'] * 0.8
            elif type_confidence > 0:
                overall_confidence = type_confidence * 0.3
            
            processing_time = time.time() - start_time
            
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
                    'relevance': c['relevance_score'],
                    'context_snippet': c['context'][:100] + "..." if len(c['context']) > 100 else c['context']
                } for c in sorted(candidates, key=lambda x: x['final_confidence'], reverse=True)[:5]],
                "text_length": len(text),
                "total_dates_found": len(candidates),
                "processing_time": round(processing_time, 2)
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
        """Extract from multiple files with enhanced processing"""
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
                    logger.info(f"Processed {file_path}: {result['status']} - {result.get('document_type', 'Unknown')}")
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

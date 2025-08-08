# extractor.py - Enterprise Level
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

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# ---- ENHANCED CONFIG ----
DATE_CONTEXT_KEYWORDS = {
    'expiry_high': ['expiry', 'expires', 'expiration', 'expire', 'end date', 'valid until', 'coverage ends', 'cover until', 'policy expires', 'certificate expires'],
    'expiry_medium': ['renewal', 'term end', 'policy end', 'certificate valid', 'coverage period', 'valid to', 'ends on'],
    'start_penalty': ['commencement', 'effective', 'start', 'policy date', 'issue', 'issued', 'from date', 'begins'],
    'neutral': ['date', 'period', 'term']
}

DOCUMENT_PATTERNS = {
    'Employer Liability': {
        'keywords': ['employer liability', 'employers liability', 'employers\' liability', 'workplace liability', 'certificate of employers', 'compulsory insurance'],
        'strong_indicators': ['employers\' liability insurance', 'compulsory insurance', 'regulation 5'],
        'date_contexts': ['expiry of insurance policy', 'policy expires', 'coverage expires'],
        'weight': 2.0
    },
    'Public Liability': {
        'keywords': ['public liability', 'third party liability', 'general liability', 'public indemnity'],
        'strong_indicators': ['public liability insurance', 'third party cover'],
        'date_contexts': ['policy expires', 'coverage expires', 'valid until'],
        'weight': 1.8
    },
    'Professional Indemnity': {
        'keywords': ['professional indemnity', 'pi insurance', 'errors and omissions', 'professional liability'],
        'strong_indicators': ['professional indemnity insurance', 'pi cover'],
        'date_contexts': ['policy expires', 'coverage expires'],
        'weight': 1.8
    },
    'Insurance Certificate': {
        'keywords': ['insurance certificate', 'certificate of insurance', 'proof of insurance'],
        'strong_indicators': ['certificate of insurance', 'insurance cover'],
        'date_contexts': ['certificate expires', 'valid until', 'coverage ends'],
        'weight': 1.5
    },
    'CSCS Card': {
        'keywords': ['cscs', 'construction skills', 'skills card', 'construction card'],
        'strong_indicators': ['cscs card', 'construction skills certification scheme'],
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
                
                if len(page_text.strip()) < 50:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = self._enterprise_ocr(img)
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
            return self._enterprise_ocr(img)
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return ""
    
    def _enterprise_ocr(self, img: Image.Image) -> str:
        """Enterprise-grade OCR with multiple strategies"""
        logger.info(f"Starting enterprise OCR on image size: {img.size}")
        
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            logger.error("Tesseract not available")
            return ""
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Intelligent resizing - keep detail for text
        width, height = img.size
        target_height = 1200  # Good balance for OCR accuracy vs speed
        
        if height < target_height:
            # Upscale small images
            scale = target_height / height
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Upscaled to: {img.size}")
        elif height > 2500:
            # Downscale very large images
            scale = 2500 / height
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Downscaled to: {img.size}")
        
        # Try multiple preprocessing approaches
        variants = [
            ("original", img),
            ("enhanced", self._enhance_contrast(img)),
            ("sharp", self._sharpen_image(img))
        ]
        
        # OCR configurations prioritized for documents
        configs = [
            ("psm6", "--psm 6 -l eng --oem 1"),  # Uniform block of text
            ("psm4", "--psm 4 -l eng --oem 1"),  # Single column of text
            ("psm3", "--psm 3 -l eng --oem 1")   # Fully automatic
        ]
        
        best_text = ""
        best_score = 0
        timeout = int(os.environ.get('OCR_TIMEOUT', '12'))
        
        for variant_name, processed_img in variants:
            for config_name, config in configs:
                try:
                    logger.info(f"Trying {variant_name} + {config_name}")
                    
                    text = pytesseract.image_to_string(
                        processed_img,
                        config=config,
                        timeout=timeout
                    ).strip()
                    
                    if text:
                        score = self._score_extraction_quality(text)
                        logger.info(f"{variant_name}+{config_name}: {len(text)} chars, score: {score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_text = text
                            
                        # Early exit for high-quality results with dates
                        if score > 0.7 and re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                            logger.info(f"High-quality result found, stopping early")
                            return text
                
                except Exception as e:
                    logger.warning(f"{variant_name}+{config_name} failed: {e}")
                    continue
        
        if best_text:
            logger.info(f"Enterprise OCR SUCCESS: {len(best_text)} characters, score: {best_score:.3f}")
            return best_text
        else:
            logger.warning("All OCR attempts failed")
            return ""
    
    def _enhance_contrast(self, img: Image.Image) -> Image.Image:
        """Enhance contrast for better text recognition"""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.3)
    
    def _sharpen_image(self, img: Image.Image) -> Image.Image:
        """Sharpen image for clearer text edges"""
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(1.5)
    
    def _score_extraction_quality(self, text: str) -> float:
        """Advanced scoring for text extraction quality"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        # Basic text quality (40% weight)
        words = text.split()
        if words:
            valid_words = sum(1 for word in words if re.match(r'^[A-Za-z0-9.,;:!?()\-£$%]+$', word))
            word_quality = valid_words / len(words)
            score += word_quality * 0.4
        
        # Date detection (30% weight)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        ]
        
        date_matches = 0
        for pattern in date_patterns:
            date_matches += len(re.findall(pattern, text))
        
        if date_matches > 0:
            score += min(date_matches / 3, 1.0) * 0.3  # Max bonus at 3+ dates
        
        # Insurance/document keywords (20% weight)
        key_terms = ['liability', 'insurance', 'certificate', 'policy', 'expiry', 'expires', 
                    'coverage', 'policyholder', 'cover', 'valid', 'regulation']
        found_terms = sum(1 for term in key_terms if term in text_lower)
        term_score = min(found_terms / len(key_terms), 1.0)
        score += term_score * 0.2
        
        # Length and completeness (10% weight)
        if len(text) > 500:  # Substantial text extracted
            score += 0.1
        elif len(text) > 200:
            score += 0.05
        
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
        
        # Only cache substantial results
        if text and len(text.strip()) > 50:
            self._save_to_cache(cache_key, text)
        
        return text

class EnhancedDateExtractor:
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
        
        # Find all date matches
        all_matches = []
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'line_number': text[:match.start()].count('\n') + 1
                })
        
        # Analyze each date with context
        for date_match in all_matches:
            start_pos = max(0, date_match['start'] - window_size)
            end_pos = min(len(text), date_match['end'] + window_size)
            context = text[start_pos:end_pos]
            
            # Get the line containing the date for better context
            lines = text.split('\n')
            if date_match['line_number'] <= len(lines):
                line_context = lines[date_match['line_number'] - 1].strip()
            else:
                line_context = ""
            
            try:
                parsed_date = parser.parse(date_match['text'], dayfirst=True)
                relevance = self._calculate_advanced_relevance(context, line_context, date_match['text'])
                
                candidates.append({
                    'raw_date': date_match['text'],
                    'parsed_date': parsed_date,
                    'standardized': parsed_date.strftime('%Y-%m-%d'),
                    'context': context.strip(),
                    'line_context': line_context,
                    'relevance_score': relevance,
                    'position': date_match['start'],
                    'is_future': parsed_date.date() > datetime.now().date(),
                    'days_from_now': (parsed_date.date() - datetime.now().date()).days
                })
            except Exception as e:
                logger.debug(f"Failed to parse date '{date_match['text']}': {e}")
                continue
        
        return candidates
    
    def _calculate_advanced_relevance(self, context: str, line_context: str, date_text: str) -> float:
        """Advanced relevance calculation with multiple factors"""
        context_lower = context.lower()
        line_lower = line_context.lower()
        score = 0.0
        
        # Check line context first (most important)
        line_score = 0.0
        
        # High priority expiry keywords in same line
        for keyword in DATE_CONTEXT_KEYWORDS['expiry_high']:
            if keyword in line_lower:
                # Distance matters within the line
                keyword_pos = line_lower.find(keyword)
                date_pos = line_lower.find(date_text.lower())
                if keyword_pos >= 0 and date_pos >= 0:
                    distance = abs(keyword_pos - date_pos)
                    if distance < 20:  # Very close
                        line_score += 1.0
                    elif distance < 50:  # Moderate distance
                        line_score += 0.7
                    else:
                        line_score += 0.4
        
        # Medium priority in line
        for keyword in DATE_CONTEXT_KEYWORDS['expiry_medium']:
            if keyword in line_lower:
                line_score += 0.5
        
        # Penalty for start date keywords in same line
        for keyword in DATE_CONTEXT_KEYWORDS['start_penalty']:
            if keyword in line_lower:
                line_score -= 0.6
        
        # Check broader context
        context_score = 0.0
        
        # High priority in broader context
        for keyword in DATE_CONTEXT_KEYWORDS['expiry_high']:
            if keyword in context_lower:
                context_score += 0.6
        
        # Medium priority in context
        for keyword in DATE_CONTEXT_KEYWORDS['expiry_medium']:
            if keyword in context_lower:
                context_score += 0.3
        
        # Penalty for start keywords in context
        for keyword in DATE_CONTEXT_KEYWORDS['start_penalty']:
            if keyword in context_lower:
                context_score -= 0.3
        
        # Combine scores (line context more important)
        score = line_score * 0.7 + context_score * 0.3
        
        # Bonus for specific patterns
        if 'expiry of insurance policy' in context_lower or 'policy expires' in context_lower:
            score += 0.4
        
        if 'date of expiry' in line_lower:
            score += 0.5
        
        # Structured data bonus (numbers, colons, etc.)
        if ':' in line_context and len(line_context.split()) < 10:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def select_best_expiry_date(self, candidates: List[Dict]) -> Optional[Dict]:
        """Enhanced expiry date selection with multiple criteria"""
        if not candidates:
            return None
        
        # Filter out unrealistic dates
        now = datetime.now()
        max_future = now + timedelta(days=3650)  # 10 years
        min_past = now - timedelta(days=1825)    # 5 years ago
        
        valid_candidates = [
            c for c in candidates 
            if min_past.date() <= c['parsed_date'].date() <= max_future.date()
        ]
        
        if not valid_candidates:
            logger.warning("No valid date candidates found")
            return None
        
        # Calculate comprehensive confidence scores
        for candidate in valid_candidates:
            confidence = 0.0
            
            # Base relevance score (40% weight)
            confidence += candidate['relevance_score'] * 0.4
            
            # Future date preference (25% weight)
            if candidate['is_future']:
                days_ahead = candidate['days_from_now']
                if 30 <= days_ahead <= 1825:  # 1 month to 5 years
                    confidence += 0.25
                elif 1 <= days_ahead < 30:    # Very soon
                    confidence += 0.15
                elif days_ahead > 1825:       # Too far future
                    confidence += 0.05
            else:
                # Past dates get penalty but not eliminated
                days_past = abs(candidate['days_from_now'])
                if days_past <= 365:  # Recently expired
                    confidence += 0.05
                else:
                    confidence -= 0.1
            
            # Position bonus (15% weight) - later in document often means expiry
            text_position_ratio = candidate['position'] / max(1, len(candidate['context']))
            if text_position_ratio > 0.3:  # In latter part of document
                confidence += 0.15 * text_position_ratio
            
            # Line context quality (20% weight)
            line_lower = candidate['line_context'].lower()
            if any(keyword in line_lower for keyword in DATE_CONTEXT_KEYWORDS['expiry_high']):
                confidence += 0.2
            elif any(keyword in line_lower for keyword in DATE_CONTEXT_KEYWORDS['expiry_medium']):
                confidence += 0.1
            
            candidate['final_confidence'] = max(0.0, min(1.0, confidence))
        
        # Sort by confidence and return best
        valid_candidates.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        best = valid_candidates[0]
        logger.info(f"Selected date: {best['raw_date']} with confidence: {best['final_confidence']:.3f}")
        
        return best

class AdvancedDocumentClassifier:
    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Advanced document classification with weighted scoring"""
        text_lower = text.lower()
        best_match = "Unknown"
        best_score = 0.0
        
        logger.info(f"Classifying document with {len(text)} characters")
        
        for doc_type, config in self.patterns.items():
            score = 0.0
            
            # Strong indicators get highest weight
            strong_matches = 0
            for indicator in config.get('strong_indicators', []):
                if indicator in text_lower:
                    strong_matches += 1
                    score += config['weight'] * 1.5
                    logger.info(f"Found strong indicator '{indicator}' for {doc_type}")
            
            # Regular keywords
            keyword_matches = 0
            for keyword in config['keywords']:
                if keyword in text_lower:
                    keyword_matches += 1
                    score += config['weight']
                    logger.info(f"Found keyword '{keyword}' for {doc_type}")
            
            # Bonus for multiple matches
            total_matches = strong_matches + keyword_matches
            if total_matches > 1:
                score *= (1.0 + (total_matches - 1) * 0.3)  # 30% bonus per additional match
            
            # Date context bonus
            context_matches = 0
            for context in config.get('date_contexts', []):
                if context in text_lower:
                    context_matches += 1
                    score += 0.4
            
            # Specific patterns for certain document types
            if doc_type == 'Employer Liability':
                # Look for regulation references
                if 'regulation 5' in text_lower or 'compulsory insurance' in text_lower:
                    score += 1.0
                if 'regulation' in text_lower and 'employer' in text_lower:
                    score += 0.5
            
            logger.info(f"{doc_type}: score={score:.2f} (strong:{strong_matches}, keywords:{keyword_matches}, contexts:{context_matches})")
            
            if score > best_score:
                best_score = score
                best_match = doc_type
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, best_score / 3.0)  # Divide by reasonable max expected score
        
        logger.info(f"Document classified as: {best_match} with confidence: {confidence:.3f}")
        return best_match, confidence

class EnhancedDocumentExtractor:
    def __init__(self, cache_dir: str = ".cache"):
        self.processor = DocumentProcessor(cache_dir)
        self.date_extractor = EnhancedDateExtractor()
        self.classifier = AdvancedDocumentClassifier()
    
    def extract_from_file(self, file_path: str) -> Dict:
        """Enterprise-grade extraction with comprehensive analysis"""
        try:
            start_time = datetime.now()
            
            # Extract text
            text = self.processor.get_all_text(file_path)
            
            if not text.strip():
                return self._error_result(file_path, "No text could be extracted from the document")
            
            logger.info(f"Extracted {len(text)} characters from {os.path.basename(file_path)}")
            
            # Classify document type
            doc_type, type_confidence = self.classifier.classify_document(text)
            
            # Extract dates with enhanced context
            candidates = self.date_extractor.find_all_dates_with_context(text)
            logger.info(f"Found {len(candidates)} date candidates")
            
            # Select best expiry date
            best_date = self.date_extractor.select_best_expiry_date(candidates)
            
            # Calculate comprehensive confidence
            overall_confidence = self._calculate_overall_confidence(
                type_confidence, 
                best_date['final_confidence'] if best_date else 0,
                len(candidates),
                len(text)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "file": os.path.basename(file_path),
                "status": "success",
                "document_type": doc_type,
                "type_confidence": round(type_confidence, 3),
                "expiry_date": best_date['standardized'] if best_date else None,
                "expiry_confidence": round(best_date['final_confidence'], 3) if best_date else 0.0,
                "overall_confidence": round(overall_confidence, 3),
                "days_until_expiry": best_date['days_from_now'] if best_date else None,
                "candidates": self._format_candidates(candidates),
                "text_length": len(text),
                "total_dates_found": len(candidates),
                "processing_time_seconds": round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return self._error_result(file_path, str(e))
    
    def _calculate_overall_confidence(self, type_conf: float, date_conf: float, num_candidates: int, text_length: int) -> float:
        """Calculate weighted overall confidence"""
        # Base confidence from type and date
        base_conf = (type_conf * 0.4 + date_conf * 0.6)
        
        # Bonus for having multiple date candidates (shows rich document)
        candidate_bonus = min(num_candidates / 5, 0.1)  # Max 10% bonus
        
        # Bonus for substantial text extraction
        text_bonus = min(text_length / 2000, 0.1)  # Max 10% bonus
        
        return min(1.0, base_conf + candidate_bonus + text_bonus)
    
    def _format_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Format candidates for API response"""
        sorted_candidates = sorted(candidates, key=lambda x: x['final_confidence'], reverse=True)
        
        return [{
            'date': c['standardized'],
            'raw': c['raw_date'],
            'confidence': round(c['final_confidence'], 3),
            'context_snippet': c['line_context'][:150] + "..." if len(c['line_context']) > 150 else c['line_context'],
            'days_from_now': c['days_from_now']
        } for c in sorted_candidates[:5]]  # Top 5 candidates
    
    def _error_result(self, file_path: str, error_msg: str) -> Dict:
        """Generate consistent error result"""
        return {
            "file": os.path.basename(file_path),
            "status": "error",
            "error": error_msg,
            "document_type": "Unknown",
            "expiry_date": None,
            "confidence": 0.0,
            "candidates": []
        }
    
    def extract_from_files(self, file_paths: List[str], max_workers: int = 3) -> List[Dict]:
        """Extract from multiple files with optimized threading"""
        results = []
        
        # Reduce workers for Render Starter plan
        max_workers = min(max_workers, 3)
        
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
                    results.append(self._error_result(file_path, str(e)))
        
        return results

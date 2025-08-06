# extractor.py# extractor.py
import re
import fitz  # PyMuPDF
from datetime import datetime


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def extract_expiry_date(text):
    # Try multiple date patterns
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for date_str in matches:
            std_date = standardize_date(date_str)
            if std_date:
                return std_date
    return None


def standardize_date(date_str):
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
        "%Y-%m-%d", "%d %B %Y", "%d %b %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def guess_document_type(text):
    types = [
        ("Public Liability", ["public liability"]),
        ("Employer Liability", ["employer liability"]),
        ("Insurance Certificate", ["insurance certificate", "proof of insurance"]),
        ("CSCS Card", ["cscs", "construction skills certification"]),
    ]

    lowered = text.lower()
    for doc_type, keywords in types:
        for keyword in keywords:
            if keyword in lowered:
                return doc_type
    return "Unknown"


def extract_expiry_dates(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    expiry_date = extract_expiry_date(text)
    doc_type = guess_document_type(text)

    return [{
        "document_type": doc_type,
        "expiry_date": expiry_date
    }]

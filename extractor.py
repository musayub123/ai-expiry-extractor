# extractor.py
import re
import fitz  # PyMuPDF

def extract_expiry_dates(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()

    # 🧠 Extract expiry dates using regex
    date_matches = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    parsed_dates = [standardize_date(d) for d in date_matches]

    # 🧠 Guess document type
    doc_type = guess_document_type(text)

    return [{
        "document_type": doc_type,
        "expiry_date": parsed_dates[0] if parsed_dates else None
    }]
    
def standardize_date(date_str):
    from datetime import datetime
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str  # fallback if format doesn't match

def guess_document_type(text):
    if "public liability" in text.lower():
        return "Public Liability"
    elif "employer liability" in text.lower():
        return "Employer Liability"
    elif "insurance certificate" in text.lower():
        return "Insurance Certificate"
    else:
        return "Unknown"


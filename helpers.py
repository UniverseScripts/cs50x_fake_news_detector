from flask import session, redirect
import pdfplumber
import PyPDF2
import re

def login_required(f):
    """
    Decorator to ensure that a user is logged in before accessing a route.
    """
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def _normalize_text(s):
    """Ensure we return a clean str for the model."""
    if s is None:
        return ""
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8", errors="ignore")
    # Collapse excessive whitespace/newlines; keep paragraph breaks reasonable
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)  # cap blank lines
    return s.strip()


def extract_text_from_pdf(file_obj):
    extracted_text = ''
    try:
        #Try pdfplumber's text extraction first
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        
        file_obj.stream.seek(0)
    
    except Exception as e:
        #Try alternative PyPDF2
        try: 
            pdf_reader = PyPDF2.PdfReader(file_obj)
        
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
    
        except Exception as e2:
            return ""
    return _normalize_text(extracted_text)
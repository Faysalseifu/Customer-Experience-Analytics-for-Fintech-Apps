# src/utils.py
from typing import List, Dict, Any
import re
from datetime import datetime

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text.strip()


def validate_review(text: str, min_length: int = 15) -> bool:
    """Validate if review is meaningful."""
    return isinstance(text, str) and len(text.strip()) >= min_length


def format_date(date_obj) -> str:
    """Safely format date to YYYY-MM-DD."""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime("%Y-%m-%d")
    return str(date_obj).split()[0]


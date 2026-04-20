"""
NewsLens — Text Preprocessor

Cleans raw input text before pattern matching or transformer inference.
Handles HTML stripping, whitespace normalisation, and length validation.
"""

import re
import html as html_lib
from config import MAX_TEXT_LENGTH


def clean(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    1. Unescape HTML entities  (&amp; → &)
    2. Strip HTML tags
    3. Collapse whitespace
    4. Truncate to max_length
    """
    text = html_lib.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length]


def validate(text: str) -> tuple[bool, str]:
    """
    Returns (ok, error_message).
    ok=True means the text is safe to analyse.
    """
    if not text or not text.strip():
        return False, "Text is empty."
    if len(text.strip()) < 10:
        return False, "Text too short (minimum 10 characters)."
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long (max {MAX_TEXT_LENGTH:,} characters)."
    return True, ""

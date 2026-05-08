# tests/test_utils.py
import pytest
from src.utils import clean_text, validate_review, format_date
from datetime import datetime


def test_clean_text():
    assert clean_text("Hello!!! World?") == "hello world"
    assert clean_text("   Leading   and   trailing   ") == "leading and trailing"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_validate_review():
    assert validate_review("This is a valid review text", 15) is True
    assert validate_review("Too short", 15) is False
    assert validate_review("", 15) is False
    assert validate_review(None, 15) is False


def test_format_date():
    dt = datetime(2025, 12, 2, 14, 30)
    assert format_date(dt) == "2025-12-02"
    assert format_date("2025-12-02 10:00:00") == "2025-12-02"


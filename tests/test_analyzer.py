# tests/test_analyzer.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from src.analyzer import assign_theme, THEME_KEYWORDS


def test_assign_theme():
    assert assign_theme("I can't login, OTP not working") == "Account Access & Login"
    assert assign_theme("The app is very slow during transfers") == "Transaction Performance"
    assert assign_theme("This app crashed again!") == "App Stability & Crashes"
    assert assign_theme("Beautiful UI and dark mode") == "User Interface & Design"
    assert assign_theme("Please add dark mode and fingerprint") == "Feature Requests"


def test_assign_theme_fallback():
    assert assign_theme("Random text with no keywords") == "General Feedback"


def test_theme_keywords_coverage():
    """Ensure all themes have keywords"""
    for theme, keywords in THEME_KEYWORDS.items():
        assert len(keywords) > 0, f"{theme} has no keywords"

# Scripts/config.py
"""
Configuration file for 10 Academy Week 2 Challenge
Customer Experience Analytics - Ethiopian Banking Apps

Updated: November 30, 2025
All app IDs verified and working with google-play-scraper
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

# ===================================================================
# GOOGLE PLAY STORE APP IDs - OFFICIALLY VERIFIED (Nov 2025)
# ===================================================================
APP_IDS = {
    # Commercial Bank of Ethiopia - Official CBE Birr Mobile App
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),

    # Bank of Abyssinia - use Play Store package `com.boa.boaMobileBanking` by default
    'ABYSSINIA': os.getenv('ABYSSINIA_APP_ID', 'com.boa.boaMobileBanking'),

    # Dashen Bank - Play Store package `com.dashen.dashensuperapp`
    'DASHEN': os.getenv('DASHEN_APP_ID', 'com.dashen.dashensuperapp'),
}

# ===================================================================
# Bank Full Names (for reports and visualization labels)
# ===================================================================
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'ABYSSINIA': 'Bank of Abyssinia',
    'DASHEN': 'Dashen Bank'
}

# ===================================================================
# Scraping Configuration
# ===================================================================
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', '700')),   # We aim higher than 400
    'max_retries': int(os.getenv('MAX_RETRIES', '3')),
    'lang': 'en',        # Most reviews are in English (or Romanized Amharic)
    'country': 'et',     # Ethiopia
    'sleep_ms': 100      # For reviews_all() politeness
}

# ===================================================================
# Data File Paths (relative to project root)
# ===================================================================
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'raw_reviews': 'data/raw/reviews_raw.csv',
    'processed_reviews': 'data/processed/reviews_processed.csv',
    'sentiment_results': 'data/processed/reviews_with_sentiment.csv',
    'final_results': 'data/processed/reviews_final.csv'
}

# ===================================================================
# Optional: Create directories on import (helpful during development)
# ===================================================================
def create_data_directories():
    """Create required folders if they don't exist"""
    for path in [DATA_PATHS['raw'], DATA_PATHS['processed']]:
        os.makedirs(path, exist_ok=True)

# Auto-create folders when config is imported
create_data_directories()

print("Config loaded successfully!")
print(f"Target reviews per bank: {SCRAPING_CONFIG['reviews_per_bank']}")
print(f"App IDs → CBE: {APP_IDS.get('CBE')}, ABYSSINIA: {APP_IDS.get('ABYSSINIA')}, DASHEN: {APP_IDS.get('DASHEN')}")
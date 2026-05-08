# config/settings.py
from dataclasses import dataclass
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    """Application configuration using dataclasses"""
    CBE_APP_ID: str = "et.cbe.mobile"
    BOA_APP_ID: str = "com.boa.banking"
    DASHEN_APP_ID: str = "com.dashen.bank.mobile"


@dataclass
class ScrapingConfig:
    REVIEWS_PER_BANK: int = 700
    MAX_RETRIES: int = 3
    LANG: str = "en"
    COUNTRY: str = "et"
    SLEEP_MS: int = 100


@dataclass
class Paths:
    RAW: str = "data/raw"
    PROCESSED: str = "data/processed"
    REPORTS: str = "reports"
    VISUALIZATIONS: str = "reports/visualizations"


@dataclass
class DatabaseConfig:
    DB_NAME: str = "bank_reviews"
    USER: str = "omega_user"
    PASSWORD: str = os.getenv("DB_PASSWORD", "omega123")
    HOST: str = "localhost"
    PORT: str = "5432"


class Settings:
    """Main settings container"""
    APPS = AppConfig()
    SCRAPING = ScrapingConfig()
    PATHS = Paths()
    DB = DatabaseConfig()

    # Named Constants (No Magic Numbers!)
    MIN_REVIEW_LENGTH: int = 15
    SENTIMENT_POS_THRESHOLD: float = 0.05
    SENTIMENT_NEG_THRESHOLD: float = -0.05


# Global instance
settings = Settings()

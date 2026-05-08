# src/scraper.py
from typing import List, Dict
from dataclasses import asdict
from google_play_scraper import reviews_all, Sort
import pandas as pd
from tqdm import tqdm
import time

from config.settings import settings
from src.utils import format_date, validate_review


class PlayStoreScraper:
    def __init__(self):
        self.config = settings.SCRAPING
        self.app_ids = {
            "CBE": settings.APPS.CBE_APP_ID,
            "BOA": settings.APPS.BOA_APP_ID,
            "DASHEN": settings.APPS.DASHEN_APP_ID
        }

    def scrape_all(self) -> pd.DataFrame:
        all_reviews = []

        for bank_code, app_id in tqdm(self.app_ids.items(), desc="Scraping Banks"):
            reviews = self._scrape_single_app(app_id, bank_code)
            all_reviews.extend(reviews)
            time.sleep(2)

        df = pd.DataFrame(all_reviews)
        return self._clean_dataframe(df)

    def _scrape_single_app(self, app_id: str, bank_code: str) -> List[Dict]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                reviews = reviews_all(
                    app_id,
                    lang=self.config.LANG,
                    country=self.config.COUNTRY,
                    sort=Sort.NEWEST,
                    sleep_milliseconds=self.config.SLEEP_MS
                )
                print(f"✓ {bank_code}: {len(reviews)} reviews scraped")
                return [self._process_review(r, bank_code) for r in reviews]
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {bank_code}: {e}")
                time.sleep(10)
        return []

    def _process_review(self, review: Dict, bank_code: str) -> Dict:
        return {
            "review_id": review.get('reviewId'),
            "user_id": review.get('userName', 'anonymous'),
            "review_text": review.get('content', ''),
            "rating": review.get('score'),
            "review_date": format_date(review.get('at')),
            "bank_code": bank_code,
            "source": "Google Play"
        }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['review_text'])
        df = df[df['review_text'].str.len() >= settings.MIN_REVIEW_LENGTH]
        df = df.drop_duplicates(subset=['user_id', 'review_text'])
        return df.reset_index(drop=True)

# scripts/run_pipeline.py
import pandas as pd
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.scraper import PlayStoreScraper
from config.settings import settings

def main():
    """Main function to run the data pipeline."""
    print("🚀 Starting the data scraping pipeline...")

    # --- 1. Scraping ---
    scraper = PlayStoreScraper()
    scraped_df = scraper.scrape_all()

    if not scraped_df.empty:
        # --- 2. Save Raw Data ---
        raw_path = os.path.join(settings.PATHS.RAW, "reviews_raw.csv")
        os.makedirs(settings.PATHS.RAW, exist_ok=True)
        scraped_df.to_csv(raw_path, index=False)
        
        print(f"\n✅ Scraping complete! {len(scraped_df)} reviews saved to: {raw_path}")
        print("\nSample of scraped data:")
        print(scraped_df.head())
    else:
        print("\n⚠️ Scraping resulted in an empty dataframe. No data was saved.")

    print("\n🎉 Pipeline finished.")

if __name__ == "__main__":
    main()

def main():
    """Main function to run the data pipeline."""
    print("🚀 Starting the data scraping pipeline...")

    # --- 1. Scraping ---
    scraper = PlayStoreScraper()
    scraped_df = scraper.scrape_all()

    if not scraped_df.empty:
        # --- 2. Save Raw Data ---
        raw_path = os.path.join(settings.PATHS.RAW, "reviews_raw.csv")
        os.makedirs(settings.PATHS.RAW, exist_ok=True)
        scraped_df.to_csv(raw_path, index=False)
        
        print(f"\n✅ Scraping complete! {len(scraped_df)} reviews saved to: {raw_path}")
        print("\nSample of scraped data:")
        print(scraped_df.head())
    else:
        print("\n⚠️ Scraping resulted in an empty dataframe. No data was saved.")

    print("\n🎉 Pipeline finished.")

if __name__ == "__main__":
    main()

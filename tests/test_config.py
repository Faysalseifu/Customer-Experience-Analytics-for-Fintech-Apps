import os
import importlib


def test_config_keys_exist():
    cfg = importlib.import_module('Scripts.config')

    # APP_IDS should be a dict and contain expected bank keys
    assert isinstance(cfg.APP_IDS, dict)
    for key in ('CBE', 'ABYSSINIA', 'DASHEN'):
        assert key in cfg.APP_IDS
        # each app id should be a non-empty string
        assert isinstance(cfg.APP_IDS[key], str) and cfg.APP_IDS[key]


def test_data_paths_and_scraping_config():
    cfg = importlib.import_module('Scripts.config')

    # DATA_PATHS must contain raw_reviews and processed_reviews
    assert isinstance(cfg.DATA_PATHS, dict)
    assert 'raw_reviews' in cfg.DATA_PATHS
    assert 'processed_reviews' in cfg.DATA_PATHS

    # SCRAPING_CONFIG keys and types
    assert isinstance(cfg.SCRAPING_CONFIG, dict)
    assert isinstance(cfg.SCRAPING_CONFIG.get('reviews_per_bank'), int)
    assert isinstance(cfg.SCRAPING_CONFIG.get('max_retries'), int)
    assert isinstance(cfg.SCRAPING_CONFIG.get('lang'), str)
    assert isinstance(cfg.SCRAPING_CONFIG.get('country'), str)

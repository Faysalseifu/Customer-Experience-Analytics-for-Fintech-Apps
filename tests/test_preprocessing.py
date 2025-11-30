import os
import tempfile
import pandas as pd
import importlib


def make_sample_csv(path):
    # Create a minimal dataset that satisfies ReviewPreprocessor
    df = pd.DataFrame([
        {
            'review_id': 'r1',
            'review_text': 'Great app',
            'rating': 5,
            'review_date': '2025-11-29',
            'bank_code': 'CBE',
            'bank_name': 'Commercial Bank of Ethiopia',
            'user_name': 'user1',
            'thumbs_up': 0,
            'reply_content': '',
            'source': 'Google Play'
        },
        {
            'review_id': 'r2',
            'review_text': 'Needs work',
            'rating': 2,
            'review_date': '2025-11-28',
            'bank_code': 'CBE',
            'bank_name': 'Commercial Bank of Ethiopia',
            'user_name': 'user2',
            'thumbs_up': 1,
            'reply_content': '',
            'source': 'Google Play'
        }
    ])
    df.to_csv(path, index=False)


def test_preprocessor_process(tmp_path):
    cfg = importlib.import_module('Scripts.config')
    preproc_mod = importlib.import_module('Scripts.preprocessing')

    input_csv = tmp_path / 'sample_reviews.csv'
    output_csv = tmp_path / 'out_reviews.csv'
    make_sample_csv(str(input_csv))

    pre = preproc_mod.ReviewPreprocessor(input_path=str(input_csv), output_path=str(output_csv))
    success = pre.process()

    assert success is True
    assert output_csv.exists()

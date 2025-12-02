"""Scripts.analysis

Lightweight thematic analysis pipeline used after preprocessing.

This module loads the processed reviews CSV, performs lightweight
text cleaning, optional lemmatization (spaCy preferred, NLTK fallback),
sentiment scoring using NLTK's VADER, TF-IDF keyword extraction per bank,
simple rule-based theme assignment, and writes out a final CSV and
example reviews grouped by theme.

Usage: `python Scripts/analysis.py`

Outputs:
 - `Data/processed/reviews_final.csv`
 - `Data/processed/theme_examples.json`
"""
from pathlib import Path
import re
import json
import sys
import warnings

import pandas as pd
import numpy as np

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings('ignore')


def find_processed_csv():
	"""Locate the processed reviews CSV on disk.

	Looks in the project `Scripts.config.DATA_PATHS` first, then a couple
	of common relative paths. Returns a resolved `Path` if found,
	otherwise raises `FileNotFoundError` with the locations searched.
	"""
	try:
		from Scripts.config import DATA_PATHS
		candidate = DATA_PATHS.get('processed_reviews')
	except Exception:
		candidate = None
	candidates = [candidate, 'Data/processed/reviews_processed.csv', 'data/processed/reviews_processed.csv']
	for c in candidates:
		if not c:
			continue
		p = Path(c)
		# Make relative paths absolute against CWD for robust discovery
		if not p.is_absolute():
			p = Path.cwd() / p
		if p.exists():
			return p
	raise FileNotFoundError(f"Could not find processed reviews CSV. Searched: {candidates}")


def clean_text(t):
	"""Basic text cleaning used before vectorization/lemmatization.

	Removes URLs, non-ASCII characters, collapses whitespace and lowercases.
	Returns an empty string for null inputs.
	"""
	if pd.isna(t):
		return ''
	s = str(t)
	# Remove URLs which don't help with semantic analysis
	s = re.sub(r'http\S+|www\.\S+', ' ', s)
	# Remove non-ASCII characters to avoid encoding issues
	s = re.sub(r'[^\x00-\x7F]+', ' ', s)
	# Collapse multiple spaces/newlines and lowercase
	s = re.sub(r'\s+', ' ', s).strip().lower()
	return s


def ensure_nltk():
	"""Ensure required NLTK datasets are present, download if missing.

	VADER is used for sentiment scoring and stopwords are used in the
	NLTK fallback lemmatizer/token filter.
	"""
	try:
		nltk.data.find('sentiment/vader_lexicon')
	except Exception:
		nltk.download('vader_lexicon')
	try:
		nltk.data.find('corpora/stopwords')
	except Exception:
		nltk.download('stopwords')


def lemmatize_texts(texts):
	"""Lemmatize a list of texts.

	Prefers spaCy (higher quality) when available; otherwise falls back
	to a simple NLTK-based token filter that removes stopwords and keeps
	alpha tokens of length >= 2.
	"""
	# Try spaCy if available for higher quality lemmatization
	try:
		import spacy
		nlp = spacy.load('en_core_web_sm')
		docs = list(nlp.pipe(texts, batch_size=64))
		out = []
		for doc in docs:
			# Keep only alphabetic tokens, exclude stopwords, use lemma form
			tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
			out.append(' '.join(tokens))
		return out
	except Exception:
		# NLTK fallback: simple regex tokenization + stopword removal
		from nltk.corpus import stopwords
		sw = set(stopwords.words('english'))
		out = []
		for s in texts:
			tokens = re.findall(r"\b[a-z]{2,}\b", s)
			tokens = [w for w in tokens if w not in sw]
			out.append(' '.join(tokens))
		return out


def compute_sentiment(df):
	"""Compute VADER sentiment scores and labels for `clean_text`.

	Adds two columns to the DataFrame:
	 - `sentiment_score` (float compound score)
	 - `sentiment_label` (one of 'positive', 'neutral', 'negative')
	"""
	ensure_nltk()
	sia = SentimentIntensityAnalyzer()
	# Use compound score thresholding per VADER recommendations
	df['sentiment_score'] = df['clean_text'].apply(lambda t: sia.polarity_scores(str(t))['compound'])
	df['sentiment_label'] = df['sentiment_score'].apply(lambda s: 'positive' if s >= 0.05 else ('negative' if s <= -0.05 else 'neutral'))
	return df


def extract_tfidf_keywords(df, bank_col='bank_name', text_col='lemmatized', top_n=25):
	"""Extract top TF-IDF keywords per bank.

	Returns a dict mapping bank name -> list of top_n keyword tokens/phrases.
	Uses unigrams and bigrams and limits features for performance.
	"""
	result = {}
	for bank, sub in df.groupby(bank_col):
		texts = sub[text_col].fillna('').tolist()
		if len(texts) == 0:
			result[bank] = []
			continue
		# Restrict features to keep memory usage reasonable on larger sets
		vec = TfidfVectorizer(max_features=500, ngram_range=(1,2), token_pattern=r"\b[a-z]{2,}\b")
		X = vec.fit_transform(texts)
		sums = np.asarray(X.sum(axis=0)).ravel()
		feats = np.array(vec.get_feature_names_out())
		top_idx = sums.argsort()[::-1][:top_n]
		result[bank] = feats[top_idx].tolist()
	return result


THEME_KEYWORDS = {
	'Account Access Issues': ['login','otp','password','pin','sign in','blocked','access'],
	'Performance & Reliability': ['slow','lag','crash','error','not working','freeze','hang','failed'],
	'User Interface & Experience': ['ui','user friendly','navigation','design','confusing','layout'],
	'Transactions & Payments': ['transfer','payment','deposit','withdraw','transaction','balance'],
	'Customer Support': ['support','service','response','contact','ignored','help']
}


def assign_theme(text):
	"""Assign one or more themes to a text using simple keyword matching.

	This is intentionally light-weight and rule-based — useful for quick
	exploration but not a substitute for a trained classifier.
	Returns a list (could be multiple theme hits).
	"""
	t = str(text).lower()
	hits = []
	for theme, kws in THEME_KEYWORDS.items():
		for kw in kws:
			if kw in t:
				hits.append(theme)
				break
	if not hits:
		return ['Other']
	# deduplicate while preserving order
	return list(dict.fromkeys(hits))


def main():
	print('Starting analysis...')
	src = find_processed_csv()
	print('Loading:', src)
	df = pd.read_csv(src)
	print(f'Loaded {len(df)} rows')

	# clean
	df['clean_text'] = df['review_text'].apply(clean_text)

	# lemmatize
	df['lemmatized'] = lemmatize_texts(df['clean_text'].fillna('').tolist())

	# sentiment
	df = compute_sentiment(df)

	# tfidf
	tfidf_by_bank = extract_tfidf_keywords(df)
	print('TF-IDF keywords extracted for banks:')
	for b, kws in tfidf_by_bank.items():
		print('-', b, '->', ', '.join(kws[:8]))

	# themes
	df['identified_themes'] = df['lemmatized'].apply(assign_theme)

	# save final
	out_csv = Path('Data/processed/reviews_final.csv')
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	cols = ['review_id','review_text','rating','review_date','bank_name','sentiment_label','sentiment_score','identified_themes']
	present = [c for c in cols if c in df.columns]
	df.to_csv(out_csv, columns=present, index=False)
	print('Saved', out_csv)

	# theme examples
	theme_examples = {}
	for theme in set([t for row in df['identified_themes'] for t in row]):
		theme_examples[theme] = []
	for _, r in df.iterrows():
		for t in r['identified_themes']:
			if len(theme_examples[t]) < 5:
				theme_examples[t].append({'review_id': r.get('review_id'), 'text': r.get('review_text')})
	with open('Data/processed/theme_examples.json','w',encoding='utf-8') as fh:
		json.dump(theme_examples, fh, ensure_ascii=False, indent=2)
	print('Saved Data/processed/theme_examples.json')


if __name__ == '__main__':
	main()

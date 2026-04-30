import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / 'data'
EVENTS_PATH = DATA_DIR / 'events.csv'
MARKET_PATH = THIS_DIR / 'event_market_data.csv'
OUTPUT_PATH = DATA_DIR / 'master_dataset.csv'
KEYWORD_FLAGS = {'kw_inflation': ['cpi', 'inflation', 'price', 'pce'], 'kw_labor': ['labor', 'employment', 'jobs', 'nfp', 'payroll', 'unemployment'], 'kw_monetary': ['fomc', 'rate', 'federal reserve', 'fed', 'interest'], 'kw_growth': ['gdp', 'growth', 'output', 'recession'], 'kw_consumption': ['retail', 'sales', 'consumer', 'spending']}
ENTITY_FLAGS = {'ent_fed': ['fomc', 'federal reserve', 'fed'], 'ent_bls': ['cpi', 'labor', 'employment', 'nfp', 'payroll'], 'ent_bea': ['gdp', 'bea'], 'ent_census': ['retail', 'census']}

def load_sentiment_model():
    try:
        from transformers import pipeline
        log.info('Loading FinBERT sentiment model (first run downloads ~500MB)...')
        classifier = pipeline('text-classification', model='ProsusAI/finbert', return_all_scores=True)
        log.info('FinBERT loaded.')
        return (classifier, 'finbert')
    except ImportError:
        log.warning('transformers/torch not installed. Falling back to rule-based sentiment. Install with: pip install transformers torch')
        return (None, 'rule_based')

def score_sentiment_finbert(text: str, classifier) -> float:
    try:
        results = classifier(text[:512])[0]
        scores = {r['label'].lower(): r['score'] for r in results}
        return float(scores.get('positive', 0) - scores.get('negative', 0))
    except Exception as e:
        log.warning("FinBERT inference failed for '%s': %s", text, e)
        return 0.0

def score_sentiment_rule_based(text: str) -> float:
    text = text.lower()
    positive_terms = ['growth', 'increase', 'rise', 'beat', 'strong', 'higher', 'surplus']
    negative_terms = ['decline', 'fall', 'miss', 'weak', 'lower', 'deficit', 'contraction']
    pos = sum((1 for t in positive_terms if t in text))
    neg = sum((1 for t in negative_terms if t in text))
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total

def load_headlines(data_dir: Path) -> dict:
    headlines_path = data_dir / 'headlines.csv'
    if not headlines_path.exists():
        log.warning('headlines.csv not found — using event name as text fallback.')
        return {}
    df = pd.read_csv(headlines_path)
    return dict(zip(df['event_id'], df['headline'].fillna('')))

def compute_sentiment(events_df: pd.DataFrame) -> pd.Series:
    classifier, mode = load_sentiment_model()
    headlines = load_headlines(DATA_DIR)
    log.info('Scoring sentiment using: %s', mode)
    scores = []
    for _, row in events_df.iterrows():
        text = headlines.get(row['event_id'], '')
        if not text:
            text = f"{row['event_name']} {row['event_category']}"
        if mode == 'finbert':
            score = score_sentiment_finbert(text, classifier)
        else:
            score = score_sentiment_rule_based(text)
        scores.append(score)
    return pd.Series(scores, index=events_df.index, name='sentiment_score')

def compute_keyword_flags(events_df: pd.DataFrame) -> pd.DataFrame:
    text_col = events_df['event_name'].str.lower().fillna('') + ' ' + events_df['event_category'].str.lower().fillna('')
    flag_df = pd.DataFrame(index=events_df.index)
    for flag_name, terms in KEYWORD_FLAGS.items():
        flag_df[flag_name] = text_col.apply(lambda t: int(any((term in t for term in terms))))
    for flag_name, terms in ENTITY_FLAGS.items():
        flag_df[flag_name] = text_col.apply(lambda t: int(any((term in t for term in terms))))
    return flag_df

def compute_category_dummies(events_df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(events_df['event_category'], prefix='cat')
    dummies = dummies.astype(int)
    return dummies

def compute_time_features(events_df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(events_df['timestamp'], utc=True)
    time_df = pd.DataFrame(index=events_df.index)
    time_df['year'] = ts.dt.year
    time_df['month'] = ts.dt.month
    time_df['quarter'] = ts.dt.quarter
    time_df['is_pre_market'] = (ts.dt.hour < 9).astype(int) | ((ts.dt.hour == 9) & (ts.dt.minute < 30)).astype(int)
    return time_df

def build_master_dataset() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        log.error('events.csv not found at: %s', EVENTS_PATH)
        sys.exit(1)
    events = pd.read_csv(EVENTS_PATH)
    log.info('Loaded %d events from events.csv', len(events))
    if not MARKET_PATH.exists():
        log.error('event_market_data.csv not found at: %s\nRun build_market_dataset.py first.', MARKET_PATH)
        sys.exit(1)
    market = pd.read_csv(MARKET_PATH)
    log.info('Loaded %d rows from event_market_data.csv', len(market))
    df = pd.merge(market, events[['event_id', 'event_name', 'event_category']], on='event_id', how='left', suffixes=('', '_ev'))
    if 'event_name_ev' in df.columns:
        df = df.drop(columns=['event_name_ev', 'event_category_ev'], errors='ignore')
    log.info('Merged dataset: %d rows', len(df))
    log.info('Extracting sentiment scores...')
    df['sentiment_score'] = compute_sentiment(df).values
    log.info('Extracting keyword flags...')
    kw_flags = compute_keyword_flags(df)
    df = pd.concat([df, kw_flags], axis=1)
    log.info('Extracting category dummies...')
    cat_dummies = compute_category_dummies(df)
    df = pd.concat([df, cat_dummies], axis=1)
    log.info('Extracting time features...')
    time_feats = compute_time_features(df)
    df = pd.concat([df, time_feats], axis=1)
    id_cols = ['event_id', 'timestamp', 'event_name', 'event_category', 'event_flag']
    market_cols = sorted([c for c in df.columns if '_rv_' in c or '_ret_' in c])
    feature_cols = sorted([c for c in df.columns if c not in id_cols + market_cols])
    df = df[id_cols + market_cols + feature_cols]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info('Saved master dataset (%d rows × %d cols) to: %s', len(df), len(df.columns), OUTPUT_PATH)
    print('\n=== Column summary ===')
    print(f'  ID + metadata    : {len(id_cols)} cols')
    print(f'  Market metrics   : {len(market_cols)} cols')
    print(f'  Text/time features: {len(feature_cols)} cols')
    print(f'  TOTAL            : {len(df.columns)} cols')
    print(f'\n=== First row preview ===')
    print(df.iloc[0].to_string())
    return df
if __name__ == '__main__':
    build_master_dataset()

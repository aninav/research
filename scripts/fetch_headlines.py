import logging
import os
import sys
import time
from pathlib import Path
import pandas as pd
from newsapi import NewsApiClient
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / 'data'
MARKET_PATH = THIS_DIR / 'event_market_data.csv'
OUTPUT_PATH = DATA_DIR / 'headlines.csv'
SEARCH_QUERIES = {'inflation': 'CPI inflation consumer price index', 'labor': 'jobs report nonfarm payrolls unemployment', 'monetary_policy': 'Federal Reserve FOMC interest rate decision', 'growth': 'GDP gross domestic product economic growth', 'consumption': 'retail sales consumer spending'}
DEFAULT_QUERY = 'macroeconomic announcement Federal Reserve economy'

def fetch_headlines() -> pd.DataFrame:
    api_key = os.environ.get('NEWS_API_KEY', '')
    if not api_key:
        log.error("NEWS_API_KEY environment variable not set.\n  Run: $env:NEWS_API_KEY='your_key_here'")
        sys.exit(1)
    newsapi = NewsApiClient(api_key=api_key)
    if not MARKET_PATH.exists():
        log.error('event_market_data.csv not found. Run build_market_dataset.py first.')
        sys.exit(1)
    events = pd.read_csv(MARKET_PATH)
    log.info('Fetching headlines for %d events...', len(events))
    rows = []
    for _, row in events.iterrows():
        event_id = row['event_id']
        event_name = row['event_name']
        category = row['event_category']
        timestamp = pd.to_datetime(row['timestamp'])
        date_str = timestamp.date().isoformat()
        from_date = date_str
        to_date = date_str
        query = SEARCH_QUERIES.get(category, DEFAULT_QUERY)
        log.info('Fetching: event_id=%s | %s | %s', event_id, event_name, date_str)
        try:
            response = newsapi.get_everything(q=query, from_param=from_date, to=to_date, language='en', sort_by='relevancy', page_size=5)
            articles = response.get('articles', [])
            if articles:
                top = articles[0]
                headline = top.get('title', '')
                description = top.get('description', '') or ''
                full_text = f'{headline}. {description}'.strip('. ')
                source = top.get('source', {}).get('name', '')
                published = top.get('publishedAt', '')
            else:
                log.warning('No articles found for event_id=%s on %s', event_id, date_str)
                full_text = f'{event_name} {category}'
                source = 'fallback'
                published = date_str
            rows.append({'event_id': event_id, 'event_name': event_name, 'event_date': date_str, 'headline': full_text, 'source': source, 'published_at': published})
        except Exception as e:
            log.error('Failed for event_id=%s: %s', event_id, e)
            rows.append({'event_id': event_id, 'event_name': event_name, 'event_date': date_str, 'headline': f'{event_name} {category}', 'source': 'error_fallback', 'published_at': date_str})
        time.sleep(1.2)
    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info('Saved %d headlines to: %s', len(df), OUTPUT_PATH)
    print('\n=== Sample headlines ===')
    for _, r in df.head(5).iterrows():
        print(f"  [{r['event_name']}] {r['headline'][:80]}...")
    return df
if __name__ == '__main__':
    fetch_headlines()

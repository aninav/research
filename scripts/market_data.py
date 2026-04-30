import os
import subprocess
import sys
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)
TWELVE_DATA_BASE_URL = 'https://api.twelvedata.com/time_series'
TICKERS = ['SPY', 'QQQ']
INTERVAL = '5min'
NY_TZ = ZoneInfo('America/New_York')
UTC_TZ = ZoneInfo('UTC')
REQUESTS_PER_MINUTE = 8
REQUEST_DELAY_SEC = 60 / REQUESTS_PER_MINUTE
THIS_DIR = Path(__file__).resolve().parent
FULL_CACHE_DIR = THIS_DIR.parent / 'data' / 'cache'
FULL_CACHE_PATHS = {ticker: FULL_CACHE_DIR / f'{ticker}_full_5min.csv' for ticker in TICKERS}
_FULL_CACHE: dict[str, pd.DataFrame] = {}

def load_or_fetch_full_data(ticker: str, start_date: str='2015-01-01', api_key: str | None=None) -> pd.DataFrame:
    ticker = ticker.upper()
    if ticker in _FULL_CACHE:
        log.debug('Full cache HIT (memory): %s', ticker)
        return _FULL_CACHE[ticker]
    csv_path = FULL_CACHE_PATHS.get(ticker)
    if csv_path is None:
        log.error('Unknown ticker: %s', ticker)
        return pd.DataFrame()
    if csv_path.exists():
        df = _load_full_csv(csv_path, ticker)
        if df is not None and (not df.empty):
            _FULL_CACHE[ticker] = df
            return df
        log.warning('Cached CSV for %s exists but could not be loaded.', ticker)
    log.info('%s: full cache CSV not found — running fetch_full_data.py to download.', ticker)
    key = api_key or os.environ.get('TWELVE_DATA_API_KEY', '')
    if not key:
        log.error('Cannot auto-download: TWELVE_DATA_API_KEY not set. Run `python fetch_full_data.py` manually first, or set the env var.')
        return pd.DataFrame()
    script_path = THIS_DIR / 'fetch_full_data.py'
    cmd = [sys.executable, str(script_path), '--tickers', ticker, '--start', start_date, '--api-key', key, '--cache-dir', str(FULL_CACHE_DIR)]
    log.info('Running: %s', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error('fetch_full_data.py exited with code %d', result.returncode)
        return pd.DataFrame()
    if csv_path.exists():
        df = _load_full_csv(csv_path, ticker)
        if df is not None and (not df.empty):
            _FULL_CACHE[ticker] = df
            return df
    log.error('fetch_full_data.py ran but no CSV was produced for %s.', ticker)
    return pd.DataFrame()

def slice_event_window(full_df: pd.DataFrame, event_ts: 'pd.Timestamp | datetime', pre_minutes: int=90, post_minutes: int=90) -> pd.DataFrame:
    if isinstance(event_ts, str):
        event_ts = pd.Timestamp(event_ts, tz=NY_TZ)
    if event_ts.tzinfo is None:
        event_ts = event_ts.replace(tzinfo=NY_TZ)
    else:
        event_ts = event_ts.astimezone(NY_TZ)
    start = event_ts - timedelta(minutes=pre_minutes)
    end = event_ts + timedelta(minutes=post_minutes)
    mask = (full_df.index >= start) & (full_df.index <= end)
    return full_df.loc[mask].copy()

def _load_full_csv(path: Path, ticker: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col='datetime', parse_dates=True)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(NY_TZ)
        else:
            df.index = df.index.tz_convert(NY_TZ)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        log.info('Full cache LOAD: %s | %d bars | %s → %s', ticker, len(df), df.index.min().date(), df.index.max().date())
        return df
    except Exception as exc:
        log.error('Failed to load %s: %s', path, exc)
        return None

class MarketDataFetcher:

    def __init__(self, api_key: str, cache_dir: str='./cache'):
        if not api_key or api_key == 'YOUR_KEY_HERE':
            raise ValueError('Please provide a valid Twelve Data API key.\nGet a free key at: https://twelvedata.com/pricing')
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._last_request_time = 0.0

    def get_bars_around_event(self, ticker: str, event_time: str | datetime, window_minutes: int=60) -> pd.DataFrame:
        ticker = ticker.upper()
        event_dt = self._parse_event_time(event_time)
        start_dt = event_dt - timedelta(minutes=window_minutes)
        end_dt = event_dt + timedelta(minutes=window_minutes)
        log.info(f'Fetching {ticker} | event: {event_dt} | window: ±{window_minutes}min')
        df = self._get_data(ticker, start_dt, end_dt)
        if df.empty:
            log.warning(f'No data returned for {ticker} around {event_dt}')
            return df
        return df
        return df

    def prefetch_events(self, events_df: pd.DataFrame, tickers: list[str]=TICKERS, window_minutes: int=60) -> None:
        timestamps = pd.to_datetime(events_df['timestamp'])
        log.info(f'Prefetching {len(timestamps)} events × {len(tickers)} tickers')
        for ticker in tickers:
            for i, event_time in enumerate(timestamps):
                log.info(f'[{i + 1}/{len(timestamps)}] {ticker} @ {event_time}')
                try:
                    self.get_bars_around_event(ticker, event_time, window_minutes)
                except Exception as e:
                    log.error(f'Failed for {ticker} @ {event_time}: {e}')

    def _get_data(self, ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        start_dt_with_buffer = start_dt - timedelta(days=1)
        end_dt_with_buffer = end_dt + timedelta(days=1)
        dates_needed = self._dates_in_range(start_dt_with_buffer, end_dt_with_buffer)
        frames = []
        for date in dates_needed:
            cached = self._load_from_cache(ticker, date)
            if cached is not None:
                frames.append(cached)
            else:
                fetched = self._fetch_day_from_api(ticker, date)
                if fetched is not None and (not fetched.empty):
                    self._save_to_cache(ticker, date, fetched)
                    frames.append(fetched)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df

    def _fetch_day_from_api(self, ticker: str, date: datetime.date) -> pd.DataFrame | None:
        start_str = f'{date} 07:00:00'
        end_str = f'{date} 16:30:00'
        params = {'symbol': ticker, 'interval': INTERVAL, 'start_date': start_str, 'end_date': end_str, 'timezone': 'America/New_York', 'format': 'JSON', 'outputsize': 200, 'apikey': self.api_key}
        self._rate_limit_wait()
        try:
            response = requests.get(TWELVE_DATA_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            log.error(f'API request failed for {ticker} on {date}: {e}')
            return None
        if payload.get('status') == 'error':
            log.error(f"Twelve Data error for {ticker} on {date}: {payload.get('message', 'unknown error')}")
            return None
        values = payload.get('values')
        if not values:
            log.warning(f'No values in response for {ticker} on {date}')
            return None
        df = self._parse_response(values)
        log.info(f'  API → {ticker} {date}: {len(df)} bars fetched')
        return df

    def _parse_response(self, values: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(values)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_localize(NY_TZ)
        df = df.set_index('datetime').sort_index()
        df = df.rename(columns={'volume': 'volume'})
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def _cache_path(self, ticker: str, date) -> str:
        return os.path.join(self.cache_dir, f'{ticker}_{date}.csv')

    def _load_from_cache(self, ticker: str, date) -> pd.DataFrame | None:
        path = self._cache_path(ticker, date)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col='datetime', parse_dates=True)
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert(NY_TZ)
            else:
                df.index = df.index.tz_localize(NY_TZ)
            log.info(f'  Cache HIT  → {ticker} {date} ({len(df)} bars)')
            return df
        except Exception as e:
            log.warning(f'Cache read failed for {ticker} {date}: {e}')
            return None

    def _save_to_cache(self, ticker: str, date, df: pd.DataFrame) -> None:
        path = self._cache_path(ticker, date)
        try:
            df.to_csv(path)
            log.info(f'  Cache WRITE → {ticker} {date} → {path}')
        except Exception as e:
            log.warning(f'Cache write failed for {ticker} {date}: {e}')

    def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SEC:
            wait = REQUEST_DELAY_SEC - elapsed
            log.debug(f'Rate limit: waiting {wait:.1f}s')
            time.sleep(wait)
        self._last_request_time = time.time()

    @staticmethod
    def _parse_event_time(event_time: str | datetime) -> datetime:
        if isinstance(event_time, str):
            event_time = datetime.fromisoformat(event_time)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=NY_TZ)
        else:
            event_time = event_time.astimezone(NY_TZ)
        return event_time

    @staticmethod
    def _dates_in_range(start_dt: datetime, end_dt: datetime) -> list:
        dates = []
        cursor = start_dt.date()
        while cursor <= end_dt.date():
            dates.append(cursor)
            cursor += timedelta(days=1)
        return dates
if __name__ == '__main__':
    import sys
    API_KEY = os.environ.get('TWELVE_DATA_API_KEY', 'YOUR_KEY_HERE')
    try:
        fetcher = MarketDataFetcher(api_key=API_KEY)
    except ValueError as e:
        print(f'\nERROR: {e}')
        sys.exit(1)
    test_event = '2024-03-20 14:00:00'
    print(f'\nTest: fetching SPY bars around {test_event}\n')
    df = fetcher.get_bars_around_event(ticker='SPY', event_time=test_event, window_minutes=60)
    if df.empty:
        print('No data returned — check your API key or event date.')
    else:
        print(df.head(10).to_string())
        print(f'\n✓ {len(df)} bars returned')

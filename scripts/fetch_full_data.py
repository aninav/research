"""
fetch_full_data.py
==================
Downloads FULL 5-minute historical price data for SPY and QQQ from the
Twelve Data API, then caches it locally as a single monolithic CSV per ticker.

This script is the foundation of the refactored pipeline. Instead of making
one API call per event (leading to 480+ calls for 240 events × 2 tickers),
the entire 2015–2025 5-minute history is fetched ONCE in ~30–60 chunked
requests per ticker, then all event windows are sliced locally with zero
additional API calls.

Architecture
------------
    fetch_full_data.py      ← run once (overnight if needed)
         ↓  writes
    data/cache/SPY_full_5min.csv
    data/cache/QQQ_full_5min.csv
         ↓  read by
    market_data.py          → load_or_fetch_full_data(ticker)
         ↓  sliced by
    build_market_dataset.py → per-event window extraction

Usage
-----
    export TWELVE_DATA_API_KEY=your_key_here
    python fetch_full_data.py                          # default: 2015-01-01 → today
    python fetch_full_data.py --start 2021-01-01       # custom start date
    python fetch_full_data.py --tickers SPY            # single ticker
    python fetch_full_data.py --dry-run                # estimate call count only

API budget
----------
Twelve Data free plan: 800 calls/day, 8 calls/minute.
Each call returns up to 5,000 bars.  5,000 bars × 5 min = ~17 trading days.
10 years of data ≈ 2,500 trading days → ~147 bars per ticker → ~18 calls per ticker.
Two tickers = ~36 total calls.  Well within the daily budget.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)




BASE_URL           = "https://api.twelvedata.com/time_series"
INTERVAL           = "5min"
MAX_BARS_PER_CALL  = 5000
CALLS_PER_MINUTE   = 8
DELAY_BETWEEN_CALLS = 60 / CALLS_PER_MINUTE + 1
MAX_RETRIES        = 3
RETRY_BACKOFF_SEC  = 15

NY_TZ  = ZoneInfo("America/New_York")

THIS_DIR   = Path(__file__).resolve().parent
CACHE_DIR  = THIS_DIR.parent / "data" / "cache"

DEFAULT_TICKERS    = ["SPY", "QQQ"]
DEFAULT_START_DATE = "2015-01-01"






class FullDataFetcher:
    """
    Downloads the complete 5-minute price history for a ticker by chunking
    backward in time.  Each chunk requests MAX_BARS_PER_CALL bars ending at
    the earliest timestamp seen so far; the loop continues until the desired
    start date is reached or the API returns no data.

    Parameters
    ----------
    api_key   : Twelve Data API key
    cache_dir : directory for output CSVs (created if absent)
    """

    def __init__(self, api_key: str, cache_dir: Path = CACHE_DIR) -> None:
        if not api_key:
            raise ValueError(
                "Twelve Data API key is required.\n"
                "Set the TWELVE_DATA_API_KEY environment variable or pass --api-key."
            )
        self.api_key   = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_time = 0.0





    def fetch_ticker(
        self,
        ticker: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Download full history for `ticker` from `start_date` to `end_date`
        (defaults to today).

        Returns the complete DataFrame (also saved to CSV).
        """
        ticker     = ticker.upper()
        start_dt   = pd.Timestamp(start_date, tz=NY_TZ)
        end_dt     = pd.Timestamp(end_date or date.today().isoformat(), tz=NY_TZ)
        end_dt     = end_dt.replace(hour=16, minute=0, second=0)

        log.info("=" * 60)
        log.info("Fetching %s | %s → %s", ticker, start_date, end_dt.date())
        log.info("=" * 60)

        out_path = self.cache_dir / f"{ticker}_full_5min.csv"


        existing = self._load_existing(out_path, ticker)
        if existing is not None:
            earliest_cached = existing.index.min()
            if earliest_cached <= start_dt:
                log.info(
                    "%s: cache already covers requested range "
                    "(earliest bar: %s). Returning cached data.",
                    ticker, earliest_cached,
                )
                return existing

            end_dt = earliest_cached - timedelta(minutes=5)
            log.info(
                "%s: resuming download from %s (earliest cached bar: %s).",
                ticker, end_dt, earliest_cached,
            )

        chunks: list[pd.DataFrame] = []
        chunk_end = end_dt

        while True:
            chunk = self._fetch_chunk(ticker, chunk_end)
            if chunk is None or chunk.empty:
                log.info("%s: API returned no data — download complete.", ticker)
                break

            chunks.append(chunk)
            chunk_earliest = chunk.index.min()
            log.info(
                "%s | chunk bars: %d | window: %s → %s | total bars so far: %d",
                ticker,
                len(chunk),
                chunk_earliest.date(),
                chunk.index.max().date(),
                sum(len(c) for c in chunks),
            )

            if chunk_earliest <= start_dt:
                log.info("%s: reached start date %s. Stopping.", ticker, start_date)
                break


            chunk_end = chunk_earliest - timedelta(minutes=5)

        if not chunks:
            log.error("%s: no data downloaded.", ticker)
            if existing is not None:
                return existing
            return pd.DataFrame()


        all_frames = chunks
        if existing is not None:
            all_frames.append(existing)

        full_df = pd.concat(all_frames)
        full_df = self._clean(full_df, start_dt)


        full_df.to_csv(out_path)
        log.info(
            "%s: saved %d bars to %s  (%.1f MB)",
            ticker,
            len(full_df),
            out_path,
            out_path.stat().st_size / 1e6,
        )
        return full_df





    def _fetch_chunk(
        self,
        ticker: str,
        end_dt: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """
        Fetch up to MAX_BARS_PER_CALL bars ending at `end_dt`.
        Retries up to MAX_RETRIES times on transient failures.
        """
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        params = {
            "symbol":     ticker,
            "interval":   INTERVAL,
            "end_date":   end_str,
            "outputsize": MAX_BARS_PER_CALL,
            "timezone":   "America/New_York",
            "format":     "JSON",
            "apikey":     self.api_key,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                log.warning(
                    "Request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * attempt)
                    continue
                return None

            if payload.get("status") == "error":
                log.error(
                    "Twelve Data API error: %s", payload.get("message", "unknown")
                )
                return None

            values = payload.get("values")
            if not values:
                return None

            df = self._parse(values)
            log.info(
                "  [%s] chunk fetched: %d bars | %s → %s",
                ticker, len(df), df.index.min().date(), df.index.max().date(),
            )
            return df

        return None

    @staticmethod
    def _parse(values: list[dict]) -> pd.DataFrame:
        """Convert raw Twelve Data JSON values to a clean, tz-aware DataFrame."""
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])


        df["datetime"] = df["datetime"].dt.tz_localize(NY_TZ, ambiguous="infer")
        df = df.set_index("datetime").sort_index()

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _clean(df: pd.DataFrame, start_dt: pd.Timestamp) -> pd.DataFrame:
        """
        Deduplicate, sort, filter to regular trading hours + start date,
        and drop rows with missing close prices.
        """
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()


        df = df.between_time("09:30", "16:00")


        df = df[df.index >= start_dt]


        df = df.dropna(subset=["close"])

        return df

    @staticmethod
    def _load_existing(path: Path, ticker: str) -> pd.DataFrame | None:
        """Load a previously saved full CSV if it exists."""
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col="datetime", parse_dates=True)
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize(NY_TZ)
            else:
                df.index = df.index.tz_convert(NY_TZ)
            log.info(
                "%s: loaded existing cache (%d bars, %s → %s)",
                ticker, len(df), df.index.min().date(), df.index.max().date(),
            )
            return df
        except Exception as exc:
            log.warning("Could not load existing cache for %s: %s", ticker, exc)
            return None

    def _rate_limit(self) -> None:
        """Enforce minimum inter-request delay to stay under 8 calls/minute."""
        elapsed = time.time() - self._last_call_time
        if elapsed < DELAY_BETWEEN_CALLS:
            wait = DELAY_BETWEEN_CALLS - elapsed
            log.debug("Rate limit: sleeping %.1f s", wait)
            time.sleep(wait)
        self._last_call_time = time.time()






def estimate_calls(start_date: str, tickers: list[str]) -> None:
    """Print an estimate of API calls needed without making any requests."""
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(date.today().isoformat())
    total_days    = (end - start).days

    trading_days  = int(total_days * 252 / 365)
    bars_per_ticker = trading_days * 78
    calls_per_ticker = -(-bars_per_ticker // MAX_BARS_PER_CALL)

    print("\n── Dry-run estimate ──────────────────────────────────────")
    print(f"  Date range     : {start_date} → {end.date()}")
    print(f"  Trading days   : ~{trading_days}")
    print(f"  Bars / ticker  : ~{bars_per_ticker:,}")
    print(f"  Calls / ticker : ~{calls_per_ticker}  (max {MAX_BARS_PER_CALL} bars/call)")
    print(f"  Tickers        : {tickers}")
    print(f"  Total API calls: ~{calls_per_ticker * len(tickers)}")
    print(f"  Daily budget   : 800 calls  →  {'✓ fits in one run' if calls_per_ticker * len(tickers) < 800 else '⚠ may need two days'}")
    print("──────────────────────────────────────────────────────────\n")






def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download full 5-minute history for SPY/QQQ from Twelve Data API "
            "and save to data/cache/{TICKER}_full_5min.csv"
        )
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=f"Tickers to download. Default: {DEFAULT_TICKERS}",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TWELVE_DATA_API_KEY", ""),
        help="Twelve Data API key. Defaults to TWELVE_DATA_API_KEY env var.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(CACHE_DIR),
        help=f"Directory for output CSVs. Default: {CACHE_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate API call count only — do not fetch any data.",
    )
    args = parser.parse_args()

    if args.dry_run:
        estimate_calls(args.start, args.tickers)
        return

    if not args.api_key:
        log.error(
            "No API key provided. Set TWELVE_DATA_API_KEY environment variable "
            "or pass --api-key <key>."
        )
        sys.exit(1)

    fetcher = FullDataFetcher(api_key=args.api_key, cache_dir=Path(args.cache_dir))

    for ticker in args.tickers:
        df = fetcher.fetch_ticker(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
        )
        if df.empty:
            log.error("No data downloaded for %s.", ticker)
        else:
            log.info(
                "✓ %s complete: %d bars | %s → %s",
                ticker,
                len(df),
                df.index.min().date(),
                df.index.max().date(),
            )

    log.info("All tickers complete.")


if __name__ == "__main__":
    main()

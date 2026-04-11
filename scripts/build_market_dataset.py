"""
build_market_dataset.py
=======================
Event-study pipeline: loads macro event timestamps, downloads 5-minute
SPY/QQQ bars from Twelve Data, and computes pre/post realized volatility
and cumulative returns around each event.

Usage
-----
    python build_market_dataset.py --events ../data/events.csv --output event_market_data.csv

Install dependencies
--------------------
    pip install pandas numpy requests scipy

Environment variable (required)
--------------------------------
    export TWELVE_DATA_API_KEY=your_key_here

    Get a free key at: https://twelvedata.com/pricing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Local data layer — sits in the same scripts/ folder
from market_data import MarketDataFetcher

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ"]

# 5-minute bar counts for each window
BARS_PRE_30M  = 6   # 30 min pre-event  (6 × 5 min)
BARS_POST_5M  = 1   # 5 min post-event  (1 × 5 min)
BARS_POST_30M = 6   # 30 min post-event (6 × 5 min)
BARS_POST_60M = 12  # 60 min post-event (12 × 5 min)

# Cache lives in research_model/data/ (one level up from scripts/)
THIS_DIR  = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR.parent / "data" / "cache"


# ---------------------------------------------------------------------------
# 1. Load events
# ---------------------------------------------------------------------------

def load_events(filepath: str) -> pd.DataFrame:
    """
    Load and validate the events CSV.

    Expected columns
    ----------------
    event_id, timestamp, event_name, event_category, event_flag

    Returns
    -------
    pd.DataFrame with a timezone-aware 'timestamp' column (America/New_York).
    Rows that fail validation are dropped with a logged reason.
    """
    required_cols = {"event_id", "timestamp", "event_name", "event_category", "event_flag"}

    log.info("Loading events from: %s", filepath)
    df = pd.read_csv(filepath, dtype=str)

    # --- column check ---
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"events.csv is missing required columns: {missing}")

    # --- strip whitespace ---
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # --- parse timestamp ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        log.warning("Dropping %d rows with unparseable timestamps: %s",
                    bad_ts.sum(), df.loc[bad_ts, "event_id"].tolist())
    df = df[~bad_ts].copy()

    # --- localise to Eastern (events.csv is assumed ET) ---
    df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")

    # --- duplicate event_id check ---
    dupes = df.duplicated(subset="event_id", keep=False)
    if dupes.any():
        log.warning("Dropping %d rows with duplicate event_id values: %s",
                    dupes.sum(), df.loc[dupes, "event_id"].tolist())
        df = df[~dupes].copy()

    # --- event_flag must be numeric ---
    df["event_flag"] = pd.to_numeric(df["event_flag"], errors="coerce")
    bad_flag = df["event_flag"].isna()
    if bad_flag.any():
        log.warning("Dropping %d rows with non-numeric event_flag: %s",
                    bad_flag.sum(), df.loc[bad_flag, "event_id"].tolist())
        df = df[~bad_flag].copy()

    log.info("Loaded %d valid events.", len(df))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Download market data  ← REPLACED: yfinance → Twelve Data
# ---------------------------------------------------------------------------

def download_market_data(
    fetcher: MarketDataFetcher,
    ticker: str,
    event_ts: pd.Timestamp,
    window_minutes: int = 90,
) -> pd.DataFrame:
    """
    Download intraday 5-minute bars from Twelve Data via MarketDataFetcher.

    Parameters
    ----------
    fetcher        : initialised MarketDataFetcher instance (shared across calls)
    ticker         : e.g. 'SPY'
    event_ts       : the event timestamp (timezone-aware, ET)
    window_minutes : total minutes to fetch on each side of the event.
                     Default 90 min gives comfortable headroom beyond the
                     60-min post-event window.

    Notes
    -----
    - Pre-market events (e.g. CPI at 08:30 ET) are handled automatically
      because Twelve Data returns pre-market bars when they exist.
    - Returns only the 'close' column, matching the original pipeline shape.

    Returns
    -------
    pd.DataFrame with a 'Close' column and a timezone-aware DatetimeIndex (ET).
    Empty DataFrame on failure.
    """
    log.info("Fetching %s bars around %s (±%d min)", ticker, event_ts, window_minutes)

    df = fetcher.get_bars_around_event(
        ticker=ticker,
        event_time=event_ts,
        window_minutes=window_minutes,
    )

    if df.empty:
        log.warning("No data returned for %s around %s", ticker, event_ts)
        return pd.DataFrame()

    # Rename 'close' → 'Close' to preserve compatibility with the rest of the pipeline
    df = df[["close"]].rename(columns={"close": "Close"})
    df.index.name = "datetime"

    return df


# ---------------------------------------------------------------------------
# 3. Log returns  (unchanged)
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute 5-minute log returns:  r_t = ln(P_t / P_{t-1})

    The first value is NaN because there is no prior bar.
    """
    return np.log(prices / prices.shift(1))


# ---------------------------------------------------------------------------
# 4. Realized volatility  (unchanged)
# ---------------------------------------------------------------------------

def compute_realized_volatility(log_returns: pd.Series) -> float:
    """
    Realized volatility = sqrt( sum( r_t^2 ) )

    Drops NaN values before summing. Returns NaN if the series is empty.
    """
    r = log_returns.dropna()
    if r.empty:
        return np.nan
    return float(np.sqrt((r ** 2).sum()))


# ---------------------------------------------------------------------------
# 5. Cumulative return  (unchanged)
# ---------------------------------------------------------------------------

def compute_cumulative_return(prices: pd.Series) -> float:
    """
    Cumulative log return over the window:  ln(P_last / P_first)

    Returns NaN if fewer than 2 prices are available.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan
    return float(np.log(prices.iloc[-1] / prices.iloc[0]))


# ---------------------------------------------------------------------------
# 6. Extract window metrics  (unchanged)
# ---------------------------------------------------------------------------

def extract_event_window_metrics(
    price_series: pd.Series,
    event_ts: pd.Timestamp,
    ticker: str,
    event_id,
) -> dict | None:
    """
    Slice pre/post windows and compute all realized volatility
    and cumulative return metrics for one event + one ticker.

    Parameters
    ----------
    price_series : Close prices with a timezone-aware DatetimeIndex.
                   Should span at least the previous trading day + event day
                   so that pre-market events (08:30 ET) have prior bars.
    event_ts     : the event timestamp (timezone-aware, ET)
    ticker       : 'SPY' or 'QQQ' — used for output column naming
    event_id     : used in log messages only

    Notes
    -----
    For pre-market events (CPI, GDP, Labor at 08:30 ET), Twelve Data's free
    plan does not provide pre-market bars. In this case the pre-event window
    is taken from the PREVIOUS trading day's closing bars. This is a standard
    approach in event-study literature — the pre-event baseline captures the
    market's settled state before the announcement was known.

    Returns
    -------
    dict of computed metrics, or None if the event should be skipped.
    """
    # Normalize index to UTC for comparison — cached data uses fixed offset
    # (UTC-05:00) while event timestamps use America/New_York zone object.
    # Converting both to UTC ensures correct bar lookup regardless of tz type.
    price_utc = price_series.copy()
    price_utc.index = price_series.index.tz_convert("UTC")
    event_utc = event_ts.astimezone(ZoneInfo("UTC"))

    # First bar at or after the event timestamp
    future_bars = price_utc.index[price_utc.index >= event_utc]
    if future_bars.empty:
        log.warning("SKIP event_id=%s (%s): no bars at or after event timestamp.", event_id, ticker)
        return None

    event_bar_idx = price_utc.index.get_loc(future_bars[0])

    # --- pre-event slice ---
    # For pre-market events there are 0 same-day bars before the event.
    # We walk backwards in the full (multi-day) price series to find
    # BARS_PRE_30M bars, which will come from the previous day's close.
    pre_start_idx = event_bar_idx - BARS_PRE_30M
    if pre_start_idx < 0:
        log.info(
            "event_id=%s (%s): pre-market event — using previous day's closing "
            "bars as pre-event baseline (have %d bars before event in series).",
            event_id, ticker, event_bar_idx,
        )
        # Use however many bars we have before the event, up to BARS_PRE_30M
        pre_start_idx = max(0, event_bar_idx - BARS_PRE_30M)
        if event_bar_idx == 0:
            log.warning("SKIP event_id=%s (%s): no bars at all before event.", event_id, ticker)
            return None

    # --- post-event slice ---
    post_end_idx = event_bar_idx + BARS_POST_60M
    if post_end_idx > len(price_series):
        log.warning("SKIP event_id=%s (%s): insufficient post-event bars (need %d, have %d after event).",
                    event_id, ticker, BARS_POST_60M, len(price_series) - event_bar_idx)
        return None

    # --- price slices ---
    pre_prices = price_series.iloc[pre_start_idx : event_bar_idx]
    post_5m    = price_series.iloc[event_bar_idx : event_bar_idx + BARS_POST_5M  + 1]
    post_30m   = price_series.iloc[event_bar_idx : event_bar_idx + BARS_POST_30M + 1]
    post_60m   = price_series.iloc[event_bar_idx : event_bar_idx + BARS_POST_60M + 1]

    # --- log returns ---
    pre_prices_with_anchor      = price_series.iloc[max(0, pre_start_idx - 1) : event_bar_idx]
    pre_returns                 = compute_log_returns(pre_prices_with_anchor)

    post_prices_with_anchor_5m  = price_series.iloc[event_bar_idx - 1 : event_bar_idx + BARS_POST_5M  + 1]
    post_prices_with_anchor_30m = price_series.iloc[event_bar_idx - 1 : event_bar_idx + BARS_POST_30M + 1]
    post_prices_with_anchor_60m = price_series.iloc[event_bar_idx - 1 : event_bar_idx + BARS_POST_60M + 1]

    post_returns_5m  = compute_log_returns(post_prices_with_anchor_5m)
    post_returns_30m = compute_log_returns(post_prices_with_anchor_30m)
    post_returns_60m = compute_log_returns(post_prices_with_anchor_60m)

    t = ticker
    return {
        f"{t}_rv_pre_30m"  : compute_realized_volatility(pre_returns),
        f"{t}_rv_post_5m"  : compute_realized_volatility(post_returns_5m),
        f"{t}_rv_post_30m" : compute_realized_volatility(post_returns_30m),
        f"{t}_rv_post_60m" : compute_realized_volatility(post_returns_60m),
        f"{t}_ret_5m"      : compute_cumulative_return(post_5m),
        f"{t}_ret_30m"     : compute_cumulative_return(post_30m),
        f"{t}_ret_60m"     : compute_cumulative_return(post_60m),
    }


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(events_path: str, output_path: str, limit: int = None) -> pd.DataFrame:
    """
    Full pipeline:
      1. Load events
      2. Initialise MarketDataFetcher (Twelve Data + local cache)
      3. Prefetch all event windows in bulk (respects rate limits)
      4. Extract window metrics per event
      5. Save to CSV

    Returns the final DataFrame.
    """
    events = load_events(events_path)

    if events.empty:
        log.error("No valid events found. Exiting.")
        sys.exit(1)

    if limit:
        events = events.head(limit)
        log.info("Limit set: processing first %d events only.", limit)

    # --- initialise the Twelve Data fetcher ---
    api_key = os.environ.get("TWELVE_DATA_API_KEY", "")
    if not api_key:
        log.error(
            "TWELVE_DATA_API_KEY environment variable not set.\n"
            "  Run:  export TWELVE_DATA_API_KEY=your_key_here\n"
            "  Get a free key at: https://twelvedata.com/pricing"
        )
        sys.exit(1)

    fetcher = MarketDataFetcher(api_key=api_key, cache_dir=str(CACHE_DIR))

    # --- bulk prefetch: download + cache everything before the analysis loop ---
    log.info("=== Prefetching market data for all events ===")
    fetcher.prefetch_events(events_df=events, tickers=TICKERS, window_minutes=90)

    # --- main event loop ---
    log.info("=== Computing event window metrics ===")
    results = []

    for _, row in events.iterrows():
        event_id   = row["event_id"]
        event_ts   = row["timestamp"]
        event_name = row["event_name"]

        log.info("--- event_id=%s | %s | %s ---", event_id, event_name, event_ts.date())

        row_metrics = {
            "event_id"       : event_id,
            "timestamp"      : event_ts,
            "event_name"     : event_name,
            "event_category" : row["event_category"],
            "event_flag"     : row["event_flag"],
        }

        valid_event = True

        for ticker in TICKERS:
            price_df = download_market_data(
                fetcher=fetcher,
                ticker=ticker,
                event_ts=event_ts,
                window_minutes=90,
            )

            if price_df.empty:
                log.warning("SKIP event_id=%s (%s): no market data.", event_id, ticker)
                valid_event = False
                break

            metrics = extract_event_window_metrics(
                price_series=price_df["Close"],
                event_ts=event_ts,
                ticker=ticker,
                event_id=event_id,
            )

            if metrics is None:
                valid_event = False
                break

            row_metrics.update(metrics)

        if valid_event:
            results.append(row_metrics)
        else:
            log.warning("Event event_id=%s skipped entirely.", event_id)

    if not results:
        log.error("No events produced valid metrics. Output file will not be created.")
        sys.exit(1)

    output_df = pd.DataFrame(results)

    # Reorder columns for readability
    id_cols     = ["event_id", "timestamp", "event_name", "event_category", "event_flag"]
    metric_cols = [c for c in output_df.columns if c not in id_cols]
    output_df   = output_df[id_cols + sorted(metric_cols)]

    output_df.to_csv(output_path, index=False)
    log.info("Saved %d event rows to: %s", len(output_df), output_path)

    return output_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Event-study pipeline: macro announcements × SPY/QQQ volatility."
    )
    parser.add_argument(
        "--events",
        default=str(THIS_DIR.parent / "data" / "events.csv"),
        help="Path to the input events CSV (default: ../data/events.csv)",
    )
    parser.add_argument(
        "--output",
        default="event_market_data.csv",
        help="Path to write the output CSV (default: event_market_data.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N events (default: all). Use for test runs.",
    )
    args = parser.parse_args()

    if not Path(args.events).exists():
        log.error("Events file not found: %s", args.events)
        sys.exit(1)

    df = build_dataset(events_path=args.events, output_path=args.output, limit=args.limit)

    print("\n=== Preview (first 5 rows) ===")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

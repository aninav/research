"""
build_market_dataset.py
=======================
Event-study pipeline: loads macro event timestamps, downloads 5-minute
SPY/QQQ bars from Yahoo Finance, and computes pre/post realized volatility
and cumulative returns around each event.

Usage
-----
    python build_market_dataset.py --events events.csv --output event_market_data.csv

Install dependencies
--------------------
    pip install pandas numpy yfinance scipy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf




logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)





TICKERS = ["SPY", "QQQ"]


BARS_PRE_30M   = 6
BARS_POST_5M   = 1
BARS_POST_30M  = 6
BARS_POST_60M  = 12


MIN_BARS_REQUIRED = max(BARS_PRE_30M, BARS_POST_60M)


FETCH_BUFFER_DAYS = 2






def load_events(filepath: str) -> pd.DataFrame:
    """
    Load and validate the events CSV.

    Expected columns
    ----------------
    event_id, timestamp, event_name, event_category, event_flag

    Returns
    -------
    pd.DataFrame with a timezone-aware 'timestamp' column (US/Eastern).
    Rows that fail validation are dropped and the reason is logged.
    """
    required_cols = {"event_id", "timestamp", "event_name", "event_category", "event_flag"}

    log.info("Loading events from: %s", filepath)
    df = pd.read_csv(filepath, dtype=str)


    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"events.csv is missing required columns: {missing}")


    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)


    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        log.warning("Dropping %d rows with unparseable timestamps: %s",
                    bad_ts.sum(), df.loc[bad_ts, "event_id"].tolist())
    df = df[~bad_ts].copy()


    df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")


    dupes = df.duplicated(subset="event_id", keep=False)
    if dupes.any():
        log.warning("Dropping %d rows with duplicate event_id values: %s",
                    dupes.sum(), df.loc[dupes, "event_id"].tolist())
        df = df[~dupes].copy()


    df["event_flag"] = pd.to_numeric(df["event_flag"], errors="coerce")
    bad_flag = df["event_flag"].isna()
    if bad_flag.any():
        log.warning("Dropping %d rows with non-numeric event_flag: %s",
                    bad_flag.sum(), df.loc[bad_flag, "event_id"].tolist())
        df = df[~bad_flag].copy()

    log.info("Loaded %d valid events.", len(df))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df






def download_market_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Download intraday OHLCV bars from Yahoo Finance.

    Parameters
    ----------
    ticker   : e.g. 'SPY'
    start    : 'YYYY-MM-DD' string
    end      : 'YYYY-MM-DD' string  (exclusive in yfinance)
    interval : '5m' for 5-minute bars

    Notes
    -----
    - prepost=True is set so that 8:30 AM ET macro releases (pre-market) are
      included.  Without it the first tradeable bar is ~9:30 AM and CPI /
      NFP / etc. would all be missed.
    - We use the 'Close' price throughout.  For 5-minute event studies this is
      more robust than 'Open' (which can gap) and 'High'/'Low' (which add
      noise unrelated to directional reactions).

    Returns
    -------
    pd.DataFrame indexed by a timezone-aware DatetimeIndex (US/Eastern).
    """
    log.info("Downloading %s  %s  %s → %s", ticker, interval, start, end)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        prepost=True,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        log.warning("No data returned for %s (%s → %s).", ticker, start, end)
        return pd.DataFrame()



    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw[["Close"]].copy()
    raw.index.name = "datetime"


    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        raw.index = raw.index.tz_convert("America/New_York")

    raw = raw.sort_index()
    return raw






def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute 5-minute log returns:  r_t = ln(P_t / P_{t-1})

    The first value is NaN because there is no prior bar.
    """
    return np.log(prices / prices.shift(1))






def compute_realized_volatility(log_returns: pd.Series) -> float:
    """
    Realized volatility = sqrt( sum( r_t^2 ) )

    We drop NaN values before summing.  Returns NaN if the series is empty
    or contains only NaN.
    """
    r = log_returns.dropna()
    if r.empty:
        return np.nan
    return float(np.sqrt((r ** 2).sum()))






def compute_cumulative_return(prices: pd.Series) -> float:
    """
    Cumulative log return over the window:  ln(P_last / P_first)

    Uses the first and last price in the supplied slice.
    Returns NaN if fewer than 2 prices are available.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan
    return float(np.log(prices.iloc[-1] / prices.iloc[0]))






def extract_event_window_metrics(
    price_series: pd.Series,
    event_ts: pd.Timestamp,
    ticker: str,
    event_id,
) -> dict | None:
    """
    Given a price series and an event timestamp, slice pre/post windows and
    compute all realized volatility and cumulative return metrics.

    Parameters
    ----------
    price_series : Close prices with a timezone-aware DatetimeIndex
    event_ts     : the event timestamp (timezone-aware, ET)
    ticker       : 'SPY' or 'QQQ'  — used only for naming output columns
    event_id     : used in log messages

    Returns
    -------
    dict of computed metrics, or None if the event should be skipped.
    """


    future_bars = price_series.index[price_series.index >= event_ts]
    if future_bars.empty:
        log.warning("SKIP event_id=%s (%s): no bars at or after event timestamp.", event_id, ticker)
        return None

    event_bar_idx = price_series.index.get_loc(future_bars[0])


    pre_start_idx = event_bar_idx - BARS_PRE_30M
    if pre_start_idx < 0:
        log.warning("SKIP event_id=%s (%s): insufficient pre-event bars (need %d, have %d).",
                    event_id, ticker, BARS_PRE_30M, event_bar_idx)
        return None


    post_end_idx = event_bar_idx + BARS_POST_60M
    if post_end_idx > len(price_series):
        log.warning("SKIP event_id=%s (%s): insufficient post-event bars (need %d, have %d after event).",
                    event_id, ticker, BARS_POST_60M, len(price_series) - event_bar_idx)
        return None


    pre_prices   = price_series.iloc[pre_start_idx : event_bar_idx]
    post_5m      = price_series.iloc[event_bar_idx : event_bar_idx + BARS_POST_5M  + 1]
    post_30m     = price_series.iloc[event_bar_idx : event_bar_idx + BARS_POST_30M + 1]
    post_60m     = price_series.iloc[event_bar_idx : event_bar_idx + BARS_POST_60M + 1]




    pre_prices_with_anchor = price_series.iloc[max(0, pre_start_idx - 1) : event_bar_idx]
    pre_returns = compute_log_returns(pre_prices_with_anchor)



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






def build_dataset(events_path: str, output_path: str) -> pd.DataFrame:
    """
    Full pipeline:
      1. Load events
      2. For each event date, download SPY + QQQ 5-minute bars
      3. Extract window metrics per event
      4. Save to CSV

    Returns the final DataFrame.
    """
    events = load_events(events_path)

    if events.empty:
        log.error("No valid events found. Exiting.")
        sys.exit(1)

    results = []

    for _, row in events.iterrows():
        event_id   = row["event_id"]
        event_ts   = row["timestamp"]
        event_name = row["event_name"]


        date_str  = event_ts.date().isoformat()
        start_dt  = (event_ts - pd.Timedelta(days=FETCH_BUFFER_DAYS)).date().isoformat()
        end_dt    = (event_ts + pd.Timedelta(days=FETCH_BUFFER_DAYS + 1)).date().isoformat()

        log.info("--- Processing event_id=%s  |  %s  |  %s ---", event_id, event_name, date_str)

        row_metrics = {
            "event_id"       : event_id,
            "timestamp"      : event_ts,
            "event_name"     : event_name,
            "event_category" : row["event_category"],
            "event_flag"     : row["event_flag"],
        }

        valid_event = True

        for ticker in TICKERS:
            price_df = download_market_data(ticker, start=start_dt, end=end_dt)

            if price_df.empty:
                log.warning("SKIP event_id=%s (%s): could not download market data.", event_id, ticker)
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


    id_cols     = ["event_id", "timestamp", "event_name", "event_category", "event_flag"]
    metric_cols = [c for c in output_df.columns if c not in id_cols]
    output_df   = output_df[id_cols + sorted(metric_cols)]

    output_df.to_csv(output_path, index=False)
    log.info("Saved %d event rows to: %s", len(output_df), output_path)

    return output_df






def main():
    parser = argparse.ArgumentParser(
        description="Event-study pipeline: macro announcements × SPY/QQQ volatility."
    )
    parser.add_argument(
        "--events",
        default="events.csv",
        help="Path to the input events CSV (default: events.csv)",
    )
    parser.add_argument(
        "--output",
        default="event_market_data.csv",
        help="Path to write the output CSV (default: event_market_data.csv)",
    )
    args = parser.parse_args()

    if not Path(args.events).exists():
        log.error("Events file not found: %s", args.events)
        sys.exit(1)

    df = build_dataset(events_path=args.events, output_path=args.output)

    print("\n=== Preview (first 5 rows) ===")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

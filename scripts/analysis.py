"""
analysis.py
===========
Statistical analysis for Research Question 1:
"Do macroeconomic announcements produce statistically measurable
short-term changes in volatility and returns in equity index ETFs?"

Outputs (all saved to ../data/tables/):
- table1_summary_stats.csv       : mean/std/min/max of all RV and return cols
- table2_rv_by_category.csv      : mean pre vs post RV by event category
- table3_ttest_results.csv       : t-test results (pre vs post RV) per ticker/window
- table4_return_by_category.csv  : mean cumulative returns by event category
- figures/                       : volatility comparison bar charts (matplotlib)

Usage
-----
    python analysis.py

Dependencies
------------
    pip install pandas numpy scipy matplotlib seaborn
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR.parent / "data"
TABLES_DIR  = DATA_DIR / "tables"
FIGURES_DIR = DATA_DIR / "figures"
MASTER_PATH = DATA_DIR / "master_dataset.csv"

TICKERS  = ["SPY", "QQQ"]
WINDOWS  = ["5m", "30m", "60m"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error("master_dataset.csv not found. Run text_features.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    log.info("Loaded master dataset: %d rows × %d cols", len(df), len(df.columns))
    return df


def rv_cols(ticker: str) -> dict:
    return {
        "pre_30m"  : f"{ticker}_rv_pre_30m",
        "post_5m"  : f"{ticker}_rv_post_5m",
        "post_30m" : f"{ticker}_rv_post_30m",
        "post_60m" : f"{ticker}_rv_post_60m",
    }


def ret_cols(ticker: str) -> dict:
    return {
        "5m"  : f"{ticker}_ret_5m",
        "30m" : f"{ticker}_ret_30m",
        "60m" : f"{ticker}_ret_60m",
    }


# ---------------------------------------------------------------------------
# Table 1: Summary statistics
# ---------------------------------------------------------------------------

def table1_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive statistics for all realized volatility and return columns.
    Rows = metric, Cols = count / mean / std / min / 25% / 75% / max
    """
    metric_cols = [c for c in df.columns if "_rv_" in c or "_ret_" in c]
    summary = df[metric_cols].describe().T
    summary.index.name = "metric"
    summary = summary.round(6)

    out = TABLES_DIR / "table1_summary_stats.csv"
    summary.to_csv(out)
    log.info("Table 1 saved: %s", out)
    return summary


# ---------------------------------------------------------------------------
# Table 2: Pre vs post RV by event category
# ---------------------------------------------------------------------------

def table2_rv_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event category, show mean pre-event RV vs post-event RV
    for both SPY and QQQ across all post windows.
    """
    rows = []
    for ticker in TICKERS:
        cols = rv_cols(ticker)
        for category, grp in df.groupby("event_category"):
            row = {
                "ticker"        : ticker,
                "event_category": category,
                "n_events"      : len(grp),
                "rv_pre_30m"    : grp[cols["pre_30m"]].mean(),
                "rv_post_5m"    : grp[cols["post_5m"]].mean(),
                "rv_post_30m"   : grp[cols["post_30m"]].mean(),
                "rv_post_60m"   : grp[cols["post_60m"]].mean(),
            }
            # ratio: how much larger is post-30m RV vs pre-30m RV?
            if row["rv_pre_30m"] and row["rv_pre_30m"] > 0:
                row["rv_ratio_30m"] = row["rv_post_30m"] / row["rv_pre_30m"]
            else:
                row["rv_ratio_30m"] = np.nan
            rows.append(row)

    result = pd.DataFrame(rows).round(6)
    out = TABLES_DIR / "table2_rv_by_category.csv"
    result.to_csv(out, index=False)
    log.info("Table 2 saved: %s", out)
    return result


# ---------------------------------------------------------------------------
# Table 3: T-tests — pre vs post RV
# ---------------------------------------------------------------------------

def table3_ttest_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t-test: pre-event RV (30m) vs post-event RV for each window.
    H0: mean(post_RV) = mean(pre_RV)
    H1: mean(post_RV) > mean(pre_RV)  [one-sided]

    Also runs by event category for granular results.
    """
    rows = []

    subsets = {"All events": df}
    for cat in df["event_category"].unique():
        subsets[cat] = df[df["event_category"] == cat]

    for label, subset in subsets.items():
        for ticker in TICKERS:
            cols   = rv_cols(ticker)
            pre    = subset[cols["pre_30m"]].dropna()

            for window in ["post_5m", "post_30m", "post_60m"]:
                post = subset[cols[window]].dropna()

                # align indices for paired test
                aligned = pd.concat([pre, post], axis=1).dropna()
                if len(aligned) < 5:
                    continue

                pre_vals  = aligned.iloc[:, 0]
                post_vals = aligned.iloc[:, 1]

                t_stat, p_two = stats.ttest_rel(post_vals, pre_vals)
                p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

                rows.append({
                    "subset"         : label,
                    "ticker"         : ticker,
                    "window"         : window,
                    "n"              : len(aligned),
                    "mean_pre_rv"    : pre_vals.mean(),
                    "mean_post_rv"   : post_vals.mean(),
                    "mean_diff"      : (post_vals - pre_vals).mean(),
                    "t_statistic"    : t_stat,
                    "p_value_2sided" : p_two,
                    "p_value_1sided" : p_one,
                    "significant_5pct": int(p_one < 0.05),
                    "significant_1pct": int(p_one < 0.01),
                })

    result = pd.DataFrame(rows).round(6)
    out = TABLES_DIR / "table3_ttest_results.csv"
    result.to_csv(out, index=False)
    log.info("Table 3 saved: %s", out)
    return result


# ---------------------------------------------------------------------------
# Table 4: Cumulative returns by event category
# ---------------------------------------------------------------------------

def table4_return_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean cumulative returns by event category and window.
    Useful for understanding directional market reactions.
    """
    rows = []
    for ticker in TICKERS:
        cols = ret_cols(ticker)
        for category, grp in df.groupby("event_category"):
            row = {
                "ticker"         : ticker,
                "event_category" : category,
                "n_events"       : len(grp),
                "mean_ret_5m"    : grp[cols["5m"]].mean(),
                "mean_ret_30m"   : grp[cols["30m"]].mean(),
                "mean_ret_60m"   : grp[cols["60m"]].mean(),
                "pct_positive_30m": (grp[cols["30m"]] > 0).mean(),
            }
            rows.append(row)

    result = pd.DataFrame(rows).round(6)
    out = TABLES_DIR / "table4_return_by_category.csv"
    result.to_csv(out, index=False)
    log.info("Table 4 saved: %s", out)
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure1_rv_comparison(df: pd.DataFrame) -> None:
    """
    Bar chart: mean pre vs post-30m realized volatility by event category.
    One panel per ticker.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mean Realized Volatility: Pre vs Post Event (30-min window)", fontsize=13)

    categories = sorted(df["event_category"].unique())
    x = np.arange(len(categories))
    width = 0.35

    for ax, ticker in zip(axes, TICKERS):
        cols    = rv_cols(ticker)
        pre_rv  = [df[df["event_category"] == c][cols["pre_30m"]].mean() for c in categories]
        post_rv = [df[df["event_category"] == c][cols["post_30m"]].mean() for c in categories]

        ax.bar(x - width/2, pre_rv,  width, label="Pre-event (30m)",  color="#4C8BE8", alpha=0.85)
        ax.bar(x + width/2, post_rv, width, label="Post-event (30m)", color="#E8634C", alpha=0.85)
        ax.set_title(ticker, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Realized Volatility")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "figure1_rv_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure 1 saved: %s", out)


def figure2_rv_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap: RV ratio (post_30m / pre_30m) by event category × ticker.
    Higher ratio = bigger volatility spike.
    """
    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed — skipping heatmap. pip install seaborn")
        return

    ratios = {}
    for ticker in TICKERS:
        cols = rv_cols(ticker)
        ratios[ticker] = df.groupby("event_category").apply(
            lambda g: g[cols["post_30m"]].mean() / g[cols["pre_30m"]].mean()
        )

    ratio_df = pd.DataFrame(ratios).round(2)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(ratio_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=1.0, ax=ax, linewidths=0.5)
    ax.set_title("Volatility Spike Ratio (post-30m RV / pre-30m RV)", fontsize=11)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Event Category")

    plt.tight_layout()
    out = FIGURES_DIR / "figure2_rv_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure 2 saved: %s", out)


def figure3_return_distribution(df: pd.DataFrame) -> None:
    """
    Histogram of 30-minute post-event returns for SPY and QQQ.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Distribution of 30-min Post-Event Returns", fontsize=13)

    for ax, ticker in zip(axes, TICKERS):
        col  = f"{ticker}_ret_30m"
        data = df[col].dropna() * 100  # convert to percent

        ax.hist(data, bins=20, color="#4C8BE8", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.axvline(data.mean(), color="#E8634C", linewidth=1.5,
                   linestyle="-", label=f"Mean: {data.mean():.3f}%")
        ax.set_title(ticker, fontsize=12)
        ax.set_xlabel("30-min Return (%)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "figure3_return_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure 3 saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_master(MASTER_PATH)

    log.info("=== Generating tables ===")
    t1 = table1_summary_stats(df)
    t2 = table2_rv_by_category(df)
    t3 = table3_ttest_results(df)
    t4 = table4_return_by_category(df)

    log.info("=== Generating figures ===")
    figure1_rv_comparison(df)
    figure2_rv_heatmap(df)
    figure3_return_distribution(df)

    log.info("=== Analysis complete ===")
    print("\n=== T-test results (all events) ===")
    all_events = t3[t3["subset"] == "All events"]
    print(all_events[["ticker", "window", "n", "mean_pre_rv", "mean_post_rv",
                       "t_statistic", "p_value_1sided", "significant_5pct"]].to_string(index=False))

    print(f"\nAll tables saved to: {TABLES_DIR}")
    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_analysis()

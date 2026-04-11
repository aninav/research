"""
generate_events.py
==================
Generates a complete events.csv file with historical macroeconomic
announcement dates from 2021–2025.

Covers:
- CPI (Bureau of Labor Statistics, ~8:30 ET)
- FOMC Rate Decision (Federal Reserve, ~14:00 ET)
- GDP Advance Estimate (BEA, ~8:30 ET)
- Labor Report / NFP (BLS, ~8:30 ET)
- Retail Sales (Census Bureau, ~8:30 ET)

Usage
-----
    python generate_events.py

Output
------
    ../data/events.csv   (overwrites existing file)

Notes
-----
All dates are sourced from official release calendars:
- BLS: https://www.bls.gov/schedule/news_release/cpi.htm
- Federal Reserve: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- BEA: https://www.bea.gov/news/schedule
- Census Bureau: https://www.census.gov/retail/index.html
"""

import csv
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
THIS_DIR    = Path(__file__).resolve().parent
OUTPUT_PATH = THIS_DIR.parent / "data" / "events.csv"

# ---------------------------------------------------------------------------
# Event definitions
# All timestamps in America/New_York (ET)
# Format: (event_name, event_category, timestamp_str)
# ---------------------------------------------------------------------------

CPI_RELEASES = [
    # 2021
    ("CPI", "inflation", "2021-01-13 08:30:00"),
    ("CPI", "inflation", "2021-02-10 08:30:00"),
    ("CPI", "inflation", "2021-03-10 08:30:00"),
    ("CPI", "inflation", "2021-04-13 08:30:00"),
    ("CPI", "inflation", "2021-05-12 08:30:00"),
    ("CPI", "inflation", "2021-06-10 08:30:00"),
    ("CPI", "inflation", "2021-07-13 08:30:00"),
    ("CPI", "inflation", "2021-08-11 08:30:00"),
    ("CPI", "inflation", "2021-09-14 08:30:00"),
    ("CPI", "inflation", "2021-10-13 08:30:00"),
    ("CPI", "inflation", "2021-11-10 08:30:00"),
    ("CPI", "inflation", "2021-12-10 08:30:00"),
    # 2022
    ("CPI", "inflation", "2022-01-12 08:30:00"),
    ("CPI", "inflation", "2022-02-10 08:30:00"),
    ("CPI", "inflation", "2022-03-10 08:30:00"),
    ("CPI", "inflation", "2022-04-12 08:30:00"),
    ("CPI", "inflation", "2022-05-11 08:30:00"),
    ("CPI", "inflation", "2022-06-10 08:30:00"),
    ("CPI", "inflation", "2022-07-13 08:30:00"),
    ("CPI", "inflation", "2022-08-10 08:30:00"),
    ("CPI", "inflation", "2022-09-13 08:30:00"),
    ("CPI", "inflation", "2022-10-13 08:30:00"),
    ("CPI", "inflation", "2022-11-10 08:30:00"),
    ("CPI", "inflation", "2022-12-13 08:30:00"),
    # 2023
    ("CPI", "inflation", "2023-01-12 08:30:00"),
    ("CPI", "inflation", "2023-02-14 08:30:00"),
    ("CPI", "inflation", "2023-03-14 08:30:00"),
    ("CPI", "inflation", "2023-04-12 08:30:00"),
    ("CPI", "inflation", "2023-05-10 08:30:00"),
    ("CPI", "inflation", "2023-06-13 08:30:00"),
    ("CPI", "inflation", "2023-07-12 08:30:00"),
    ("CPI", "inflation", "2023-08-10 08:30:00"),
    ("CPI", "inflation", "2023-09-13 08:30:00"),
    ("CPI", "inflation", "2023-10-12 08:30:00"),
    ("CPI", "inflation", "2023-11-14 08:30:00"),
    ("CPI", "inflation", "2023-12-12 08:30:00"),
    # 2024
    ("CPI", "inflation", "2024-01-11 08:30:00"),
    ("CPI", "inflation", "2024-02-13 08:30:00"),
    ("CPI", "inflation", "2024-03-12 08:30:00"),
    ("CPI", "inflation", "2024-04-10 08:30:00"),
    ("CPI", "inflation", "2024-05-15 08:30:00"),
    ("CPI", "inflation", "2024-06-12 08:30:00"),
    ("CPI", "inflation", "2024-07-11 08:30:00"),
    ("CPI", "inflation", "2024-08-14 08:30:00"),
    ("CPI", "inflation", "2024-09-11 08:30:00"),
    ("CPI", "inflation", "2024-10-10 08:30:00"),
    ("CPI", "inflation", "2024-11-13 08:30:00"),
    ("CPI", "inflation", "2024-12-11 08:30:00"),
    # 2025
    ("CPI", "inflation", "2025-01-15 08:30:00"),
    ("CPI", "inflation", "2025-02-12 08:30:00"),
    ("CPI", "inflation", "2025-03-12 08:30:00"),
    ("CPI", "inflation", "2025-04-10 08:30:00"),
    ("CPI", "inflation", "2025-05-13 08:30:00"),
    ("CPI", "inflation", "2025-06-11 08:30:00"),
    ("CPI", "inflation", "2025-07-15 08:30:00"),
    ("CPI", "inflation", "2025-08-12 08:30:00"),
    ("CPI", "inflation", "2025-09-10 08:30:00"),
    ("CPI", "inflation", "2025-10-15 08:30:00"),
    ("CPI", "inflation", "2025-11-13 08:30:00"),
    ("CPI", "inflation", "2025-12-10 08:30:00"),
]

FOMC_RELEASES = [
    # 2021
    ("FOMC Rate Decision", "monetary_policy", "2021-01-27 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-03-17 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-04-28 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-06-16 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-07-28 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-09-22 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-11-03 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2021-12-15 14:00:00"),
    # 2022
    ("FOMC Rate Decision", "monetary_policy", "2022-01-26 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-03-16 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-05-04 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-06-15 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-07-27 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-09-21 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-11-02 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2022-12-14 14:00:00"),
    # 2023
    ("FOMC Rate Decision", "monetary_policy", "2023-02-01 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-03-22 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-05-03 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-06-14 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-07-26 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-09-20 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-11-01 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2023-12-13 14:00:00"),
    # 2024
    ("FOMC Rate Decision", "monetary_policy", "2024-01-31 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-03-20 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-05-01 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-06-12 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-07-31 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-09-18 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-11-07 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2024-12-18 14:00:00"),
    # 2025
    ("FOMC Rate Decision", "monetary_policy", "2025-01-29 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-03-19 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-05-07 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-06-18 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-07-30 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-09-17 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-10-29 14:00:00"),
    ("FOMC Rate Decision", "monetary_policy", "2025-12-10 14:00:00"),
]

GDP_RELEASES = [
    # 2021 (advance estimates, Q4 2020 through Q3 2021)
    ("GDP", "growth", "2021-01-28 08:30:00"),
    ("GDP", "growth", "2021-04-29 08:30:00"),
    ("GDP", "growth", "2021-07-29 08:30:00"),
    ("GDP", "growth", "2021-10-28 08:30:00"),
    # 2022
    ("GDP", "growth", "2022-01-27 08:30:00"),
    ("GDP", "growth", "2022-04-28 08:30:00"),
    ("GDP", "growth", "2022-07-28 08:30:00"),
    ("GDP", "growth", "2022-10-27 08:30:00"),
    # 2023
    ("GDP", "growth", "2023-01-26 08:30:00"),
    ("GDP", "growth", "2023-04-27 08:30:00"),
    ("GDP", "growth", "2023-07-27 08:30:00"),
    ("GDP", "growth", "2023-10-26 08:30:00"),
    # 2024
    ("GDP", "growth", "2024-01-25 08:30:00"),
    ("GDP", "growth", "2024-04-25 08:30:00"),
    ("GDP", "growth", "2024-07-25 08:30:00"),
    ("GDP", "growth", "2024-10-30 08:30:00"),
    # 2025
    ("GDP", "growth", "2025-01-30 08:30:00"),
    ("GDP", "growth", "2025-04-30 08:30:00"),
    ("GDP", "growth", "2025-07-30 08:30:00"),
    ("GDP", "growth", "2025-10-30 08:30:00"),
]

LABOR_RELEASES = [
    # 2021
    ("Labor Report", "labor", "2021-01-08 08:30:00"),
    ("Labor Report", "labor", "2021-02-05 08:30:00"),
    ("Labor Report", "labor", "2021-03-05 08:30:00"),
    ("Labor Report", "labor", "2021-04-02 08:30:00"),
    ("Labor Report", "labor", "2021-05-07 08:30:00"),
    ("Labor Report", "labor", "2021-06-04 08:30:00"),
    ("Labor Report", "labor", "2021-07-02 08:30:00"),
    ("Labor Report", "labor", "2021-08-06 08:30:00"),
    ("Labor Report", "labor", "2021-09-03 08:30:00"),
    ("Labor Report", "labor", "2021-10-08 08:30:00"),
    ("Labor Report", "labor", "2021-11-05 08:30:00"),
    ("Labor Report", "labor", "2021-12-03 08:30:00"),
    # 2022
    ("Labor Report", "labor", "2022-01-07 08:30:00"),
    ("Labor Report", "labor", "2022-02-04 08:30:00"),
    ("Labor Report", "labor", "2022-03-04 08:30:00"),
    ("Labor Report", "labor", "2022-04-01 08:30:00"),
    ("Labor Report", "labor", "2022-05-06 08:30:00"),
    ("Labor Report", "labor", "2022-06-03 08:30:00"),
    ("Labor Report", "labor", "2022-07-08 08:30:00"),
    ("Labor Report", "labor", "2022-08-05 08:30:00"),
    ("Labor Report", "labor", "2022-09-02 08:30:00"),
    ("Labor Report", "labor", "2022-10-07 08:30:00"),
    ("Labor Report", "labor", "2022-11-04 08:30:00"),
    ("Labor Report", "labor", "2022-12-02 08:30:00"),
    # 2023
    ("Labor Report", "labor", "2023-01-06 08:30:00"),
    ("Labor Report", "labor", "2023-02-03 08:30:00"),
    ("Labor Report", "labor", "2023-03-10 08:30:00"),
    ("Labor Report", "labor", "2023-04-07 08:30:00"),
    ("Labor Report", "labor", "2023-05-05 08:30:00"),
    ("Labor Report", "labor", "2023-06-02 08:30:00"),
    ("Labor Report", "labor", "2023-07-07 08:30:00"),
    ("Labor Report", "labor", "2023-08-04 08:30:00"),
    ("Labor Report", "labor", "2023-09-01 08:30:00"),
    ("Labor Report", "labor", "2023-10-06 08:30:00"),
    ("Labor Report", "labor", "2023-11-03 08:30:00"),
    ("Labor Report", "labor", "2023-12-08 08:30:00"),
    # 2024
    ("Labor Report", "labor", "2024-01-05 08:30:00"),
    ("Labor Report", "labor", "2024-02-02 08:30:00"),
    ("Labor Report", "labor", "2024-03-08 08:30:00"),
    ("Labor Report", "labor", "2024-04-05 08:30:00"),
    ("Labor Report", "labor", "2024-05-03 08:30:00"),
    ("Labor Report", "labor", "2024-06-07 08:30:00"),
    ("Labor Report", "labor", "2024-07-05 08:30:00"),
    ("Labor Report", "labor", "2024-08-02 08:30:00"),
    ("Labor Report", "labor", "2024-09-06 08:30:00"),
    ("Labor Report", "labor", "2024-10-04 08:30:00"),
    ("Labor Report", "labor", "2024-11-01 08:30:00"),
    ("Labor Report", "labor", "2024-12-06 08:30:00"),
    # 2025
    ("Labor Report", "labor", "2025-01-10 08:30:00"),
    ("Labor Report", "labor", "2025-02-07 08:30:00"),
    ("Labor Report", "labor", "2025-03-07 08:30:00"),
    ("Labor Report", "labor", "2025-04-04 08:30:00"),
    ("Labor Report", "labor", "2025-05-02 08:30:00"),
    ("Labor Report", "labor", "2025-06-06 08:30:00"),
    ("Labor Report", "labor", "2025-07-03 08:30:00"),
    ("Labor Report", "labor", "2025-08-01 08:30:00"),
    ("Labor Report", "labor", "2025-09-05 08:30:00"),
    ("Labor Report", "labor", "2025-10-03 08:30:00"),
    ("Labor Report", "labor", "2025-11-07 08:30:00"),
    ("Labor Report", "labor", "2025-12-05 08:30:00"),
]

RETAIL_SALES_RELEASES = [
    # 2021
    ("Retail Sales", "consumption", "2021-01-15 08:30:00"),
    ("Retail Sales", "consumption", "2021-02-17 08:30:00"),
    ("Retail Sales", "consumption", "2021-03-17 08:30:00"),
    ("Retail Sales", "consumption", "2021-04-15 08:30:00"),
    ("Retail Sales", "consumption", "2021-05-14 08:30:00"),
    ("Retail Sales", "consumption", "2021-06-15 08:30:00"),
    ("Retail Sales", "consumption", "2021-07-16 08:30:00"),
    ("Retail Sales", "consumption", "2021-08-17 08:30:00"),
    ("Retail Sales", "consumption", "2021-09-16 08:30:00"),
    ("Retail Sales", "consumption", "2021-10-15 08:30:00"),
    ("Retail Sales", "consumption", "2021-11-16 08:30:00"),
    ("Retail Sales", "consumption", "2021-12-16 08:30:00"),
    # 2022
    ("Retail Sales", "consumption", "2022-01-14 08:30:00"),
    ("Retail Sales", "consumption", "2022-02-16 08:30:00"),
    ("Retail Sales", "consumption", "2022-03-16 08:30:00"),
    ("Retail Sales", "consumption", "2022-04-14 08:30:00"),
    ("Retail Sales", "consumption", "2022-05-17 08:30:00"),
    ("Retail Sales", "consumption", "2022-06-15 08:30:00"),
    ("Retail Sales", "consumption", "2022-07-15 08:30:00"),
    ("Retail Sales", "consumption", "2022-08-17 08:30:00"),
    ("Retail Sales", "consumption", "2022-09-15 08:30:00"),
    ("Retail Sales", "consumption", "2022-10-14 08:30:00"),
    ("Retail Sales", "consumption", "2022-11-16 08:30:00"),
    ("Retail Sales", "consumption", "2022-12-15 08:30:00"),
    # 2023
    ("Retail Sales", "consumption", "2023-01-18 08:30:00"),
    ("Retail Sales", "consumption", "2023-02-15 08:30:00"),
    ("Retail Sales", "consumption", "2023-03-15 08:30:00"),
    ("Retail Sales", "consumption", "2023-04-14 08:30:00"),
    ("Retail Sales", "consumption", "2023-05-16 08:30:00"),
    ("Retail Sales", "consumption", "2023-06-15 08:30:00"),
    ("Retail Sales", "consumption", "2023-07-18 08:30:00"),
    ("Retail Sales", "consumption", "2023-08-15 08:30:00"),
    ("Retail Sales", "consumption", "2023-09-15 08:30:00"),
    ("Retail Sales", "consumption", "2023-10-17 08:30:00"),
    ("Retail Sales", "consumption", "2023-11-15 08:30:00"),
    ("Retail Sales", "consumption", "2023-12-15 08:30:00"),
    # 2024
    ("Retail Sales", "consumption", "2024-01-17 08:30:00"),
    ("Retail Sales", "consumption", "2024-02-15 08:30:00"),
    ("Retail Sales", "consumption", "2024-03-15 08:30:00"),
    ("Retail Sales", "consumption", "2024-04-15 08:30:00"),
    ("Retail Sales", "consumption", "2024-05-15 08:30:00"),
    ("Retail Sales", "consumption", "2024-06-18 08:30:00"),
    ("Retail Sales", "consumption", "2024-07-16 08:30:00"),
    ("Retail Sales", "consumption", "2024-08-15 08:30:00"),
    ("Retail Sales", "consumption", "2024-09-17 08:30:00"),
    ("Retail Sales", "consumption", "2024-10-17 08:30:00"),
    ("Retail Sales", "consumption", "2024-11-15 08:30:00"),
    ("Retail Sales", "consumption", "2024-12-17 08:30:00"),
    # 2025
    ("Retail Sales", "consumption", "2025-01-16 08:30:00"),
    ("Retail Sales", "consumption", "2025-02-14 08:30:00"),
    ("Retail Sales", "consumption", "2025-03-17 08:30:00"),
    ("Retail Sales", "consumption", "2025-04-16 08:30:00"),
    ("Retail Sales", "consumption", "2025-05-15 08:30:00"),
    ("Retail Sales", "consumption", "2025-06-17 08:30:00"),
    ("Retail Sales", "consumption", "2025-07-17 08:30:00"),
    ("Retail Sales", "consumption", "2025-08-15 08:30:00"),
    ("Retail Sales", "consumption", "2025-09-16 08:30:00"),
    ("Retail Sales", "consumption", "2025-10-16 08:30:00"),
    ("Retail Sales", "consumption", "2025-11-14 08:30:00"),
    ("Retail Sales", "consumption", "2025-12-16 08:30:00"),
]

# ---------------------------------------------------------------------------
# Combine, sort, assign IDs, write
# ---------------------------------------------------------------------------

def generate_events(output_path: Path) -> None:
    all_events = (
        CPI_RELEASES
        + FOMC_RELEASES
        + GDP_RELEASES
        + LABOR_RELEASES
        + RETAIL_SALES_RELEASES
    )

    # Sort by timestamp
    all_events.sort(key=lambda x: x[2])

    # Assign sequential event_ids
    rows = [
        {
            "event_id"       : i + 1,
            "timestamp"      : ts,
            "event_name"     : name,
            "event_category" : category,
            "event_flag"     : 1,
        }
        for i, (name, category, ts) in enumerate(all_events)
    ]

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["event_id", "timestamp", "event_name", "event_category", "event_flag"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\nWrote {len(rows)} events to: {output_path}\n")
    counts = {}
    for name, _, _ in all_events:
        counts[name] = counts.get(name, 0) + 1
    for name, count in sorted(counts.items()):
        print(f"  {name:<22} {count:>3} events")
    print(f"  {'TOTAL':<22} {len(rows):>3} events")
    print(f"\nDate range: {rows[0]['timestamp'][:10]} → {rows[-1]['timestamp'][:10]}")


if __name__ == "__main__":
    generate_events(OUTPUT_PATH)

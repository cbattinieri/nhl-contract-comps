"""
NHL Aging Model — Data Pipeline
Builds a long-format player-season dataset for the Bayesian aging model.
Pulls from the same NHL API as generate_data.py but is fully independent.

Output: aging_data.parquet — one row per player-season, with age and
primary production metrics (PPG for forwards, TOI/game for defensemen).
"""

import sys
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# 2007-08 onward — enough history for reliable aging curves
# Natural Stat Trick advanced stats also start here if needed later
START_YEAR = 2007
END_YEAR = 2024

ALL_SEASONS = [
    f"{y}{y+1}"
    for y in range(START_YEAR, END_YEAR + 1)
    if f"{y}{y+1}" != "20042005"
]

NHL_STATS_URL = (
    "https://api.nhle.com/stats/rest/en/skater/summary"
    "?sort=points&limit=-1&cayenneExp=seasonId="
)
NHL_BIOS_URL = (
    "https://api.nhle.com/stats/rest/en/skater/bios"
    "?limit=-1&cayenneExp=seasonId="
)

# Minimum games per season — removes noisy short stints
# and partial seasons that distort production rates
MIN_GP_PER_SEASON = 20

OUTPUT_DIR = Path(__file__).parent
OUTPUT_PATH = OUTPUT_DIR / "aging_data.parquet"

# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_nhl_data(base_url: str, seasons: list[str]) -> pd.DataFrame:
    """Fetch skater data from NHL API across multiple seasons."""
    all_rows = []
    for season in seasons:
        url = f"{base_url}{season}%20and%20gameTypeId=2"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            all_rows.extend(rows)
        except Exception as e:
            print(f"  Warning: season {season} — {e}", file=sys.stderr)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# FEATURE BUILDING
# ---------------------------------------------------------------------------

def build_aging_dataset(
    raw_stats: pd.DataFrame,
    raw_bios: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge stats and bios, compute age, and return a clean long-format
    dataset ready for the Bayesian aging model.
    """
    # --- Age calculation ---
    # Use Oct 1 of the season start year as the reference date.
    # This approximates how old a player is at the start of the season,
    # which is the most natural anchor for production aging curves.
    bios_slim = raw_bios[["playerId", "birthDate"]].drop_duplicates("playerId").copy()
    bios_slim["birthDate"] = pd.to_datetime(bios_slim["birthDate"], errors="coerce")

    df = raw_stats.merge(bios_slim, on="playerId", how="left")
    df["season_start"] = pd.to_datetime(
        df["seasonId"].astype(str).str[:4] + "-10-01"
    )
    df["age"] = (
        (df["season_start"] - df["birthDate"]).dt.days / 365.25
    ).round(2)

    # --- Filter ---
    df = df[df["positionCode"] != "G"].copy()          # exclude goalies
    df = df[df["gamesPlayed"] >= MIN_GP_PER_SEASON]    # minimum games threshold
    df = df[df["age"].between(18, 42)]                 # exclude extreme outliers

    # --- Position grouping ---
    df["position_group"] = np.where(
        df["positionCode"] == "D", "Defense", "Forward"
    )

    # --- Primary production metrics ---
    # Forwards: pointsPerGame is the primary aging signal
    # Defensemen: toi_per_game is the primary aging signal
    # TOI from NHL API is in seconds — convert to minutes
    df["toi_per_game"] = (df["timeOnIcePerGame"] / 60).round(3)
    df["ppg"] = df["pointsPerGame"].round(4)
    df["ev_ppg"] = (
        df["evPoints"] / df["gamesPlayed"]
    ).fillna(0).round(4)

    # --- Keep only what the aging model needs ---
    aging = df[[
        "playerId",
        "skaterFullName",
        "seasonId",
        "position_group",
        "positionCode",
        "age",
        "gamesPlayed",
        "ppg",
        "ev_ppg",
        "toi_per_game",
    ]].rename(columns={"skaterFullName": "name"}).copy()

    # Sort for readability
    aging = aging.sort_values(["playerId", "seasonId"]).reset_index(drop=True)

    return aging


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_and_plot(aging_data: pd.DataFrame) -> None:
    """
    Plot population-level production by age for each position.
    Expected:
      - Forwards: PPG peaks mid-20s, declines after
      - Defense: TOI/game peaks mid-to-late 20s, flatter decline
    If either curve looks obviously wrong, there is a data issue to fix
    before building the Bayesian model on top.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        (axes[0], "Forward",  "ppg",          "Points / Game"),
        (axes[1], "Defense",  "toi_per_game",  "TOI / Game (min)"),
    ]

    for ax, pos, metric, label in configs:
        sub = aging_data[aging_data["position_group"] == pos].copy()
        sub["age_bin"] = sub["age"].round()
        agg = sub.groupby("age_bin")[metric].agg(["mean", "median", "count"])
        agg = agg[agg["count"] >= 30]  # only show ages with at least 30 observations

        ax.plot(agg.index, agg["mean"],   label="Mean",   marker="o", linewidth=2)
        ax.plot(agg.index, agg["median"], label="Median", marker="s", linewidth=1.5,
                linestyle="--", alpha=0.8)
        ax.fill_between(agg.index, agg["mean"] * 0.9, agg["mean"] * 1.1,
                        alpha=0.1, label="±10% band")
        ax.set_title(f"{pos} — {label} by Age", fontsize=12)
        ax.set_xlabel("Age")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Population Aging Curves — Sanity Check", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aging_curves_sanity_check.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Plot saved to aging_curves_sanity_check.png")


def print_summary(aging_data: pd.DataFrame) -> None:
    """Print key dataset statistics."""
    print("\n--- Dataset Summary ---")
    print(f"  Total player-seasons : {len(aging_data):,}")
    print(f"  Unique players       : {aging_data['playerId'].nunique():,}")
    print(f"  Season range         : {aging_data['seasonId'].min()} – {aging_data['seasonId'].max()}")
    print(f"  Age range            : {aging_data['age'].min():.1f} – {aging_data['age'].max():.1f}")
    print()

    for pos in ["Forward", "Defense"]:
        sub = aging_data[aging_data["position_group"] == pos]
        metric = "ppg" if pos == "Forward" else "toi_per_game"
        label  = "PPG"  if pos == "Forward" else "TOI/gm"
        print(f"  {pos}:")
        print(f"    Player-seasons : {len(sub):,}")
        print(f"    Unique players : {sub['playerId'].nunique():,}")
        print(f"    Avg {label:<8}  : {sub[metric].mean():.3f}")
        seasons_per = sub.groupby("playerId")["seasonId"].count()
        print(f"    Avg seasons/player: {seasons_per.mean():.1f} "
              f"(min {seasons_per.min()}, max {seasons_per.max()})")
        print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"Pulling {len(ALL_SEASONS)} seasons "
          f"({ALL_SEASONS[0]}–{ALL_SEASONS[-1]})...")

    raw_stats = fetch_nhl_data(NHL_STATS_URL, ALL_SEASONS)
    print(f"  {len(raw_stats):,} stat rows")

    raw_bios = fetch_nhl_data(NHL_BIOS_URL, ALL_SEASONS)
    raw_bios = raw_bios.drop_duplicates(subset="playerId")
    print(f"  {len(raw_bios):,} bio rows")

    print("\nBuilding aging dataset...")
    aging_data = build_aging_dataset(raw_stats, raw_bios)

    print_summary(aging_data)

    print("Running sanity check plots...")
    validate_and_plot(aging_data)

    print(f"Saving to {OUTPUT_PATH}...")
    aging_data.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
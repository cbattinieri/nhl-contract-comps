"""
NHL Contract Comparables - Data Pipeline
Fetches stats + contracts, runs KNN models, outputs JSON for static site.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PUCKPEDIA_API_KEY = os.environ.get("PUCKPEDIA_API_KEY", "")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dynamic season calculation
NOW = datetime.now()
# NHL season spans two calendar years: a season starting in Oct 2024 is "20242025"
CURRENT_YEAR = NOW.year if NOW.month >= 7 else NOW.year - 1
CURRENT_SEASON = int(f"{CURRENT_YEAR}{CURRENT_YEAR + 1}")

# Build season list dynamically from 2003-04 onward (skip 04-05 lockout)
ALL_SEASONS = []
for y in range(2003, CURRENT_YEAR + 1):
    sid = f"{y}{y + 1}"
    if sid == "20042005":  # lockout
        continue
    ALL_SEASONS.append(sid)

# Cap upper limits by season
# ──────────────────────────────────────────────────────────────────────
# UPDATE THIS EACH SEASON: add the new confirmed cap, update projections.
# Use absolute season IDs so historical data stays correct forever.
# ──────────────────────────────────────────────────────────────────────
CAP_LIMITS = {
    20032004: 39_000_000,
    20052006: 39_000_000,
    20062007: 44_000_000,
    20072008: 50_300_000,
    20082009: 56_700_000,
    20092010: 56_800_000,
    20102011: 59_400_000,
    20112012: 64_300_000,
    20122013: 70_200_000,
    20132014: 64_300_000,
    20142015: 69_000_000,
    20152016: 71_400_000,
    20162017: 73_000_000,
    20172018: 75_000_000,
    20182019: 79_500_000,
    20192020: 81_500_000,
    20202021: 81_500_000,
    20212022: 81_500_000,
    20222023: 82_500_000,
    20232024: 83_500_000,
    20242025: 88_000_000,
    20252026: 95_500_000,   # confirmed
    20262027: 104_000_000,  # agreed, may rise to ~$107M
    20272028: 113_500_000,  # agreed, subject to minor adjustment
}

KNN_NEIGHBORS = 5
LOOKBACK_SEASONS = 3  # how many prior FA classes to use as training data

# Estimation uses inverse-distance weighting (IDW) instead of arbitrary
# fixed weights. Closer comps get exponentially more influence.
# A small epsilon prevents division-by-zero for exact matches.
IDW_EPSILON = 1e-6

# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------
NHL_STATS_URL = "https://api.nhle.com/stats/rest/en/skater/summary?sort=points&limit=-1&cayenneExp=seasonId="
NHL_BIOS_URL = "https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&cayenneExp=seasonId="


def fetch_nhl_data(base_url: str, seasons: list[str]) -> pd.DataFrame:
    """Fetch data from NHL API across multiple seasons."""
    all_rows = []
    for season in seasons:
        url = f"{base_url}{season}%20and%20gameTypeId=2"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            all_rows.extend(data)
        except Exception as e:
            print(f"  Warning: season {season} – {e}", file=sys.stderr)
    return pd.DataFrame(all_rows)


def fetch_contracts() -> pd.DataFrame:
    """Fetch contract data from PuckPedia."""
    if not PUCKPEDIA_API_KEY:
        raise RuntimeError("PUCKPEDIA_API_KEY env var is required")

    resp = requests.get(
        f"https://puckpedia.com/api/players2?api_key={PUCKPEDIA_API_KEY}",
        timeout=60,
    )
    resp.raise_for_status()
    records = resp.json().get("data", [])
    df = pd.DataFrame(records)

    # Explode current contracts
    df_current = df.explode("current")
    df_current = pd.concat(
        [df_current.drop(columns=["current"]),
         df_current["current"].apply(pd.Series)],
        axis=1,
    )
    df_current = df_current.drop(columns=["future"], errors="ignore")

    # Keep only rows with a valid contract
    df_current = df_current[df_current["contract_id"].notna()].copy()
    return df_current


# ---------------------------------------------------------------------------
# DATA WRANGLING
# ---------------------------------------------------------------------------

def build_stats(raw_stats: pd.DataFrame) -> pd.DataFrame:
    """Clean stats and compute career-to-date features."""
    df = raw_stats.copy()

    # July 1 reference date for age calc
    # For season 20252026, July 1 age = age on 2026-07-01 (when FA opens)
    df["july_1"] = pd.to_datetime(
        df["seasonId"].astype(str).str[4:8] + "-07-01"
    )

    # Sort for cumulative calculations
    df = df.sort_values(["playerId", "seasonId"]).reset_index(drop=True)

    # Season number per player
    df["szn_no"] = df.groupby("playerId").cumcount() + 1

    # Career-to-date cumulative stats
    for col in ["gamesPlayed", "points", "evPoints"]:
        df[f"ctd_{col}"] = df.groupby("playerId")[col].cumsum()

    # Career-to-date averages (excluding current season)
    df["ctd_p_pg"] = (
        (df["ctd_points"] - df["points"]) / (df["ctd_gamesPlayed"] - df["gamesPlayed"])
    ).fillna(0).round(4)

    df["ctd_ev_p_pg"] = (
        (df["ctd_evPoints"] - df["evPoints"]) / (df["ctd_gamesPlayed"] - df["gamesPlayed"])
    ).fillna(0).round(4)

    df["ctd_toi_avg"] = (
        (df.groupby("playerId")["timeOnIcePerGame"].cumsum() - df["timeOnIcePerGame"])
        / (df["szn_no"] - 1)
    ).fillna(0).round(2)

    df["pct_gp"] = (df["ctd_gamesPlayed"] / (df["szn_no"] * 82)).round(4)

    # Even-strength per game
    df["evPointsPerGame"] = (df["evPoints"] / df["gamesPlayed"]).fillna(0).round(4)

    # L2/L3 rolling averages
    rolling_cols = ["gamesPlayed", "points", "pointsPerGame",
                    "evPoints", "evPointsPerGame", "timeOnIcePerGame"]
    for col in rolling_cols:
        shifted_1 = df.groupby("playerId")[col].shift(1).fillna(0)
        shifted_2 = df.groupby("playerId")[col].shift(2).fillna(0)
        df[f"{col}_L2"] = ((df[col] + shifted_1) / 2).round(4)
        df[f"{col}_L3"] = ((df[col] + shifted_1 + shifted_2) / 3).round(4)

    return df


def build_contracts(raw_contracts: pd.DataFrame) -> pd.DataFrame:
    """Derive contract_year and signing_status for each contract."""
    df = raw_contracts.copy()

    # Derive the season the contract started
    # contract_end "2025-2026" with length 3 → started 2022 → season ID 20222023
    df["contract_end_year"] = df["contract_end"].astype(str).str.split("-").str[0].astype(int)
    df["contract_start_year"] = df["contract_end_year"] - df["length"].astype(int)
    df["contract_year"] = (
        df["contract_start_year"] * 10000
        + df["contract_start_year"]
        + 1
    ).astype(int)

    # For current-season expiring FAs, use expiry_status as signing_status
    # In the 2025-2026 season, current FAs have contract_end == "2025-2026"
    current_mask = df["contract_end"] == f"{CURRENT_YEAR}-{CURRENT_YEAR + 1}"
    # Fallback: also check computed contract_year
    current_mask_2 = df["contract_year"] == CURRENT_SEASON
    is_current = current_mask | current_mask_2
    df.loc[is_current, "signing_status"] = df.loc[is_current, "expiry_status"]

    # Override contract_year for current expiring contracts
    df.loc[is_current, "contract_year"] = CURRENT_SEASON

    df["nhl_id"] = df["nhl_id"].astype(int)

    # Drop ELC slides (signing_status == False)
    df = df[df["signing_status"] != False]  # noqa: E712

    return df


def merge_stats_contracts(stats: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """Merge on player ID + season, filter to skaters only."""
    merged = pd.merge(
        contracts, stats,
        how="inner",
        left_on=["nhl_id", "contract_year"],
        right_on=["playerId", "seasonId"],
    )
    # Exclude goalies
    merged = merged[merged["positionCode"] != "G"].copy()

    # Normalize position
    merged["position_group"] = np.where(
        merged["positionCode"] == "D", "Defense", "Forward"
    )
    merged["position_display"] = np.where(
        merged["positionCode"].isin(["L", "R"]), "Winger",
        np.where(merged["positionCode"] == "C", "Center", "Defense")
    )

    return merged


# ---------------------------------------------------------------------------
# KNN MODEL
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FEATURE CONFIGURATION (validated via feature_study.ipynb)
# ---------------------------------------------------------------------------
# Career-to-date base features — used by all 4 models
BASE_FEATURES = [
    "july_1_age", "ctd_gamesPlayed", "ctd_p_pg", "ctd_ev_p_pg",
    "pct_gp", "ctd_toi_avg",
]

# L3 rolling averages — added to all models (biggest MAE improvement)
L3_FEATURES = [
    "gamesPlayed_L3", "points_L3", "pointsPerGame_L3",
    "evPoints_L3", "evPointsPerGame_L3", "timeOnIcePerGame_L3",
]

# RFA models: CTD + all L3, NO weights
# Ablation result: MAE drops from ~1.91% to ~1.33% (fwd), ~2.01% to ~1.83% (def)
RFA_FEATURES = BASE_FEATURES + L3_FEATURES
RFA_WEIGHTS = None

# UFA models: CTD + all L3, light L3 weight (2x)
# Ablation result: MAE drops from ~1.99% to ~1.54% (fwd), ~2.17% to ~1.78% (def)
UFA_FEATURES = BASE_FEATURES + L3_FEATURES
UFA_WEIGHTS = {
    "gamesPlayed_L3": 2, "points_L3": 2, "pointsPerGame_L3": 2,
    "evPoints_L3": 2, "evPointsPerGame_L3": 2, "timeOnIcePerGame_L3": 2,
}


def run_knn_model(
    data: pd.DataFrame,
    features: list[str],
    feature_weights: dict | None = None,
) -> dict:
    """
    Fit KNN on historical FAs, predict comps for current-season FAs.
    Returns {pending_player_id: [comp_player_id, ...]}
    """
    min_season = CURRENT_SEASON - (LOOKBACK_SEASONS * 10001)

    pending = data[data["contract_year"] == CURRENT_SEASON].copy()
    historical = data[
        (data["contract_year"] != CURRENT_SEASON)
        & (data["contract_year"] >= min_season)
    ].copy()

    if pending.empty or historical.empty:
        return {}

    scaler = StandardScaler()
    X_hist = pd.DataFrame(
        scaler.fit_transform(historical[features]),
        columns=features,
        index=historical.index,
    )
    X_pend = pd.DataFrame(
        scaler.transform(pending[features]),
        columns=features,
        index=pending.index,
    )

    # Apply feature weights if provided
    if feature_weights:
        for feat, w in feature_weights.items():
            if feat in X_hist.columns:
                X_hist[feat] *= w
                X_pend[feat] *= w

    knn = NearestNeighbors(n_neighbors=min(KNN_NEIGHBORS, len(historical)), algorithm="auto")
    knn.fit(X_hist)
    distances, indices = knn.kneighbors(X_pend)

    results = {}
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        pid = int(pending.iloc[i]["playerId"])
        comp_ids = [int(historical.iloc[idx]["playerId"]) for idx in idxs]
        comp_dists = [round(float(d), 4) for d in dists]
        results[pid] = {"comps": comp_ids, "distances": comp_dists}

    return results


# ---------------------------------------------------------------------------
# ESTIMATION — INVERSE DISTANCE WEIGHTING
# ---------------------------------------------------------------------------

def idw_estimate(cap_hit_pcts: list[float], distances: list[float]) -> dict:
    """
    Compute a cap-hit-% estimate using inverse-distance weighting (IDW).

    Instead of arbitrary fixed weights (30/25/20/15/10), each comp's
    influence is proportional to 1/distance.  Closer comps dominate;
    distant comps contribute less.  This is the standard approach in
    spatial statistics and KNN regression.

    Returns dict with:
      estimate   – IDW weighted mean cap hit %
      weights    – the normalised weight each comp received
      ci_low     – lower bound of a simple ±1σ interval
      ci_high    – upper bound
      std        – weighted standard deviation (measure of uncertainty)
    """
    if not cap_hit_pcts or not distances:
        return {"estimate": 0, "weights": [], "ci_low": 0, "ci_high": 0, "std": 0}

    n = len(cap_hit_pcts)
    vals = np.array(cap_hit_pcts, dtype=float)
    dists = np.array(distances, dtype=float)

    # Inverse distance weights (with epsilon to avoid div/0)
    raw_weights = 1.0 / (dists + IDW_EPSILON)
    norm_weights = raw_weights / raw_weights.sum()

    # Weighted mean
    estimate = float(np.dot(norm_weights, vals))

    # Weighted standard deviation (Bessel-corrected for small n)
    # Formula: sqrt( sum(w_i * (x_i - mu)^2) / (1 - sum(w_i^2)) )
    variance_numer = np.dot(norm_weights, (vals - estimate) ** 2)
    variance_denom = 1.0 - np.dot(norm_weights, norm_weights)  # effective sample correction
    if variance_denom > 0:
        w_std = float(np.sqrt(variance_numer / variance_denom))
    else:
        w_std = float(np.std(vals))

    return {
        "estimate": round(estimate, 2),
        "weights": [round(float(w), 4) for w in norm_weights],
        "ci_low": round(max(0, estimate - w_std), 2),
        "ci_high": round(estimate + w_std, 2),
        "std": round(w_std, 2),
    }


def backtest_model(
    merged: pd.DataFrame,
    fa_status: str,
    pos_group: str,
    features: list[str],
    feature_weights: dict | None,
    cap_limits: dict,
) -> dict:
    """
    Leave-one-season-out backtest.

    For each historical FA class, hold it out as "pending", train on the
    remaining classes, predict cap hit %, and compare to the actual.
    Returns MAE, median absolute error, and per-player errors.
    """
    subset = merged[
        (merged["signing_status"] == fa_status)
        & (merged["position_group"] == pos_group)
    ].copy()

    # Only backtest on seasons that have actual contract data
    seasons_with_data = sorted(
        subset[subset["contract_year"] != CURRENT_SEASON]["contract_year"].unique()
    )

    if len(seasons_with_data) < 2:
        return {"mae": None, "median_ae": None, "n": 0, "errors": []}

    all_errors = []
    for hold_out_season in seasons_with_data:
        pending = subset[subset["contract_year"] == hold_out_season].copy()
        min_train = hold_out_season - (LOOKBACK_SEASONS * 10001)
        historical = subset[
            (subset["contract_year"] != hold_out_season)
            & (subset["contract_year"] >= min_train)
            & (subset["contract_year"] < hold_out_season)
        ].copy()

        if pending.empty or len(historical) < KNN_NEIGHBORS:
            continue

        available_features = [f for f in features if f in pending.columns]
        scaler = StandardScaler()
        X_hist = pd.DataFrame(
            scaler.fit_transform(historical[available_features]),
            columns=available_features, index=historical.index,
        )
        X_pend = pd.DataFrame(
            scaler.transform(pending[available_features]),
            columns=available_features, index=pending.index,
        )
        if feature_weights:
            for feat, w in feature_weights.items():
                if feat in X_hist.columns:
                    X_hist[feat] *= w
                    X_pend[feat] *= w

        k = min(KNN_NEIGHBORS, len(historical))
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        knn.fit(X_hist)
        distances, indices = knn.kneighbors(X_pend)

        for i in range(len(pending)):
            comp_pcts = []
            for idx in indices[i]:
                comp_row = historical.iloc[idx]
                cy = int(comp_row["contract_year"])
                ul = cap_limits.get(cy, 95_500_000)
                length = int(comp_row.get("length", 1)) if pd.notna(comp_row.get("length")) else 1
                value = int(comp_row.get("value", 0)) if pd.notna(comp_row.get("value")) else 0
                aav = value / length if length > 0 else 0
                comp_pcts.append((aav / ul) * 100 if ul > 0 else 0)

            est = idw_estimate(comp_pcts, list(distances[i]))

            # Actual cap hit % for this pending player
            prow = pending.iloc[i]
            actual_cy = int(prow["contract_year"])
            actual_ul = cap_limits.get(actual_cy, 95_500_000)
            actual_length = int(prow.get("length", 1)) if pd.notna(prow.get("length")) else 1
            actual_value = int(prow.get("value", 0)) if pd.notna(prow.get("value")) else 0
            actual_aav = actual_value / actual_length if actual_length > 0 else 0
            actual_pct = (actual_aav / actual_ul) * 100 if actual_ul > 0 else 0

            error = est["estimate"] - actual_pct
            all_errors.append({
                "playerId": int(prow["playerId"]),
                "season": actual_cy,
                "predicted": est["estimate"],
                "actual": round(actual_pct, 2),
                "error": round(error, 2),
                "abs_error": round(abs(error), 2),
            })

    if not all_errors:
        return {"mae": None, "median_ae": None, "n": 0, "errors": []}

    abs_errors = [e["abs_error"] for e in all_errors]
    return {
        "mae": round(float(np.mean(abs_errors)), 2),
        "median_ae": round(float(np.median(abs_errors)), 2),
        "n": len(all_errors),
        "errors": all_errors,
    }



# ---------------------------------------------------------------------------
# TERM SENSITIVITY
# ---------------------------------------------------------------------------

def build_term_curves(merged: pd.DataFrame, cap_limits: dict) -> dict:
    """Build term-adjustment model from historical data."""
    df = merged.copy()
    df["_ul"] = df["contract_year"].map(cap_limits).fillna(95_500_000)
    df["_length"] = df["length"].fillna(1).astype(int).clip(lower=1)
    df["_value"] = df["value"].fillna(0).astype(float)
    df["_aav"] = df["_value"] / df["_length"]
    df["_cap_pct"] = (df["_aav"] / df["_ul"]) * 100

    df = df[
        (df["contract_year"] != CURRENT_SEASON)
        & (df["_length"] >= 2)
        & (df["_cap_pct"] > 0)
    ].copy()

    curves = {}

    for fa in ["RFA", "UFA"]:
        for pos in ["Forward", "Defense"]:
            seg = df[
                (df["signing_status"] == fa)
                & (df["position_group"] == pos)
            ].copy()

            if len(seg) < 10:
                curves[f"{fa}_{pos}"] = None
                continue

            tier_edges = [0, 2.5, 5.0, 8.0, 100.0]
            seg["_tier"] = pd.cut(seg["_cap_pct"], bins=tier_edges, labels=False)

            tier_coefficients = {}
            for tier in seg["_tier"].dropna().unique():
                tier_data = seg[seg["_tier"] == tier]
                if len(tier_data) < 5:
                    continue

                terms = tier_data["_length"].values.astype(float)
                pcts = tier_data["_cap_pct"].values

                if terms.std() == 0:
                    continue
                b = np.cov(terms, pcts)[0, 1] / np.var(terms)
                a = pcts.mean() - b * terms.mean()

                tier_coefficients[int(tier)] = {
                    "intercept": round(float(a), 4),
                    "slope": round(float(b), 4),
                    "n": len(tier_data),
                    "mean_term": round(float(terms.mean()), 1),
                    "mean_pct": round(float(pcts.mean()), 2),
                }

            all_terms = seg["_length"].values.astype(float)
            all_pcts = seg["_cap_pct"].values
            if all_terms.std() > 0:
                b_all = np.cov(all_terms, all_pcts)[0, 1] / np.var(all_terms)
                a_all = all_pcts.mean() - b_all * all_terms.mean()
            else:
                a_all, b_all = float(all_pcts.mean()), 0.0

            curves[f"{fa}_{pos}"] = {
                "tiers": tier_coefficients,
                "fallback": {
                    "intercept": round(float(a_all), 4),
                    "slope": round(float(b_all), 4),
                    "n": len(seg),
                },
                "tier_edges": tier_edges,
            }

    return curves


def compute_term_table(
    base_cap_pct: float,
    fa_status: str,
    pos_group: str,
    curves: dict,
    current_cap: int,
) -> list[dict]:
    """Compute adjusted AAV at each term length."""
    key = f"{fa_status}_{pos_group}"
    curve = curves.get(key)

    if not curve:
        return [
            {"term": t, "capHitPct": base_cap_pct,
             "aav": round(base_cap_pct * current_cap / 100)}
            for t in range(2, 9)
        ]

    tier_edges = curve["tier_edges"]
    tier_idx = None
    for i in range(len(tier_edges) - 1):
        if tier_edges[i] <= base_cap_pct < tier_edges[i + 1]:
            tier_idx = i
            break

    tier_data = curve["tiers"].get(tier_idx) if tier_idx is not None else None
    if tier_data and tier_data["n"] >= 5:
        intercept = tier_data["intercept"]
        slope = tier_data["slope"]
    else:
        intercept = curve["fallback"]["intercept"]
        slope = curve["fallback"]["slope"]

    if slope != 0:
        implied_term = (base_cap_pct - intercept) / slope
    else:
        implied_term = 4.0

    table = []
    for t in range(2, 9):
        adjusted_pct = base_cap_pct + slope * (t - implied_term)
        adjusted_pct = max(base_cap_pct * 0.5, min(base_cap_pct * 1.5, adjusted_pct))
        adjusted_pct = round(max(0, adjusted_pct), 2)
        aav = round(adjusted_pct * current_cap / 100)
        table.append({"term": t, "capHitPct": adjusted_pct, "aav": aav})

    return table


# ---------------------------------------------------------------------------
# BUILD OUTPUT
# ---------------------------------------------------------------------------

def compute_age(stats: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """Add july_1_age and draft position data to stats using bios."""
    bios_slim = bios[["playerId", "birthDate"]].drop_duplicates(subset="playerId")
    bios_slim["birthDate"] = pd.to_datetime(bios_slim["birthDate"], errors="coerce")

    # Draft info — NHL bios API includes draftYear, draftOverall, draftRound
    draft_cols = ["playerId"]
    for col in ["draftYear", "draftOverall", "draftRound"]:
        if col in bios.columns:
            draft_cols.append(col)
    draft_slim = bios[draft_cols].drop_duplicates(subset="playerId")

    merged = stats.merge(bios_slim, on="playerId", how="left")
    merged["july_1_age"] = (
        (merged["july_1"] - merged["birthDate"]).dt.days / 365.25
    ).fillna(0).astype(int)

    # Merge draft info
    if "draftOverall" in draft_slim.columns:
        merged = merged.merge(draft_slim, on="playerId", how="left")
        # Fill undrafted players with 999 (clearly out of range)
        merged["draftOverall"] = merged["draftOverall"].fillna(999).astype(int)
    else:
        merged["draftOverall"] = 999

    # Create draft bucket feature (non-linear grouping)
    # Top-5, Top-15, 1st round, 2nd round, 3rd-7th round, undrafted
    conditions = [
        merged["draftOverall"] <= 5,
        merged["draftOverall"] <= 15,
        merged["draftOverall"] <= 31,
        merged["draftOverall"] <= 62,
        merged["draftOverall"] <= 224,
    ]
    bucket_values = [1, 2, 3, 4, 5]  # 1=elite, 5=late round
    merged["draftBucket"] = np.select(conditions, bucket_values, default=6)  # 6=undrafted

    return merged


def build_player_record(row, cap_limits: dict) -> dict:
    """Build a clean player record dict for JSON output."""
    contract_year = int(row.get("contract_year", 0))
    upper_limit = cap_limits.get(contract_year, 95_500_000)
    length = int(row.get("length", 1)) if pd.notna(row.get("length")) else 1
    value = int(row.get("value", 0)) if pd.notna(row.get("value")) else 0
    aav = round(value / length) if length > 0 else 0
    cap_hit_pct = round((aav / upper_limit) * 100, 2) if upper_limit > 0 else 0

    return {
        "playerId": int(row["playerId"]),
        "name": str(row.get("skaterFullName", "")),
        "position": str(row.get("position_display", "")),
        "positionGroup": str(row.get("position_group", "")),
        "signingStatus": str(row.get("signing_status", "")),
        "contractYear": contract_year,
        "age": int(row.get("july_1_age", 0)),
        "height": str(row.get("height", "")),
        "weight": str(row.get("weight", "")),
        "shoots": str(row.get("shootsCatches", "")),
        # Draft info
        "draftOverall": int(row.get("draftOverall", 999)),
        "draftBucket": int(row.get("draftBucket", 6)),
        # Contract info
        "term": length,
        "value": value,
        "aav": aav,
        "capHitPct": cap_hit_pct,
        # Season stats
        "gp": int(row.get("gamesPlayed", 0)),
        "goals": int(row.get("goals", 0)),
        "assists": int(row.get("assists", 0)),
        "points": int(row.get("points", 0)),
        "ppg": round(float(row.get("pointsPerGame", 0)), 2),
        "evPoints": int(row.get("evPoints", 0)),
        "toi": round(float(row.get("timeOnIcePerGame", 0)) / 60, 2),
        # Career-to-date
        "careerGP": int(row.get("ctd_gamesPlayed", 0)),
        "careerPoints": int(row.get("ctd_points", 0)),
        "careerEVPoints": int(row.get("ctd_evPoints", 0)),
        "careerPPG": round(float(row.get("ctd_p_pg", 0)), 2),
        "careerEVPPG": round(float(row.get("ctd_ev_p_pg", 0)), 2),
        "careerTOI": round(float(row.get("ctd_toi_avg", 0)) / 60, 2),
        "careerGPPct": round(float(row.get("pct_gp", 0)) * 100, 2),
        "seasonNo": int(row.get("szn_no", 0)),
    }


def build_career_history(player_id: int, stats: pd.DataFrame) -> list[dict]:
    """Get full career season-by-season for plotting."""
    player_data = stats[stats["playerId"] == player_id].sort_values("seasonId")
    history = []
    for _, row in player_data.iterrows():
        history.append({
            "season": int(row["seasonId"]),
            "age": int(row.get("july_1_age", 0)),
            "gp": int(row.get("gamesPlayed", 0)),
            "points": int(row.get("points", 0)),
            "ppg": round(float(row.get("pointsPerGame", 0)), 2),
            "evPoints": int(row.get("evPoints", 0)),
            "evPPG": round(float(row.get("evPointsPerGame", 0)), 2),
            "toi": round(float(row.get("timeOnIcePerGame", 0)) / 60, 2),
        })
    return history


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"Current season: {CURRENT_SEASON}")
    print(f"Fetching {len(ALL_SEASONS)} seasons of stats...")

    # 1. Fetch NHL stats
    raw_stats = fetch_nhl_data(NHL_STATS_URL, ALL_SEASONS)
    print(f"  Got {len(raw_stats)} stat rows")

    # 2. Fetch bios
    raw_bios = fetch_nhl_data(NHL_BIOS_URL, ALL_SEASONS)
    raw_bios = raw_bios.drop(
        columns=["assists", "goals", "gamesPlayed", "points",
                 "isInHallOfFameYn", "birthCity", "birthStateProvinceCode"],
        errors="ignore",
    ).drop_duplicates(subset="playerId")
    print(f"  Got {len(raw_bios)} bio rows")

    # 3. Build stats
    stats = build_stats(raw_stats)
    stats = compute_age(stats, raw_bios)
    print(f"  Built {len(stats)} stat records with features")

    # 4. Fetch contracts
    print("Fetching contracts from PuckPedia...")
    raw_contracts = fetch_contracts()
    contracts = build_contracts(raw_contracts)
    print(f"  Got {len(contracts)} contract rows")

    # 5. Merge
    merged = merge_stats_contracts(stats, contracts)
    print(f"  Merged to {len(merged)} player-seasons")

    # 6. Run 4 KNN models + backtests
    print("Running KNN models...")
    all_comps = {}
    backtest_results = {}

    for fa_status in ["RFA", "UFA"]:
        for pos_group in ["Forward", "Defense"]:
            subset = merged[
                (merged["signing_status"] == fa_status)
                & (merged["position_group"] == pos_group)
            ].reset_index(drop=True)

            if fa_status == "UFA":
                features = UFA_FEATURES
                weights = UFA_WEIGHTS
            else:
                features = RFA_FEATURES
                weights = RFA_WEIGHTS

            # Only include features that exist
            features = [f for f in features if f in subset.columns]

            result = run_knn_model(subset, features, weights)
            for pid, comp_data in result.items():
                all_comps[pid] = comp_data

            # Backtest to measure model accuracy
            bt = backtest_model(merged, fa_status, pos_group, features, weights, CAP_LIMITS)
            model_key = f"{fa_status}_{pos_group}"
            backtest_results[model_key] = {
                "mae": bt["mae"],
                "median_ae": bt["median_ae"],
                "n": bt["n"],
            }
            print(f"  {fa_status} {pos_group}: {len(result)} players | "
                  f"Backtest MAE={bt['mae']}%, Median AE={bt['median_ae']}% (n={bt['n']})")

    # 7. Build output JSON
    print("Building output JSON...")

    # Build term sensitivity curves from historical data
    term_curves = build_term_curves(merged, CAP_LIMITS)

    # Player lookup for all merged data
    player_records = {}
    for _, row in merged.iterrows():
        pid = int(row["playerId"])
        key = f"{pid}_{int(row['contract_year'])}"
        player_records[key] = build_player_record(row, CAP_LIMITS)

    # Current-season pending FAs
    pending_fas = merged[merged["contract_year"] == CURRENT_SEASON].copy()

    output = {
        "meta": {
            "currentSeason": CURRENT_SEASON,
            "currentCap": CAP_LIMITS.get(CURRENT_SEASON, 95_500_000),
            "futureCaps": {
                str(CURRENT_SEASON + 10001): CAP_LIMITS.get(CURRENT_SEASON + 10001, 104_000_000),
                str(CURRENT_SEASON + 20002): CAP_LIMITS.get(CURRENT_SEASON + 20002, 113_500_000),
            },
            "generatedAt": datetime.now().isoformat(),
            "estimationMethod": "inverse_distance_weighting",
            "backtest": backtest_results,
        },
        "players": [],
    }

    for _, row in pending_fas.iterrows():
        pid = int(row["playerId"])
        if pid not in all_comps:
            continue

        player_data = build_player_record(row, CAP_LIMITS)
        comp_info = all_comps[pid]

        # Build comp records
        comp_records = []
        for comp_pid in comp_info["comps"]:
            # Find the comp's contract-year record
            comp_rows = merged[merged["playerId"] == comp_pid]
            if comp_rows.empty:
                continue
            # Use the most recent contract record
            comp_row = comp_rows.sort_values("contract_year").iloc[-1]
            comp_records.append(build_player_record(comp_row, CAP_LIMITS))

        # --- IDW cap hit estimate with confidence interval ---
        comp_pcts = [c["capHitPct"] for c in comp_records]
        comp_dists = comp_info["distances"]
        est = idw_estimate(comp_pcts, comp_dists)

        current_cap = CAP_LIMITS.get(CURRENT_SEASON, 95_500_000)
        estimated_aav = round(est["estimate"] * current_cap / 100)
        aav_low = round(est["ci_low"] * current_cap / 100)
        aav_high = round(est["ci_high"] * current_cap / 100)

        # Career history for chart
        career = build_career_history(pid, stats)
        comp_careers = {}
        for comp_pid in comp_info["comps"]:
            comp_careers[str(comp_pid)] = build_career_history(comp_pid, stats)

        # Term sensitivity table
        fa_status = str(row.get("signing_status", ""))
        pos_group = str(row.get("position_group", ""))
        term_table = compute_term_table(
            est["estimate"], fa_status, pos_group, term_curves, current_cap
        )

        output["players"].append({
            **player_data,
            "estimatedCapHitPct": est["estimate"],
            "estimatedAAV": estimated_aav,
            "ciLow": est["ci_low"],
            "ciHigh": est["ci_high"],
            "aavLow": aav_low,
            "aavHigh": aav_high,
            "estimateStd": est["std"],
            "compWeights": est["weights"],
            "termTable": term_table,
            "comps": comp_records,
            "compDistances": comp_info["distances"],
            "career": career,
            "compCareers": comp_careers,
        })

    # Sort by name
    output["players"].sort(key=lambda p: p["name"])

    # Write JSON
    out_path = os.path.join(OUTPUT_DIR, "comps.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(output['players'])} players to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()

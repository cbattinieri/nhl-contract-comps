"""
NHL Contract Comparables - Data Pipeline (EXTENSION-ELIGIBLE VARIANT)

Duplicate of generate_data.py. The ONLY substantive difference is the target
pool: instead of players who have become free agents (0 current contracts),
this variant targets players in the FINAL YEAR of their current contract
(contract expires next season) — i.e. players who are eligible to sign an
extension now. Everything else (KNN models, weights, term tables, IDW) is
identical and reused unchanged.

Writes data/comps_ext.json (loaded by the "Extension Eligible" UI toggle) and
data/fa_class_ext_{YEAR}.json. Local-only; not wired into the deploy chain.
Keep model/logic changes in sync with generate_data.py.
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
# During the NHL off-season (June–September), PuckPedia clears expired contracts
# and rolls to the upcoming season. Advance to match so pending FAs are found.
NOW = datetime.now()
if NOW.month >= 10:
    CURRENT_YEAR = NOW.year       # Oct 2026 → 2026-27 season
elif NOW.month >= 6:
    CURRENT_YEAR = NOW.year       # Jun-Sep 2026 → advance to 2026-27 (off-season)
else:
    CURRENT_YEAR = NOW.year - 1   # Jan-May 2026 → still 2025-26 season
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

# Actual NHL regular-season game counts for shortened seasons; default 82.
SEASON_LENGTHS = {
    20042005: 0,   # lockout — no games
    20122013: 48,
    20192020: 70,
    20202021: 56,
}
DEFAULT_SEASON_LENGTH = 82

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


def fetch_raw_puckpedia() -> list:
    """Fetch raw player records from PuckPedia API."""
    if not PUCKPEDIA_API_KEY:
        raise RuntimeError("PUCKPEDIA_API_KEY env var is required")
    resp = requests.get(
        f"https://puckpedia.com/api/players2?api_key={PUCKPEDIA_API_KEY}",
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def load_previous_fa_class() -> dict:
    """Load existing comps_ext.json if it belongs to the same class year."""
    out_path = os.path.join(OUTPUT_DIR, "comps_ext.json")
    if not os.path.exists(out_path):
        return {}
    try:
        with open(out_path) as f:
            data = json.load(f)
        if data.get("meta", {}).get("faClassYear") == CURRENT_YEAR:
            return data
    except Exception:
        pass
    return {}


OFFSEASON_CUTOFF_MMDD = "04-01"  # April 1 — captures playoff/post-season signings

def get_actual_contract(pp_record: dict, signing_year: int) -> dict | None:
    """Return the contract a player signed this offseason, or None if not yet signed."""
    for c in pp_record.get("current", []):
        signing_date = (c.get("signing_date") or "")
        # Contract signed on or after April 1 of the signing year
        if signing_date >= f"{signing_year}-{OFFSEASON_CUTOFF_MMDD}":
            try:
                length = int(float(c.get("length") or 1))
                value  = int(float(c.get("value")  or 0))
            except (ValueError, TypeError):
                continue
            aav       = round(value / length) if length > 0 else 0
            cap_limit = CAP_LIMITS.get(CURRENT_SEASON, 95_500_000)
            return {
                "term":          length,
                "value":         value,
                "aav":           aav,
                "capHitPct":     round((aav / cap_limit) * 100, 2) if cap_limit > 0 else 0,
                "signingDate":   signing_date,
                "team":          c.get("team_name", ""),
                "signingStatus": c.get("signing_status", ""),
            }
    return None


def get_actual_extension(pp_record: dict) -> dict | None:
    """Return a future-dated extension (signed while still under contract), or None.

    An extension-eligible player who has ALREADY signed keeps his in-progress
    deal in `current` and the freshly-signed extension in PuckPedia's `future`
    array (it begins in a later season). We surface it as the 'actual' contract
    so the UI can show estimate-vs-actual. Cap-hit % is measured against the cap
    of the season the extension STARTS, which is the correct market basis and is
    comparable to the model's % estimate even across cap years.
    """
    for c in pp_record.get("future", []):
        try:
            length = int(float(c.get("length") or 1))
            value  = int(float(c.get("value")  or 0))
        except (ValueError, TypeError):
            continue
        # Prefer PuckPedia's stated first-year AAV; fall back to value / length.
        try:
            aav = int(float(c.get("current_season_aav") or 0))
        except (ValueError, TypeError):
            aav = 0
        if aav <= 0:
            aav = round(value / length) if length > 0 else 0
        # Cap basis = the season the extension starts (e.g. "2027-2028").
        start_season = (c.get("current_season") or "")
        try:
            start_season_id = int(start_season.replace("-", "")) if start_season else CURRENT_SEASON
        except (ValueError, TypeError):
            start_season_id = CURRENT_SEASON
        cap_limit = CAP_LIMITS.get(start_season_id) or CAP_LIMITS.get(CURRENT_SEASON, 95_500_000)
        return {
            "term":          length,
            "value":         value,
            "aav":           aav,
            "capHitPct":     round((aav / cap_limit) * 100, 2) if cap_limit > 0 else 0,
            "signingDate":   c.get("signing_date", ""),
            "team":          c.get("team_name", ""),
            "signingStatus": c.get("signing_status") or c.get("expiry_status", ""),
            "startSeason":   start_season,
            "isExtension":   True,
        }
    return None


def fetch_contracts(puckpedia_records: list) -> pd.DataFrame:
    """Build contracts DataFrame from raw PuckPedia records.

    Returns a combined DataFrame of:
      1. Historical/active contracts (for KNN training)
      2. Extension-eligible TARGET rows (players in the final year of their
         current contract — expiring next offseason). These rows have
         is_pending_fa=True and their contract value/length nulled so the KNN
         treats them as targets; signing_status is inferred from ufa_year.
    """
    records = puckpedia_records
    df = pd.DataFrame(records)

    # ── 1. ACTIVE CONTRACTS (historical comps) ──────────────────────────────
    df_with = df[df["current"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    df_exploded = df_with.explode("current")
    df_active = pd.concat(
        [df_exploded.drop(columns=["current"]),
         df_exploded["current"].apply(pd.Series)],
        axis=1,
    )
    df_active = df_active.drop(columns=["future"], errors="ignore")
    df_active = df_active.loc[:, ~df_active.columns.duplicated()]  # drop duplicate cols
    df_active = df_active[df_active["contract_id"].notna()].copy()
    df_active["is_pending_fa"] = False

    # ── 2. EXTENSION-ELIGIBLE PLAYERS (final year of current contract) ───────
    # Under the CBA a player can sign an extension starting July 1 before the
    # final season of their current SPC. We treat these players as "pending"
    # TARGETS: estimate what a fresh extension would cost, using the same KNN
    # comps model as the FA class. They are sourced from the active-contract
    # pool (they ARE signed) — the ones whose contract expires next season
    # (contract_end year == CURRENT_YEAR + 1).
    #
    # Their own current contract stays in df_active as a comp row, but
    # run_knn_model() excludes any playerId present in the pending pool from the
    # historical/comp pool, so a player can never become his own comp.
    #
    # Eligibility test: the current contract's FINAL season is the upcoming
    # season (CURRENT_SEASON). PuckPedia encodes contract_end as the final
    # season string "YYYY-YYYY" (e.g. a deal expiring in July 2027 has its last
    # season 2026-27 → "2026-2027"). This is the same convention the FA pipeline
    # uses for its signing-season check. NOTE: contract_end's first year is
    # CURRENT_YEAR, not CURRENT_YEAR+1 — the expiry (July) is CURRENT_YEAR+1.
    ext_final_season = f"{CURRENT_YEAR}-{CURRENT_YEAR + 1}"   # e.g. "2026-2027"
    ext_mask = (
        (df_active["contract_end"].astype(str) == ext_final_season)
        & df_active["nhl_id"].notna()
    )
    df_free = df_active[ext_mask].copy()

    def infer_status(ufa_yr):
        # When the extension begins (after the current deal ends next season) will
        # the player be UFA or RFA? Compare ufa_year to the extension's start year.
        try:
            return "UFA" if int(ufa_yr or 9999) <= (CURRENT_YEAR + 1) else "RFA"
        except (ValueError, TypeError):
            return "UFA"

    df_free["signing_status"] = df_free["ufa_year"].apply(infer_status)
    df_free["expiry_status"]   = df_free["signing_status"]
    df_free["is_pending_fa"]   = True
    df_free["is_retro_signed"] = False
    df_free["contract_id"]     = "ext_pending"
    df_free["contract_end"]    = None
    df_free["length"]          = np.nan
    df_free["value"]           = np.nan

    print(f"  Extension-eligible players (final season {ext_final_season}, "
          f"expire {CURRENT_YEAR + 1}): {len(df_free)}")

    # ── 3. (No retro-signed pool for the extension tool) ─────────────────────
    # The retro-signed pool exists to catch FAs who signed before tracking began.
    # It does not apply to the extension cohort (these players are all under
    # contract by definition), so it stays empty here.
    df_retro = pd.DataFrame(columns=df_free.columns)

    # ── 4. COMBINE ────────────────────────────────────────────────────────────
    return pd.concat([df_active, df_free, df_retro], ignore_index=True)


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

    # ── Candidate features (additive; validated in validate_features.py) ──
    df["season_start_year"] = df["seasonId"] // 10000
    debut = df.groupby("playerId")["seasonId"].transform("min")
    df["debut_season"] = debut
    debut_year = debut // 10000
    df["seasons_since_debut"] = (df["season_start_year"] - debut_year + 1).astype(int)

    _base = int(df["season_start_year"].min())
    _top = int(df["season_start_year"].max())
    _cum, _run = {_base - 1: 0}, 0
    for _y in range(_base, _top + 1):
        _run += SEASON_LENGTHS.get(_y * 10000 + (_y + 1), DEFAULT_SEASON_LENGTH)
        _cum[_y] = _run
    df["possible_gp"] = (
        df["season_start_year"].map(_cum) - (debut_year - 1).map(_cum)
    )

    df["pct_gp_v2"] = (
        (df["ctd_gamesPlayed"] / df["possible_gp"].replace(0, np.nan))
        .clip(lower=0, upper=1).fillna(0).round(4)
    )

    df["platform_gp"] = df["gamesPlayed"]

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
    LAST_STATS_SEASON = CURRENT_SEASON - 10001  # most recent season with stats

    # ── Active contracts: derive contract_year from contract_end + length ────
    active = df["is_pending_fa"] != True
    df.loc[active, "contract_end_year"] = (
        df.loc[active, "contract_end"].astype(str).str.split("-").str[0].astype(int)
    )
    df.loc[active, "contract_start_year"] = (
        df.loc[active, "contract_end_year"] - df.loc[active, "length"].astype(float).astype(int)
    )
    df.loc[active, "contract_year"] = (
        df.loc[active, "contract_start_year"] * 10000
        + df.loc[active, "contract_start_year"]
        + 1
    ).astype(int)

    # signing_status for active contracts from expiry_status on the signing season
    is_signing_season = df.loc[active, "contract_end"] == f"{CURRENT_YEAR}-{CURRENT_YEAR + 1}"
    signing_idx = df[active].index[is_signing_season]
    df.loc[signing_idx, "signing_status"] = df.loc[signing_idx, "expiry_status"]

    # ── Pending FAs: use most recent stats season for the merge ──────────────
    pending = df["is_pending_fa"] == True
    df.loc[pending, "contract_year"] = LAST_STATS_SEASON

    df = df.dropna(subset=["nhl_id"])
    df["nhl_id"] = df["nhl_id"].astype(int)

    # Drop ELC slides for active contracts only
    df = df[(df["signing_status"] != False) | (df["is_pending_fa"] == True)]  # noqa: E712

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

# Features used across all models
BASE_FEATURES = [
    "july_1_age", "ctd_gamesPlayed", "ctd_p_pg", "ctd_ev_p_pg",
    "pct_gp", "ctd_toi_avg",
]
# Recent-form features. See docs/model_review_2026-07.md for validation.
UFA_EXTRA_FEATURES = [
    "gamesPlayed_L2", "pointsPerGame_L2", "evPointsPerGame_L2",
    "gamesPlayed_L3", "pointsPerGame_L3", "evPointsPerGame_L3",
]
UFA_WEIGHTS = {
    "july_1_age": 15, "ctd_gamesPlayed": 6, "ctd_p_pg": 6,
    "pointsPerGame_L2": 8, "evPointsPerGame_L2": 7,
    "pointsPerGame_L3": 3, "evPointsPerGame_L3": 3, "gamesPlayed_L3": 1,
}
# RFA Forwards are platform-year driven; RFA Defense stays on career baseline
# (reweighting degraded its backtest — see review doc).
RFA_FWD_EXTRA_FEATURES = [
    "pointsPerGame_L2", "evPointsPerGame_L2", "gamesPlayed_L2",
]
RFA_FWD_WEIGHTS = {
    "pointsPerGame_L2": 10, "evPointsPerGame_L2": 8,
    "ctd_p_pg": 4, "july_1_age": 3, "pct_gp": 2,
}


def model_config(fa_status: str, pos_group: str) -> tuple[list[str], dict | None]:
    """Return (features, feature_weights) for a given model cell."""
    if fa_status == "UFA":
        return BASE_FEATURES + UFA_EXTRA_FEATURES, UFA_WEIGHTS
    if fa_status == "RFA" and pos_group == "Forward":
        return BASE_FEATURES + RFA_FWD_EXTRA_FEATURES, RFA_FWD_WEIGHTS
    return BASE_FEATURES, None


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

    pending = data[data["is_pending_fa"] == True].copy()
    # Exclude any player who appears in the pending pool from historical, even if
    # they also have an active contract row (e.g. retro-signed players whose new
    # deal is dated to the current or prior season by PuckPedia). Without this,
    # a retro-signed player can appear in both pools and become their own comp.
    pending_player_ids = set(pending["playerId"].unique())
    historical = data[
        (data["is_pending_fa"] != True)
        & ~data["playerId"].isin(pending_player_ids)
        & (data["contract_year"] >= min_season)
        & (data["contract_year"] < CURRENT_SEASON)
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

# Cap-hit tiers used to bucket the term/AAV relationship. A $12M player and a
# $2M player trade term for money at very different rates, so slopes are fit
# separately within each tier.
TERM_SLOPE_TIERS = [("high", 8.0, np.inf), ("mid", 5.0, 8.0), ("low", 0.0, 5.0)]

# Used only when a tier has too few historical signings to fit a reliable slope.
TERM_SLOPE_FALLBACKS = {
    "RFA": {"high": 0.25, "mid": 0.15, "low": 0.08},
    "UFA": {"high": -0.20, "mid": -0.12, "low": -0.06},
}
TERM_SLOPE_MIN_CONTRACTS = 15


def derive_term_slopes(merged: pd.DataFrame, cap_limits: dict) -> dict:
    """Fit term-vs-cap-hit% slopes from actual historical signings.

    Replaces hand-picked slope constants with linear regressions of capHitPct
    on contract term, run separately per (signing_status, cap-hit tier) on
    real signed contracts. RFA slopes come out positive (teams extract a term
    discount), UFA slopes negative (players trade term for security) — but the
    magnitude is now backed by data instead of guessed.

    Returns {fa_status: {tier_name: {"slope": float, "n": int, "fitted": bool}}}
    """
    signed = merged[merged["is_pending_fa"] != True].copy()  # noqa: E712
    signed["length"] = pd.to_numeric(signed["length"], errors="coerce")
    signed["value"] = pd.to_numeric(signed["value"], errors="coerce")
    signed = signed[
        signed["length"].notna() & signed["value"].notna() & (signed["length"] >= 1)
    ].copy()
    signed["aav"] = signed["value"] / signed["length"]
    signed["cap_limit"] = signed["contract_year"].astype(int).map(
        lambda cy: cap_limits.get(cy, 95_500_000)
    )
    signed["cap_hit_pct"] = signed["aav"] / signed["cap_limit"] * 100

    slopes = {}
    for fa_status in ["RFA", "UFA"]:
        sub = signed[signed["signing_status"] == fa_status]
        tier_slopes = {}
        for tier_name, lo, hi in TERM_SLOPE_TIERS:
            tier_data = sub[(sub["cap_hit_pct"] >= lo) & (sub["cap_hit_pct"] < hi)]
            fallback = TERM_SLOPE_FALLBACKS[fa_status][tier_name]
            if len(tier_data) >= TERM_SLOPE_MIN_CONTRACTS:
                fitted_slope, _ = np.polyfit(tier_data["length"], tier_data["cap_hit_pct"], 1)
                fitted_slope = round(float(fitted_slope), 4)
                # Talent confound guard: across the full population, better players get
                # BOTH higher AAV% and longer term, so a naive regression often recovers
                # "talent level" rather than the true term-for-security trade-off. Reject
                # any fit whose sign contradicts the known economics (RFA: team extracts
                # more for term => slope >= 0; UFA: player trades term for AAV => slope <= 0)
                # rather than ship an economically-backwards term table.
                expected_sign_ok = (fitted_slope >= 0) if fa_status == "RFA" else (fitted_slope <= 0)
                if expected_sign_ok:
                    tier_slopes[tier_name] = {"slope": fitted_slope, "n": int(len(tier_data)), "fitted": True}
                else:
                    tier_slopes[tier_name] = {
                        "slope": fallback, "n": int(len(tier_data)),
                        "fitted": False, "rejected_fit": fitted_slope, "reject_reason": "wrong_sign",
                    }
            else:
                tier_slopes[tier_name] = {"slope": fallback, "n": int(len(tier_data)), "fitted": False}
        slopes[fa_status] = tier_slopes
    return slopes


def fit_ufa_adjustments(merged: pd.DataFrame, cap_limits: dict) -> dict:
    """Fit bridge-discount and UFA-premium factors from historical RFA signings.

    For each historical RFA contract, compute how many UFA years were purchased
    (end_age - ufa_eligibility_age, floored at 0). Then fit a log-linear regression
    of cap_hit_pct on ufa_years_bought within each cap-hit tier.

    Returns:
        {tier: {"bridge_factor": float, "ufa_slope": float, "n_bridge": int, "n_ufa": int}}
    """
    rfa = merged[
        (merged["signing_status"] == "RFA") &
        (merged["is_pending_fa"] != True)  # noqa: E712
    ].copy()

    rfa["length"] = pd.to_numeric(rfa["length"], errors="coerce")
    rfa["value"]  = pd.to_numeric(rfa["value"],  errors="coerce")
    rfa = rfa[rfa["length"].notna() & rfa["value"].notna() & (rfa["length"] >= 1)].copy()
    rfa["aav"]         = rfa["value"] / rfa["length"]
    rfa["cap_limit"]   = rfa["contract_year"].astype(int).map(lambda cy: cap_limits.get(cy, 95_500_000))
    rfa["cap_hit_pct"] = rfa["aav"] / rfa["cap_limit"] * 100

    def _ufa_yrs(row):
        age  = int(row.get("july_1_age", 0))
        term = int(row.get("length", 1))
        cy   = int(row.get("contract_year", CURRENT_SEASON))
        signing_year = cy // 10000
        ufa_yr = row.get("ufa_year")
        try:
            ufa_age = int(ufa_yr) - signing_year + age if pd.notna(ufa_yr) else 27
        except (ValueError, TypeError):
            ufa_age = 27
        return max(0, age + term - ufa_age), age + term < ufa_age

    rfa[["ufa_yrs_bought", "is_bridge"]] = rfa.apply(
        lambda r: pd.Series(_ufa_yrs(r)), axis=1
    )

    results = {}
    for tier_name, lo, hi in TERM_SLOPE_TIERS:
        tier = rfa[(rfa["cap_hit_pct"] >= lo) & (rfa["cap_hit_pct"] < hi)]
        bridge  = tier[tier["is_bridge"]]
        ufa_buy = tier[~tier["is_bridge"]]

        # Bridge discount: median ratio of bridge cap% to 1-UFA-yr cap%
        one_ufa = ufa_buy[ufa_buy["ufa_yrs_bought"] == 1]
        if len(bridge) >= 3 and len(one_ufa) >= 3:
            bridge_factor = float(np.median(bridge["cap_hit_pct"]) / np.median(one_ufa["cap_hit_pct"]))
        else:
            bridge_factor = 0.55  # conservative fallback

        # UFA slope: log-linear regression of cap_hit_pct on ufa_years_bought
        if len(ufa_buy) >= 5:
            try:
                ufa_slope, _ = np.polyfit(ufa_buy["ufa_yrs_bought"], ufa_buy["cap_hit_pct"], 1)
                ufa_slope = max(0.0, round(float(ufa_slope), 4))  # must be positive
            except Exception:
                ufa_slope = 0.5
        else:
            ufa_slope = 0.5

        results[tier_name] = {
            "bridge_factor": round(bridge_factor, 4),
            "ufa_slope":     round(ufa_slope, 4),
            "n_bridge":      int(len(bridge)),
            "n_ufa":         int(len(ufa_buy)),
        }
        print(f"  UFA adj {tier_name}: bridge_factor={bridge_factor:.3f}, "
              f"ufa_slope={ufa_slope:.3f} (n_bridge={len(bridge)}, n_ufa={len(ufa_buy)})")

    return results


def compute_term_table(
    base_cap_pct: float,
    fa_status: str,
    comp_records: list[dict],
    current_cap: int,
    term_slopes: dict,
    ufa_adjustments: dict | None = None,
    player_age: int = 0,
    player_ufa_year: int | None = None,
) -> list[dict]:
    """Compute adjusted AAV estimate at each term length (1–8 years).

    RFA: two-regime model anchored at the UFA crossover term.
      - Bridge years (expire before UFA eligibility): apply fitted bridge discount.
      - UFA-buying years: apply fitted premium per UFA year purchased.
    UFA: linear slope (player trades term for AAV security).
    """
    comp_terms = [
        c["term"] for c in comp_records
        if c.get("term") and isinstance(c.get("term"), (int, float)) and c["term"] >= 1
    ]
    ref_term = float(np.mean(comp_terms)) if comp_terms else 4.0
    tier_name = "high" if base_cap_pct >= 8.0 else "mid" if base_cap_pct >= 5.0 else "low"

    # ── UFA: keep existing linear slope logic ────────────────────────────────
    if fa_status != "RFA" or ufa_adjustments is None:
        fallback = TERM_SLOPE_FALLBACKS[fa_status][tier_name]
        slope = term_slopes.get(fa_status, {}).get(tier_name, {}).get("slope", fallback)
        table = []
        for t in range(1, 9):
            adj = base_cap_pct + slope * (t - ref_term)
            adj = max(base_cap_pct * 0.65, min(base_cap_pct * 1.35, adj))
            adj = round(max(0.0, adj), 2)
            table.append({"term": t, "capHitPct": adj, "aav": round(adj * current_cap / 100)})
        return table

    # ── RFA: two-regime model ─────────────────────────────────────────────────
    adj_params  = ufa_adjustments.get(tier_name, {})
    bridge_factor = adj_params.get("bridge_factor", 0.55)
    ufa_slope     = adj_params.get("ufa_slope", 0.5)

    # Determine how many years from now the player hits UFA
    # ufa_crossover_term = number of years until UFA eligibility
    if player_ufa_year and player_age:
        ufa_crossover_term = max(1, player_ufa_year - CURRENT_YEAR)
    else:
        ufa_crossover_term = max(1, 27 - player_age) if player_age else 4

    # Anchor: base estimate is calibrated at the crossover term (or avg comp term)
    # Use the crossover term as the anchor so the model estimate is exactly right
    # when the slider is at the UFA threshold.
    anchor_term = ufa_crossover_term

    table = []
    for t in range(1, 9):
        ufa_yrs = max(0, t - anchor_term)
        is_bridge = t < anchor_term

        if is_bridge:
            # Scale bridge discount: deeper discount for shorter bridge
            # bridge_factor is the discount at a 1yr bridge; scale linearly toward 1.0 at crossover
            years_from_crossover = anchor_term - t
            bridge_depth = bridge_factor + (1.0 - bridge_factor) * (1 - years_from_crossover / max(anchor_term - 1, 1))
            adj = base_cap_pct * bridge_depth
        elif ufa_yrs == 0:
            # Crossover term — base estimate
            adj = base_cap_pct
        else:
            # UFA-buying: add premium per UFA year purchased
            adj = base_cap_pct + ufa_slope * ufa_yrs

        adj = round(max(0.0, adj), 2)
        ufa_yrs_label = ufa_yrs if not is_bridge else 0
        table.append({
            "term": t,
            "capHitPct": adj,
            "aav": round(adj * current_cap / 100),
            "regime": "bridge" if is_bridge else ("crossover" if ufa_yrs == 0 else "ufa"),
            "ufaYrsBought": int(ufa_yrs_label),
        })
    return table


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

    ufa_yr = row.get("ufa_year")
    try:
        ufa_year_out = int(ufa_yr) if pd.notna(ufa_yr) else None
    except (ValueError, TypeError):
        ufa_year_out = None

    return {
        "playerId": int(row["playerId"]),
        "name": str(row.get("skaterFullName", "")),
        "position": str(row.get("position_display", "")),
        "positionGroup": str(row.get("position_group", "")),
        "signingStatus": str(row.get("signing_status", "")),
        "contractYear": contract_year,
        "age": int(row.get("july_1_age", 0)),
        "ufaYear": ufa_year_out,
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

    # 0. Load previous FA class (for tracking signed players)
    prev_fa_data = load_previous_fa_class()
    prev_fa_ids  = {p["playerId"]: p for p in prev_fa_data.get("players", [])}
    if prev_fa_ids:
        print(f"  Loaded {len(prev_fa_ids)} players from previous FA class snapshot")

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

    # 4. Fetch contracts (single PuckPedia API call, reused for signed tracking)
    print("Fetching contracts from PuckPedia...")
    puckpedia_records = fetch_raw_puckpedia()
    pp_by_nhl_id = {int(r["nhl_id"]): r for r in puckpedia_records if r.get("nhl_id")}
    raw_contracts = fetch_contracts(puckpedia_records)
    contracts = build_contracts(raw_contracts)
    print(f"  Got {len(contracts)} contract rows")

    # 5. Merge
    merged = merge_stats_contracts(stats, contracts)
    print(f"  Merged to {len(merged)} player-seasons")

    # 6. Derive term-vs-cap-hit% slopes and UFA adjustment factors
    term_slopes = derive_term_slopes(merged, CAP_LIMITS)
    print("Fitting UFA premium adjustments for RFA term table...")
    ufa_adjustments = fit_ufa_adjustments(merged, CAP_LIMITS)
    for fa_status, tiers in term_slopes.items():
        for tier_name, info in tiers.items():
            if info["fitted"]:
                src = "fitted"
            elif "rejected_fit" in info:
                src = f"fallback (fitted {info['rejected_fit']} rejected: {info['reject_reason']})"
            else:
                src = "fallback (insufficient data)"
            print(f"  Term slope {fa_status}/{tier_name}: {info['slope']} (n={info['n']}, {src})")

    # 7. Run 4 KNN models + backtests
    print("Running KNN models...")
    all_comps = {}
    backtest_results = {}

    for fa_status in ["RFA", "UFA"]:
        for pos_group in ["Forward", "Defense"]:
            subset = merged[
                (merged["signing_status"] == fa_status)
                & (merged["position_group"] == pos_group)
            ].reset_index(drop=True)

            features, weights = model_config(fa_status, pos_group)

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

    # 8. Build output JSON
    print("Building output JSON...")

    # Player lookup for all merged data
    player_records = {}
    for _, row in merged.iterrows():
        pid = int(row["playerId"])
        key = f"{pid}_{int(row['contract_year'])}"
        player_records[key] = build_player_record(row, CAP_LIMITS)

    # Current-season pending FAs — only players whose contracts actually expire this season
    pending_fas = merged[merged["is_pending_fa"] == True].copy()

    output = {
        "meta": {
            "currentSeason": CURRENT_SEASON,
            "faClassYear": CURRENT_YEAR,
            "currentCap": CAP_LIMITS.get(CURRENT_SEASON, 95_500_000),
            "futureCaps": {
                str(CURRENT_SEASON + 10001): CAP_LIMITS.get(CURRENT_SEASON + 10001, 104_000_000),
                str(CURRENT_SEASON + 20002): CAP_LIMITS.get(CURRENT_SEASON + 20002, 113_500_000),
            },
            "generatedAt": datetime.now().isoformat(),
            "estimationMethod": "inverse_distance_weighting",
            "backtest": backtest_results,
            "termSlopes": term_slopes,
        },
        "players": [],
    }

    pending_fa_player_ids = set(pending_fas["playerId"].unique())

    for _, row in pending_fas.iterrows():
        pid = int(row["playerId"])
        if pid not in all_comps:
            continue

        player_data = build_player_record(row, CAP_LIMITS)
        comp_info = all_comps[pid]

        # Build comp records — use historical signed contracts only.
        # Exclude any player in the current FA class to prevent leakage from
        # retro-signed players whose new contract is dated to the current/prior season.
        comp_records = []
        comp_dists = []
        for comp_pid, dist in zip(comp_info["comps"], comp_info["distances"]):
            if comp_pid in pending_fa_player_ids:
                continue
            comp_rows = merged[
                (merged["playerId"] == comp_pid) &
                (merged["is_pending_fa"] != True)
            ]
            if comp_rows.empty:
                continue
            # Use the most recent signed contract record
            comp_row = comp_rows.sort_values("contract_year").iloc[-1]
            comp_records.append(build_player_record(comp_row, CAP_LIMITS))
            comp_dists.append(dist)

        # --- IDW cap hit estimate with confidence interval ---
        comp_pcts = [c["capHitPct"] for c in comp_records]
        est = idw_estimate(comp_pcts, comp_dists)

        current_cap = CAP_LIMITS.get(CURRENT_SEASON, 95_500_000)
        estimated_aav = round(est["estimate"] * current_cap / 100)
        aav_low = round(est["ci_low"] * current_cap / 100)
        aav_high = round(est["ci_high"] * current_cap / 100)

        # --- Term sensitivity table ---
        fa_status = str(row.get("signing_status", "UFA"))
        p_age      = int(row.get("july_1_age", 0))
        p_ufa_year = None
        try:
            raw_ufa = row.get("ufa_year")
            p_ufa_year = int(raw_ufa) if pd.notna(raw_ufa) else None
        except (ValueError, TypeError):
            pass
        term_table = compute_term_table(
            est["estimate"], fa_status, comp_records, current_cap, term_slopes,
            ufa_adjustments=ufa_adjustments if fa_status == "RFA" else None,
            player_age=p_age,
            player_ufa_year=p_ufa_year,
        )
        comp_terms = [c["term"] for c in comp_records if c.get("term") and c["term"] >= 1]
        avg_comp_term = round(float(np.mean(comp_terms)), 1) if comp_terms else 4.0

        # Career history for chart
        career = build_career_history(pid, stats)
        comp_careers = {}
        for comp_pid in comp_info["comps"]:
            comp_careers[str(comp_pid)] = build_career_history(comp_pid, stats)

        # Extension-eligible players who have ALREADY signed keep their new deal
        # in PuckPedia's `future` array (a future-dated extension). Surface it as
        # the actual contract so the card shows estimate-vs-actual, and mark them
        # signed. e.g. Tyson Foerster: bridge in `current` (ends 2026-27), 8yr /
        # $7.1M extension in `future` (starts 2027-28).
        actual_contract = None
        pp_record = pp_by_nhl_id.get(pid)
        if pp_record:
            actual_contract = get_actual_extension(pp_record)

        output["players"].append({
            **player_data,
            "signed": actual_contract is not None,
            "actualContract": actual_contract,
            "estimatedCapHitPct": est["estimate"],
            "estimatedAAV": estimated_aav,
            "ciLow": est["ci_low"],
            "ciHigh": est["ci_high"],
            "aavLow": aav_low,
            "aavHigh": aav_high,
            "estimateStd": est["std"],
            "compWeights": est["weights"],
            "avgCompTerm": avg_comp_term,
            "termTable": term_table,
            "comps": comp_records,
            "compDistances": comp_info["distances"],
            "career": career,
            "compCareers": comp_careers,
        })

    # ── Carry forward previously-pending players who have since signed ────────
    current_output_ids = {p["playerId"] for p in output["players"]}
    signed_count = 0

    for pid, prev_player in prev_fa_ids.items():
        if pid in current_output_ids:
            continue  # still pending this run, already included above

        if prev_player.get("signed"):
            # An older run may have mistakenly tagged an ELC slide as a signed FA
            # (see the ELC guard above) — drop those instead of re-propagating them.
            if prev_player.get("actualContract", {}).get("signingStatus") is False:
                continue
            # Already confirmed signed in a previous run — carry forward as-is
            output["players"].append(prev_player)
            signed_count += 1
        else:
            # Was pending — check if they signed an extension since last run
            pp_record = pp_by_nhl_id.get(pid)
            if not pp_record:
                continue
            actual = get_actual_extension(pp_record)
            if actual:
                signed_player = dict(prev_player)
                signed_player["signed"] = True
                signed_player["actualContract"] = actual
                output["players"].append(signed_player)
                signed_count += 1

    print(f"  {signed_count} previously-pending players carried forward as signed")

    # Sort: pending first (alphabetical), then signed (alphabetical)
    output["players"].sort(key=lambda p: (p.get("signed", False), p["name"]))

    # Flag the mode so the UI can label the view.
    output["meta"]["mode"] = "extension_eligible"

    # ── Write fa_class_ext_{CURRENT_YEAR}.json — extension-eligible cohort ────
    archive_path = os.path.join(OUTPUT_DIR, f"fa_class_ext_{CURRENT_YEAR}.json")
    signed_players   = [p for p in output["players"] if p.get("signed")]
    unsigned_players = [p for p in output["players"] if not p.get("signed")]
    archive_data = {
        "meta": {
            "faClassYear":  CURRENT_YEAR,
            "mode":         "extension_eligible",
            "generatedAt":  datetime.now().isoformat(),
            "signedCount":  len(signed_players),
            "pendingCount": len(unsigned_players),
            "currentCap":   CAP_LIMITS.get(CURRENT_SEASON, 95_500_000),
        },
        "players": output["players"],  # full cohort
    }
    with open(archive_path, "w") as f:
        json.dump(archive_data, f, indent=2)
    print(f"  Wrote extension archive: fa_class_ext_{CURRENT_YEAR}.json  "
          f"({len(signed_players)} signed, {len(unsigned_players)} pending)")

    # ── Scan for available archive years and embed in meta ───────────────────
    available_archives = []
    for yr in range(CURRENT_YEAR - 5, CURRENT_YEAR + 1):
        if os.path.exists(os.path.join(OUTPUT_DIR, f"fa_class_ext_{yr}.json")):
            available_archives.append(yr)
    output["meta"]["archives"] = available_archives

    # Write main comps_ext.json
    out_path = os.path.join(OUTPUT_DIR, "comps_ext.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(output['players'])} players to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()

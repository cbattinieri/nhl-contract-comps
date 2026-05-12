# pip install pymc arviz numpy pandas scikit-learn matplotlib requests pyarrow
# Usage: python aging_model.py  (requires PUCKPEDIA_API_KEY env var)

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PUCKPEDIA_API_KEY = os.environ.get("PUCKPEDIA_API_KEY", "")

NOW = datetime.now()
CURRENT_YEAR = NOW.year if NOW.month >= 7 else NOW.year - 1
CURRENT_SEASON = int(f"{CURRENT_YEAR}{CURRENT_YEAR + 1}")

_AGING_START = 20072008

_KNN_SEASONS = [
    f"{y}{y+1}" for y in range(2003, CURRENT_YEAR + 1)
    if f"{y}{y+1}" != "20042005"
]

MIN_GP           = 20
KNN_NEIGHBORS    = 5
IDW_EPSILON      = 1e-6
LOOKBACK_SEASONS = 3

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
    20252026: 95_500_000,
    20262027: 104_000_000,
    20272028: 113_500_000,
}
# Seasons beyond 2027-28 are filled at runtime by the Monte Carlo cap model.
FUTURE_CAPS_PROJECTED: set = set()  # populated by build_cap_projections()

# Historical caps for MC growth-rate training (anomalous years excluded at runtime)
_CAP_HISTORY_FOR_MC = {
    2005: 39_000_000,
    2006: 44_000_000,
    2007: 50_300_000,
    2008: 56_700_000,
    2009: 56_800_000,
    2010: 59_400_000,
    2011: 64_300_000,
    # 2012 excluded — lockout reset
    2013: 64_300_000,
    2014: 69_000_000,
    2015: 71_400_000,
    2016: 73_000_000,
    2017: 75_000_000,
    2018: 79_500_000,
    2019: 81_500_000,
    # 2020, 2021 excluded — COVID freeze
    2022: 82_500_000,
    2023: 83_500_000,
    2024: 88_000_000,
}

# Known future caps — treated as hard anchors, not training data
_KNOWN_FUTURE_CAPS = {
    2025: 95_500_000,
    2026: 104_000_000,
    2027: 113_500_000,
}

# Populated by build_cap_projections(); keyed by calendar year (int)
_CAP_PROJECTIONS: pd.DataFrame | None = None

AGE_CENTER     = {"Forward": 26, "Defense": 27}
PRIMARY_METRIC = {"Forward": "ppg", "Defense": "toi_per_game"}

_NHL_STATS_URL = (
    "https://api.nhle.com/stats/rest/en/skater/summary"
    "?sort=points&limit=-1&cayenneExp=seasonId="
)
_NHL_BIOS_URL = (
    "https://api.nhle.com/stats/rest/en/skater/bios"
    "?limit=-1&cayenneExp=seasonId="
)

BASE_FEATURES = [
    "july_1_age", "ctd_gamesPlayed", "ctd_p_pg", "ctd_ev_p_pg",
    "pct_gp", "ctd_toi_avg",
]
L3_FEATURES = [
    "gamesPlayed_L3", "points_L3", "pointsPerGame_L3",
    "evPoints_L3", "evPointsPerGame_L3", "timeOnIcePerGame_L3",
]
ALL_KNN_FEATURES = BASE_FEATURES + L3_FEATURES

RFA_WEIGHTS = None
UFA_WEIGHTS = {
    "gamesPlayed_L3": 2, "points_L3": 2, "pointsPerGame_L3": 2,
    "evPoints_L3": 2, "evPointsPerGame_L3": 2, "timeOnIcePerGame_L3": 2,
}

_L3_SET        = set(L3_FEATURES)
_PROD_FEATURES = {"ctd_p_pg", "ctd_ev_p_pg", "ctd_toi_avg"}
_SLOW_FEATURES = {"ctd_gamesPlayed", "pct_gp"}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_nhl(base_url: str, seasons: list) -> pd.DataFrame:
    rows = []
    for season in seasons:
        url = f"{base_url}{season}%20and%20gameTypeId=2"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            rows.extend(resp.json().get("data", []))
        except Exception as e:
            print(f"  Warning: season {season} — {e}", file=sys.stderr)
    return pd.DataFrame(rows)


def _fetch_contracts() -> pd.DataFrame:
    if not PUCKPEDIA_API_KEY:
        raise RuntimeError("Set PUCKPEDIA_API_KEY env var before running")
    resp = requests.get(
        f"https://puckpedia.com/api/players2?api_key={PUCKPEDIA_API_KEY}",
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json().get("data", []))
    exploded = df.explode("current")
    exploded = pd.concat(
        [exploded.drop(columns=["current"]),
         exploded["current"].apply(pd.Series)],
        axis=1,
    )
    exploded = exploded.drop(columns=["future"], errors="ignore")
    return exploded[exploded["contract_id"].notna()].copy()


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _build_knn_features(raw_stats: pd.DataFrame, raw_bios: pd.DataFrame) -> pd.DataFrame:
    df = raw_stats.copy()
    df["july_1"] = pd.to_datetime(
        df["seasonId"].astype(str).str[4:8] + "-07-01"
    )
    df = df.sort_values(["playerId", "seasonId"]).reset_index(drop=True)
    df["szn_no"] = df.groupby("playerId").cumcount() + 1

    for col in ["gamesPlayed", "points", "evPoints"]:
        df[f"ctd_{col}"] = df.groupby("playerId")[col].cumsum()

    df["ctd_p_pg"] = (
        (df["ctd_points"] - df["points"])
        / (df["ctd_gamesPlayed"] - df["gamesPlayed"])
    ).fillna(0).round(4)

    df["ctd_ev_p_pg"] = (
        (df["ctd_evPoints"] - df["evPoints"])
        / (df["ctd_gamesPlayed"] - df["gamesPlayed"])
    ).fillna(0).round(4)

    df["ctd_toi_avg"] = (
        (df.groupby("playerId")["timeOnIcePerGame"].cumsum()
         - df["timeOnIcePerGame"])
        / (df["szn_no"] - 1)
    ).fillna(0).round(2)

    df["pct_gp"]          = (df["ctd_gamesPlayed"] / (df["szn_no"] * 82)).round(4)
    df["evPointsPerGame"] = (df["evPoints"] / df["gamesPlayed"]).fillna(0).round(4)

    for col in ["gamesPlayed", "points", "pointsPerGame",
                "evPoints", "evPointsPerGame", "timeOnIcePerGame"]:
        s1 = df.groupby("playerId")[col].shift(1).fillna(0)
        s2 = df.groupby("playerId")[col].shift(2).fillna(0)
        df[f"{col}_L3"] = ((df[col] + s1 + s2) / 3).round(4)

    bios_slim = (
        raw_bios[["playerId", "birthDate"]]
        .drop_duplicates("playerId")
        .copy()
    )
    bios_slim["birthDate"] = pd.to_datetime(bios_slim["birthDate"], errors="coerce")
    df = df.merge(bios_slim, on="playerId", how="left")
    df["july_1_age"] = (
        (df["july_1"] - df["birthDate"]).dt.days / 365.25
    ).fillna(0).astype(int)

    df = df[df["positionCode"] != "G"].copy()
    df["position_group"] = np.where(df["positionCode"] == "D", "Defense", "Forward")
    return df


def _build_aging_features(raw_stats: pd.DataFrame, raw_bios: pd.DataFrame) -> pd.DataFrame:
    bios_slim = (
        raw_bios[["playerId", "birthDate"]]
        .drop_duplicates("playerId")
        .copy()
    )
    bios_slim["birthDate"] = pd.to_datetime(bios_slim["birthDate"], errors="coerce")
    df = raw_stats.merge(bios_slim, on="playerId", how="left")
    df["season_start"] = pd.to_datetime(
        df["seasonId"].astype(str).str[:4] + "-10-01"
    )
    df["age"] = ((df["season_start"] - df["birthDate"]).dt.days / 365.25).round(2)
    df = df[df["positionCode"] != "G"].copy()
    df = df[df["gamesPlayed"] >= MIN_GP]
    df = df[df["age"].between(18, 42)]
    df["position_group"] = np.where(df["positionCode"] == "D", "Defense", "Forward")
    df["toi_per_game"]   = (df["timeOnIcePerGame"] / 60).round(3)
    df["ppg"]            = df["pointsPerGame"].round(4)
    return df[[
        "playerId", "skaterFullName", "seasonId", "position_group",
        "age", "gamesPlayed", "ppg", "toi_per_game",
    ]].copy()


def _build_contracts_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["contract_end_year"] = (
        df["contract_end"].astype(str).str.split("-").str[0].astype(int)
    )
    df["contract_start_year"] = df["contract_end_year"] - df["length"].astype(int)
    df["contract_year"] = (
        df["contract_start_year"] * 10000 + df["contract_start_year"] + 1
    ).astype(int)
    curr  = df["contract_end"] == f"{CURRENT_YEAR}-{CURRENT_YEAR + 1}"
    curr2 = df["contract_year"] == CURRENT_SEASON
    is_curr = curr | curr2
    df.loc[is_curr, "signing_status"] = df.loc[is_curr, "expiry_status"]
    df.loc[is_curr, "contract_year"]  = CURRENT_SEASON
    df["nhl_id"] = df["nhl_id"].astype(int)
    return df[df["signing_status"] != False].copy()  # noqa: E712


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC — STEP 0: FETCH AND BUILD DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_build_data() -> dict:
    """
    Pull NHL stats + bios + PuckPedia contracts; return ready-to-use datasets.
    Keys: knn_merged, aging_data, knn_stats.
    """
    print(f"Fetching {len(_KNN_SEASONS)} seasons of NHL data...")
    raw_stats = _fetch_nhl(_NHL_STATS_URL, _KNN_SEASONS)
    raw_bios  = _fetch_nhl(_NHL_BIOS_URL,  _KNN_SEASONS)
    raw_bios  = raw_bios.drop(
        columns=["assists", "goals", "gamesPlayed", "points",
                 "isInHallOfFameYn", "birthCity", "birthStateProvinceCode"],
        errors="ignore",
    ).drop_duplicates("playerId")
    print(f"  {len(raw_stats):,} stat rows | {len(raw_bios):,} bio rows")

    print("Building aging feature set (2007-08 onward)...")
    aging_raw  = raw_stats[raw_stats["seasonId"] >= _AGING_START].copy()
    aging_data = _build_aging_features(aging_raw, raw_bios)
    print(f"  {len(aging_data):,} player-seasons for aging model")

    print("Building KNN feature set (all seasons)...")
    knn_stats = _build_knn_features(raw_stats, raw_bios)
    print(f"  {len(knn_stats):,} stat rows with KNN features")

    print("Fetching contracts from PuckPedia...")
    raw_contracts = _fetch_contracts()
    contracts     = _build_contracts_df(raw_contracts)
    print(f"  {len(contracts):,} contract rows")

    merged = pd.merge(
        contracts, knn_stats,
        how="inner",
        left_on=["nhl_id", "contract_year"],
        right_on=["playerId", "seasonId"],
    )
    merged = merged[merged["positionCode"] != "G"].copy()
    merged["position_group"] = np.where(
        merged["positionCode"] == "D", "Defense", "Forward"
    )
    print(f"  Merged to {len(merged):,} contract player-seasons")

    return {"knn_merged": merged, "aging_data": aging_data, "knn_stats": knn_stats}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — KNN (reconstructed identically to generate_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def _idw_estimate(cap_hit_pcts: list, distances: list) -> dict:
    vals  = np.array(cap_hit_pcts, dtype=float)
    dists = np.array(distances,    dtype=float)
    raw_w = 1.0 / (dists + IDW_EPSILON)
    norm_w = raw_w / raw_w.sum()
    estimate = float(np.dot(norm_w, vals))
    var_num  = np.dot(norm_w, (vals - estimate) ** 2)
    var_den  = 1.0 - np.dot(norm_w, norm_w)
    w_std = float(np.sqrt(var_num / var_den)) if var_den > 0 else float(np.std(vals))
    return {
        "estimate": round(estimate, 4),
        "weights":  norm_w.tolist(),
        "ci_low":   round(max(0.0, estimate - w_std), 4),
        "ci_high":  round(estimate + w_std, 4),
        "std":      round(w_std, 4),
    }


def build_knn_models(knn_merged: pd.DataFrame, max_season: int | None = None) -> dict:
    """
    Fit 4 KNN models (RFA/UFA × Forward/Defense).
    max_season: if set, only use historical FAs up to this season ID (for holdout).
    Returns dict keyed by 'RFA_Forward', 'UFA_Defense', etc.
    """
    effective_current = max_season if max_season is not None else CURRENT_SEASON
    min_season = effective_current - (LOOKBACK_SEASONS * 10001)
    models = {}

    for fa_status in ["RFA", "UFA"]:
        for pos_group in ["Forward", "Defense"]:
            weights  = UFA_WEIGHTS if fa_status == "UFA" else RFA_WEIGHTS
            features = [f for f in ALL_KNN_FEATURES if f in knn_merged.columns]

            subset = knn_merged[
                (knn_merged["signing_status"] == fa_status)
                & (knn_merged["position_group"] == pos_group)
            ].copy()

            historical = subset[
                (subset["contract_year"] != effective_current)
                & (subset["contract_year"] >= min_season)
                & (subset["contract_year"] < effective_current)
            ].copy()

            if len(historical) < KNN_NEIGHBORS:
                print(f"  Warning: {fa_status} {pos_group} — only {len(historical)} rows, skipping")
                continue

            scaler = StandardScaler()
            X_hist = pd.DataFrame(
                scaler.fit_transform(historical[features]),
                columns=features, index=historical.index,
            )
            if weights:
                for feat, w in weights.items():
                    if feat in X_hist.columns:
                        X_hist[feat] *= w

            knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, algorithm="auto")
            knn.fit(X_hist)

            key = f"{fa_status}_{pos_group}"
            models[key] = {
                "scaler":     scaler,
                "knn":        knn,
                "historical": historical,
                "features":   features,
                "weights":    weights,
            }
            print(f"  KNN fitted: {key} ({len(historical)} historical FAs)")

    return models


def _query_knn(feature_vector: pd.Series, model: dict) -> dict:
    """Run one feature vector through a fitted KNN model, return IDW cap-hit estimate."""
    features   = model["features"]
    weights    = model["weights"]
    historical = model["historical"]

    X_q = pd.DataFrame(
        [feature_vector.reindex(features).fillna(0).values],
        columns=features,
    )
    X_scaled = pd.DataFrame(
        model["scaler"].transform(X_q), columns=features,
    )
    if weights:
        for feat, w in weights.items():
            if feat in X_scaled.columns:
                X_scaled[feat] *= w

    k = min(KNN_NEIGHBORS, len(historical))
    dists, idxs = model["knn"].kneighbors(X_scaled, n_neighbors=k)
    dists = dists[0]
    idxs  = idxs[0]

    comp_pcts = []
    for idx in idxs:
        row    = historical.iloc[idx]
        cy     = int(row["contract_year"])
        ul     = CAP_LIMITS.get(cy, 95_500_000)
        length = int(row.get("length", 1)) if pd.notna(row.get("length")) else 1
        value  = int(row.get("value",  0)) if pd.notna(row.get("value"))  else 0
        aav    = value / length if length > 0 else 0
        comp_pcts.append((aav / ul) * 100 if ul > 0 else 0)

    return _idw_estimate(comp_pcts, list(dists))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BAYESIAN HIERARCHICAL AGING MODEL
# ─────────────────────────────────────────────────────────────────────────────

def fit_aging_model(data: pd.DataFrame, position_group: str) -> dict:
    """
    Fit a PyMC population-level quadratic aging curve for the given position group.
    Uses NUTS on 4 parameters (fast even without a C compiler).
    Caches the trace to disk as a .netcdf file; reloads on subsequent runs.
    Returns dict: {trace, position_group, age_center, metric}.
    """
    metric     = PRIMARY_METRIC[position_group]
    age_center = AGE_CENTER[position_group]

    sub = data[data["position_group"] == position_group].dropna(subset=[metric, "age"])
    sub = sub[sub[metric] > 0].copy()
    print(f"  {position_group}: {len(sub):,} player-seasons | "
          f"{sub['playerId'].nunique():,} players | metric={metric}")

    age_c  = (sub["age"].values - age_center).astype(float)
    y      = sub[metric].values.astype(float)

    y_mean = float(np.mean(y))
    y_std  = float(np.std(y))

    gamma_mu    = -0.3 * y_mean / 100.0
    gamma_sigma = abs(gamma_mu) * 0.5

    cache_path = OUTPUT_DIR / f"aging_trace_{position_group.lower()}.netcdf"
    if cache_path.exists():
        print(f"  Loading cached trace from {cache_path.name}...")
        trace = az.from_netcdf(str(cache_path))
        print("  Cache loaded.")
        return {
            "trace":          trace,
            "position_group": position_group,
            "age_center":     age_center,
            "metric":         metric,
        }

    print(f"  Fitting population quadratic curve (n_obs={len(y)})...")
    print(f"  gamma prior: Normal({gamma_mu:.5f}, {gamma_sigma:.5f})")

    with pm.Model():
        mu_alpha  = pm.Normal("mu_alpha", mu=y_mean,   sigma=y_std * 2)
        mu_beta   = pm.Normal("mu_beta",  mu=0.0,      sigma=y_std * 0.05)
        gamma     = pm.Normal("gamma",    mu=gamma_mu, sigma=gamma_sigma)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=y_std * 0.5)

        mu_obs = mu_alpha + mu_beta * age_c + gamma * age_c ** 2
        pm.Normal("y_obs", mu=mu_obs, sigma=sigma_obs, observed=y)

        print("  Sampling (1 000 draws × 2 chains)...")
        trace = pm.sample(
            draws=1_000, tune=500, chains=2, target_accept=0.9,
            progressbar=True, return_inferencedata=True,
        )

    print(f"  Saving trace to {cache_path.name}...")
    trace.to_netcdf(str(cache_path))

    return {
        "trace":          trace,
        "position_group": position_group,
        "age_center":     age_center,
        "metric":         metric,
    }


def get_retention_rates(
    fitted: dict,
    current_age: float,
    position_group: str,
) -> dict:
    """
    Return year-by-year retention rates (years 1–8) from the population-level curve.
    Keys: 'best' (75th pct), 'base' (50th), 'downside' (25th).
    Each value is a list of 8 floats.
    """
    trace      = fitted["trace"]
    age_center = fitted["age_center"]

    mu_alpha = trace.posterior["mu_alpha"].values.flatten()
    mu_beta  = trace.posterior["mu_beta"].values.flatten()
    gamma    = trace.posterior["gamma"].values.flatten()

    def _curve(age: float) -> np.ndarray:
        ac = age - age_center
        return mu_alpha + mu_beta * ac + gamma * ac ** 2

    current_prod = _curve(current_age)
    current_prod = np.where(np.abs(current_prod) < 1e-6, 1e-6, current_prod)

    best, base, downside = [], [], []
    for yr in range(1, 9):
        future_prod    = _curve(current_age + yr)
        retention      = np.clip(future_prod / current_prod, 0.05, 1.50)
        best.append(    round(float(np.percentile(retention, 75)), 4))
        base.append(    round(float(np.percentile(retention, 50)), 4))
        downside.append(round(float(np.percentile(retention, 25)), 4))

    return {"best": best, "base": base, "downside": downside}


# ─────────────────────────────────────────────────────────────────────────────
# CAP PROJECTION — Monte Carlo bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def compute_growth_rates(cap_dict: dict, exclude_years: set | None = None) -> list:
    """Year-over-year growth rates from a cap dict, skipping non-consecutive gaps."""
    years = sorted(cap_dict.keys())
    rates = []
    for i in range(1, len(years)):
        y_prev, y_curr = years[i - 1], years[i]
        if exclude_years and y_curr in exclude_years:
            continue
        if y_curr - y_prev == 1:
            rates.append((cap_dict[y_curr] / cap_dict[y_prev]) - 1)
    return rates


def run_cap_monte_carlo(
    current_cap: float = 95_500_000,
    anchor_years: dict | None = None,
    project_from_year: int = 2027,
    n_seasons: int = 5,
    n_simulations: int = 10_000,
    regime_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Bootstrap cap growth rates from history, anchor to known future values,
    then project probabilistically beyond them.
    Returns DataFrame indexed by calendar year with p10/p25/p50/p75/p90/mean columns.
    """
    if anchor_years is None:
        anchor_years = _KNOWN_FUTURE_CAPS

    _EXCLUDE = {2012, 2020, 2021}
    all_rates    = compute_growth_rates(_CAP_HISTORY_FOR_MC, exclude_years=_EXCLUDE)
    recent_caps  = {k: v for k, v in _CAP_HISTORY_FOR_MC.items() if k >= 2022}
    recent_rates = compute_growth_rates(recent_caps)

    blended_pool = (
        recent_rates * int(len(all_rates) * regime_weight)
        + all_rates  * int(len(all_rates) * (1 - regime_weight))
    )

    results = []
    for _ in range(n_simulations):
        sim = {}
        cap = current_cap
        for year, known_cap in sorted(anchor_years.items()):
            sim[year] = known_cap
            cap = known_cap
        for offset in range(1, n_seasons + 1):
            proj_year = project_from_year + offset
            growth = max(float(np.random.choice(blended_pool)), 0.02)
            cap = cap * (1 + growth)
            sim[proj_year] = cap
        results.append(sim)

    df = pd.DataFrame(results)
    summary = {}
    for col in df.columns:
        summary[col] = {
            "p10":  df[col].quantile(0.10),
            "p25":  df[col].quantile(0.25),
            "p50":  df[col].quantile(0.50),
            "p75":  df[col].quantile(0.75),
            "p90":  df[col].quantile(0.90),
            "mean": df[col].mean(),
        }
    return pd.DataFrame(summary).T


def build_cap_projections(
    n_seasons: int = 8,
    n_simulations: int = 10_000,
    regime_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Run Monte Carlo cap projection and update the module-level CAP_LIMITS
    and FUTURE_CAPS_PROJECTED dicts with the results.
    Returns the full percentile summary DataFrame.
    """
    global _CAP_PROJECTIONS, CAP_LIMITS, FUTURE_CAPS_PROJECTED

    print("  Running cap Monte Carlo "
          f"({n_simulations:,} simulations, {n_seasons} seasons beyond anchors)...")

    proj = run_cap_monte_carlo(
        current_cap      = _KNOWN_FUTURE_CAPS[min(_KNOWN_FUTURE_CAPS)],
        anchor_years     = _KNOWN_FUTURE_CAPS,
        project_from_year= max(_KNOWN_FUTURE_CAPS),
        n_seasons        = n_seasons,
        n_simulations    = n_simulations,
        regime_weight    = regime_weight,
    )
    _CAP_PROJECTIONS = proj

    # Backfill CAP_LIMITS with p50 for projected years and mark them
    projected_season_ids = set()
    for cal_year, row in proj.iterrows():
        if cal_year in _KNOWN_FUTURE_CAPS:
            continue  # anchor — already in CAP_LIMITS exactly
        season_id = int(cal_year) * 10000 + int(cal_year) + 1
        p50_cap   = int(round(row["p50"]))
        CAP_LIMITS[season_id] = p50_cap
        projected_season_ids.add(season_id)

    FUTURE_CAPS_PROJECTED = projected_season_ids

    print(f"  Cap projections (p50) beyond {max(_KNOWN_FUTURE_CAPS)}-{max(_KNOWN_FUTURE_CAPS)+1}:")
    for cal_year, row in proj.iterrows():
        if cal_year in _KNOWN_FUTURE_CAPS:
            continue
        sid = int(cal_year) * 10000 + int(cal_year) + 1
        print(f"    {sid}: ${int(row['p50'])/1e6:.1f}M  "
              f"[p25=${int(row['p25'])/1e6:.1f}M – p75=${int(row['p75'])/1e6:.1f}M]")

    return proj


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CONTRACT VALUE PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def _cap_for_year(year: int) -> dict:
    """
    Cap info for contract year N (1-indexed, relative to CURRENT_SEASON).
    Returns dict: {cap, cap_p25, cap_p75, is_projected}.
    For known years, p25 == cap == p75. For MC-projected years, bands diverge.
    """
    target   = CURRENT_SEASON + (year - 1) * 10001
    cal_year = target // 10000

    if target in CAP_LIMITS and target not in FUTURE_CAPS_PROJECTED:
        cap = CAP_LIMITS[target]
        return {"cap": cap, "cap_p25": cap, "cap_p75": cap, "is_projected": False}

    if _CAP_PROJECTIONS is not None and cal_year in _CAP_PROJECTIONS.index:
        row = _CAP_PROJECTIONS.loc[cal_year]
        return {
            "cap":          int(round(row["p50"])),
            "cap_p25":      int(round(row["p25"])),
            "cap_p75":      int(round(row["p75"])),
            "is_projected": True,
        }

    # Fallback before projections are built
    last = max(s for s in CAP_LIMITS)
    cap  = CAP_LIMITS[last]
    return {"cap": cap, "cap_p25": cap, "cap_p75": cap, "is_projected": True}


def _apply_retention(
    feature_vector: pd.Series,
    retention: float,
    current_age: float,
    year: int,
) -> pd.Series:
    fv = feature_vector.copy().astype(float)
    fv["july_1_age"] = current_age + year

    for feat in _L3_SET:
        if feat in fv.index:
            fv[feat] = fv[feat] * retention

    for feat in _PROD_FEATURES:
        if feat in fv.index:
            fv[feat] = fv[feat] * retention

    for feat in _SLOW_FEATURES:
        if feat in fv.index:
            partial = 1.0 + (retention - 1.0) * 0.5
            fv[feat] = fv[feat] * partial

    return fv


def _verdict(contracted_pct: float, projected_pct: float) -> str:
    diff = projected_pct - contracted_pct
    if diff > 1.0:
        return "Bargain"
    elif diff > -0.5:
        return "Fair"
    elif diff > -1.5:
        return "Watch"
    return "Risk"


def project_contract_value(
    player_id: int,
    contract_years: int,
    contracted_aav: float,
    fa_status: str,
    position_group: str,
    knn_models: dict,
    fitted_aging: dict,
    knn_stats: pd.DataFrame,
) -> list:
    """
    Project cap hit % and AAV for each year of a contract.

    Returns list of dicts, one per contract year:
      year, age, contracted_cap_pct, projected_cap_pct_{best,base,downside},
      projected_aav_{best,base,downside}, verdict, cap_ceiling, cap_is_projected
    """
    model_key = f"{fa_status}_{position_group}"
    knn_model = knn_models.get(model_key)
    if knn_model is None:
        raise ValueError(f"No KNN model found for key '{model_key}'")

    player_rows = knn_stats[knn_stats["playerId"] == player_id]
    if player_rows.empty:
        raise ValueError(f"Player {player_id} not found in knn_stats")
    player_row  = player_rows.sort_values("seasonId").iloc[-1]
    current_age = float(player_row["july_1_age"])
    features    = knn_model["features"]

    ag = fitted_aging.get(position_group)
    if ag is None:
        raise ValueError(f"No fitted aging model for position_group='{position_group}'")

    print(f"  Getting retention rates for player {player_id} "
          f"(age {current_age:.0f}, {position_group})...")
    retention = get_retention_rates(ag, current_age, position_group)

    results = []
    for yr in range(1, contract_years + 1):
        cap_info  = _cap_for_year(yr)
        cap       = cap_info["cap"]
        cap_p25   = cap_info["cap_p25"]
        cap_p75   = cap_info["cap_p75"]
        is_proj   = cap_info["is_projected"]

        # Contracted cap % at each cap scenario (AAV fixed, cap varies)
        contracted_pct      = round((contracted_aav / cap)     * 100, 4)
        contracted_pct_bear = round((contracted_aav / cap_p25) * 100, 4)  # low cap = worse %
        contracted_pct_bull = round((contracted_aav / cap_p75) * 100, 4)  # high cap = better %

        row = {
            "year":                    yr,
            "age":                     round(current_age + yr, 1),
            "contracted_cap_pct":      contracted_pct,
            "contracted_cap_pct_bear": contracted_pct_bear,
            "contracted_cap_pct_bull": contracted_pct_bull,
            "cap_ceiling":             cap,
            "cap_p25":                 cap_p25,
            "cap_p75":                 cap_p75,
            "cap_is_projected":        is_proj,
        }

        for scenario in ["best", "base", "downside"]:
            ret = retention[scenario][yr - 1]
            fv  = _apply_retention(
                player_row[features], ret, current_age, yr
            )
            est      = _query_knn(fv, knn_model)
            proj_pct = est["estimate"]
            row[f"projected_cap_pct_{scenario}"] = proj_pct
            row[f"projected_aav_{scenario}"]     = round(proj_pct * cap / 100)

        row["verdict"] = _verdict(contracted_pct, row["projected_cap_pct_base"])
        results.append(row)
        print(f"    Year {yr} (age {row['age']}): contracted={contracted_pct:.2f}% | "
              f"base={row['projected_cap_pct_base']:.2f}% | {row['verdict']}"
              + (f" [cap MC: ${cap/1e6:.1f}M p50, ${cap_p25/1e6:.1f}M–${cap_p75/1e6:.1f}M]"
                 if is_proj else ""))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — demo run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NHL Aging Model — Full Pipeline")
    print("=" * 60)

    data       = fetch_and_build_data()
    knn_merged = data["knn_merged"]
    aging_data = data["aging_data"]
    knn_stats  = data["knn_stats"]

    print("\nBuilding cap projections...")
    build_cap_projections()

    print("\nBuilding KNN models...")
    knn_models = build_knn_models(knn_merged)

    print("\nFitting Bayesian aging models...")
    fitted_aging = {}
    for pos in ["Forward", "Defense"]:
        print(f"\n--- {pos} aging model ---")
        fitted_aging[pos] = fit_aging_model(aging_data, pos)

    # Demo: pick the highest-cap-pct UFA forward with a multi-year contract
    demo_subset = knn_merged[
        (knn_merged["signing_status"] == "UFA")
        & (knn_merged["position_group"] == "Forward")
        & (knn_merged["length"].fillna(0).astype(int) >= 3)
        & (knn_merged["contract_year"] >= CURRENT_SEASON - 10001)
    ].copy()
    demo_subset["_ul"] = demo_subset["contract_year"].map(CAP_LIMITS).fillna(95_500_000)
    demo_subset["_length"] = demo_subset["length"].fillna(1).astype(int).clip(lower=1)
    demo_subset["_value"]  = demo_subset["value"].fillna(0).astype(float)
    demo_subset["_aav"]    = demo_subset["_value"] / demo_subset["_length"]
    demo_subset["_pct"]    = demo_subset["_aav"] / demo_subset["_ul"] * 100

    if not demo_subset.empty:
        demo_row = demo_subset.sort_values("_pct", ascending=False).iloc[0]
        demo_id    = int(demo_row["playerId"])
        demo_name  = str(demo_row.get("skaterFullName", demo_id))
        demo_aav   = float(demo_row["_aav"])
        demo_years = int(demo_row["_length"])
        demo_fa    = str(demo_row["signing_status"])
        demo_pos   = str(demo_row["position_group"])

        print(f"\n--- Demo: {demo_name} | {demo_fa} {demo_pos} | "
              f"AAV ${demo_aav:,.0f} × {demo_years}yr ---")
        try:
            projection = project_contract_value(
                player_id       = demo_id,
                contract_years  = demo_years,
                contracted_aav  = demo_aav,
                fa_status       = demo_fa,
                position_group  = demo_pos,
                knn_models      = knn_models,
                fitted_aging    = fitted_aging,
                knn_stats       = knn_stats,
            )
            print("\n  Year | Age  | Contracted% | Base%  | Best%  | Down%  | Verdict")
            print("  " + "-" * 65)
            for r in projection:
                print(f"  {r['year']:>4} | {r['age']:>4} | "
                      f"{r['contracted_cap_pct']:>10.2f}% | "
                      f"{r['projected_cap_pct_base']:>5.2f}% | "
                      f"{r['projected_cap_pct_best']:>5.2f}% | "
                      f"{r['projected_cap_pct_downside']:>5.2f}% | "
                      f"{r['verdict']}")
        except Exception as e:
            print(f"  Demo failed: {e}")
    else:
        print("\nNo suitable demo player found in merged data.")

    print("\nDone.")

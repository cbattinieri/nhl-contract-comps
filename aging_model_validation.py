# pip install pymc arviz numpy pandas scikit-learn matplotlib requests pyarrow
# Requires: PUCKPEDIA_API_KEY env var and aging_model.py in the same directory.

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from aging_model import (
    fetch_and_build_data,
    build_cap_projections,
    build_knn_models,
    fit_aging_model,
    get_retention_rates,
    project_contract_value,
    CAP_LIMITS,
    PRIMARY_METRIC,
    AGE_CENTER,
    CURRENT_SEASON,
    IDW_EPSILON,
    KNN_NEIGHBORS,
    ALL_KNN_FEATURES,
    UFA_WEIGHTS,
    _idw_estimate,
    _query_knn,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path(__file__).parent

HOLDOUT_CUTOFF = 20222023   # train on < this season, test on >= this season


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Population aging curves with posterior uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def plot_population_curves(
    fitted_fwd: dict,
    fitted_def: dict,
    aging_data: pd.DataFrame,
) -> None:
    """Overlay posterior population curves on binned empirical data."""
    print("Plotting population aging curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    configs = [
        (axes[0], fitted_fwd, "Forward",  "ppg",         "Points Per Game"),
        (axes[1], fitted_def, "Defense",  "toi_per_game", "TOI Per Game (min)"),
    ]

    for ax, fitted, pos, metric, ylabel in configs:
        trace      = fitted["trace"]
        age_center = fitted["age_center"]

        mu_alpha = trace.posterior["mu_alpha"].values.flatten()
        mu_beta  = trace.posterior["mu_beta"].values.flatten()
        gamma    = trace.posterior["gamma"].values.flatten()

        ages    = np.linspace(18, 42, 200)
        age_c   = ages - age_center
        curves  = (
            mu_alpha[:, None]
            + mu_beta[:, None] * age_c[None, :]
            + gamma[:, None]   * age_c[None, :] ** 2
        )

        mean_curve = np.mean(curves, axis=0)
        lo50       = np.percentile(curves, 25, axis=0)
        hi50       = np.percentile(curves, 75, axis=0)
        lo90       = np.percentile(curves, 5,  axis=0)
        hi90       = np.percentile(curves, 95, axis=0)

        # Empirical binned data
        sub = aging_data[aging_data["position_group"] == pos].copy()
        sub["age_bin"] = sub["age"].round()
        agg = (
            sub.groupby("age_bin")[metric]
            .agg(["mean", "median", "count"])
        )
        agg = agg[agg["count"] >= 30]

        ax.fill_between(ages, lo90, hi90, alpha=0.15, color="steelblue", label="90% credible")
        ax.fill_between(ages, lo50, hi50, alpha=0.30, color="steelblue", label="50% credible")
        ax.plot(ages, mean_curve, color="steelblue", lw=2.0, label="Posterior mean")
        ax.scatter(agg.index, agg["mean"],   s=30, color="black",  zorder=5, label="Empirical mean")
        ax.scatter(agg.index, agg["median"], s=20, color="dimgray", zorder=5,
                   marker="s", alpha=0.7, label="Empirical median")

        ax.axvline(age_center, ls="--", color="salmon", lw=1.2,
                   label=f"Age center ({age_center})")
        ax.set_title(f"{pos} — {ylabel}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Bayesian Population Aging Curves — Posterior vs Empirical",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "validation_population_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved to {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# HOLDOUT TEST — train pre-2022, test 2022-2024
# ─────────────────────────────────────────────────────────────────────────────

def run_holdout_test(knn_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Train KNN on FA classes before HOLDOUT_CUTOFF; predict contracts signed
    from HOLDOUT_CUTOFF through the season before the current one.
    Reports MAE vs actual cap hit %.
    """
    print(f"\nRunning holdout test (train < {HOLDOUT_CUTOFF}, test >= {HOLDOUT_CUTOFF})...")

    train = knn_merged[knn_merged["contract_year"] < HOLDOUT_CUTOFF].copy()
    test  = knn_merged[
        (knn_merged["contract_year"] >= HOLDOUT_CUTOFF)
        & (knn_merged["contract_year"] < CURRENT_SEASON)
    ].copy()

    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")

    all_errors = []
    for fa_status in ["RFA", "UFA"]:
        for pos_group in ["Forward", "Defense"]:
            weights  = UFA_WEIGHTS if fa_status == "UFA" else None
            features = [f for f in ALL_KNN_FEATURES if f in knn_merged.columns]

            tr = train[
                (train["signing_status"] == fa_status)
                & (train["position_group"] == pos_group)
            ].copy()
            te = test[
                (test["signing_status"] == fa_status)
                & (test["position_group"] == pos_group)
            ].copy()

            if len(tr) < KNN_NEIGHBORS or te.empty:
                print(f"  {fa_status} {pos_group}: insufficient data, skipping")
                continue

            scaler = StandardScaler()
            X_tr = pd.DataFrame(
                scaler.fit_transform(tr[features]),
                columns=features, index=tr.index,
            )
            X_te = pd.DataFrame(
                scaler.transform(te[features]),
                columns=features, index=te.index,
            )
            if weights:
                for feat, w in weights.items():
                    if feat in X_tr.columns:
                        X_tr[feat] *= w
                        X_te[feat] *= w

            k   = min(KNN_NEIGHBORS, len(tr))
            knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
            knn.fit(X_tr)
            dists, idxs = knn.kneighbors(X_te)

            for i in range(len(te)):
                comp_pcts = []
                for idx in idxs[i]:
                    comp_row = tr.iloc[idx]
                    cy       = int(comp_row["contract_year"])
                    ul       = CAP_LIMITS.get(cy, 95_500_000)
                    length   = int(comp_row.get("length", 1)) if pd.notna(comp_row.get("length")) else 1
                    value    = int(comp_row.get("value",  0)) if pd.notna(comp_row.get("value"))  else 0
                    aav      = value / length if length > 0 else 0
                    comp_pcts.append((aav / ul) * 100 if ul > 0 else 0)

                est    = _idw_estimate(comp_pcts, list(dists[i]))
                te_row = te.iloc[i]
                acy    = int(te_row["contract_year"])
                aul    = CAP_LIMITS.get(acy, 95_500_000)
                al     = int(te_row.get("length", 1)) if pd.notna(te_row.get("length")) else 1
                av     = int(te_row.get("value",  0)) if pd.notna(te_row.get("value"))  else 0
                actual_aav = av / al if al > 0 else 0
                actual_pct = (actual_aav / aul) * 100 if aul > 0 else 0

                all_errors.append({
                    "playerId":   int(te_row["playerId"]),
                    "name":       str(te_row.get("skaterFullName", "")),
                    "season":     acy,
                    "fa_status":  fa_status,
                    "pos_group":  pos_group,
                    "predicted":  round(est["estimate"], 2),
                    "actual":     round(actual_pct, 2),
                    "error":      round(est["estimate"] - actual_pct, 2),
                    "abs_error":  round(abs(est["estimate"] - actual_pct), 2),
                })

    if not all_errors:
        print("  No errors computed — check data.")
        return pd.DataFrame()

    errors_df = pd.DataFrame(all_errors)
    mae    = errors_df["abs_error"].mean()
    median = errors_df["abs_error"].median()
    print(f"\n  Holdout results (n={len(errors_df)}):")
    print(f"    MAE        : {mae:.3f}%")
    print(f"    Median AE  : {median:.3f}%")

    for fa in ["RFA", "UFA"]:
        for pos in ["Forward", "Defense"]:
            sub = errors_df[(errors_df["fa_status"] == fa) & (errors_df["pos_group"] == pos)]
            if not sub.empty:
                print(f"    {fa} {pos}: MAE={sub['abs_error'].mean():.3f}% (n={len(sub)})")

    return errors_df


def plot_holdout_results(errors_df: pd.DataFrame) -> None:
    if errors_df.empty:
        return
    print("Plotting holdout scatter...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    i = 0
    for fa in ["RFA", "UFA"]:
        for pos in ["Forward", "Defense"]:
            sub = errors_df[(errors_df["fa_status"] == fa) & (errors_df["pos_group"] == pos)]
            ax  = axes[i]; i += 1
            if sub.empty:
                ax.set_visible(False)
                continue
            lo = min(sub["actual"].min(), sub["predicted"].min()) - 0.5
            hi = max(sub["actual"].max(), sub["predicted"].max()) + 0.5
            ax.scatter(sub["actual"], sub["predicted"], alpha=0.5, s=20, color="steelblue")
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Perfect")
            mae = sub["abs_error"].mean()
            ax.set_title(f"{fa} {pos} | MAE={mae:.2f}%", fontsize=10, fontweight="bold")
            ax.set_xlabel("Actual cap hit %")
            ax.set_ylabel("Predicted cap hit %")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.suptitle(f"Holdout Test: KNN Predictions vs Actuals (train <{HOLDOUT_CUTOFF})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "validation_holdout_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved to {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PLAYER CURVES
# ─────────────────────────────────────────────────────────────────────────────

def _select_example_players(knn_merged: pd.DataFrame) -> list:
    """
    Pick three players: star forward, depth forward, defenseman.
    All must have multi-year contracts in recent seasons.
    """
    recent = knn_merged[
        knn_merged["contract_year"] >= CURRENT_SEASON - 30003
    ].copy()
    recent["_ul"]     = recent["contract_year"].map(CAP_LIMITS).fillna(95_500_000)
    recent["_length"] = recent["length"].fillna(1).astype(int).clip(lower=1)
    recent["_value"]  = recent["value"].fillna(0).astype(float)
    recent["_aav"]    = recent["_value"] / recent["_length"]
    recent["_pct"]    = recent["_aav"] / recent["_ul"] * 100

    multi   = recent[recent["_length"] >= 3].copy()
    fwds    = multi[multi["position_group"] == "Forward"].sort_values("_pct", ascending=False)
    defs    = multi[multi["position_group"] == "Defense"].sort_values("_pct", ascending=False)

    picks = []
    if len(fwds) >= 1:
        picks.append(("Star Forward",  fwds.iloc[0]))
    if len(fwds) > 5:
        mid_idx = max(1, len(fwds) // 3)
        picks.append(("Depth Forward", fwds.iloc[mid_idx]))
    if len(defs) >= 1:
        picks.append(("Defenseman",    defs.iloc[0]))

    return picks


def plot_individual_curves(
    knn_merged: pd.DataFrame,
    knn_models: dict,
    fitted_aging: dict,
    knn_stats: pd.DataFrame,
) -> None:
    """
    For 3 selected players, plot year-by-year projected cap hit % (3 scenarios)
    alongside the contracted cap hit % as a reference line.
    """
    players = _select_example_players(knn_merged)
    if not players:
        print("  No suitable players found for individual curves.")
        return

    n = len(players)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    print(f"\nPlotting individual curves for {n} players...")

    for ax, (label, row) in zip(axes, players):
        pid         = int(row["playerId"])
        name        = str(row.get("skaterFullName", pid))
        fa_status   = str(row["signing_status"])
        pos_group   = str(row["position_group"])
        contracted_aav = float(row["_aav"])
        contract_years = int(row["_length"])

        print(f"  {label}: {name} | {fa_status} {pos_group} | "
              f"${contracted_aav:,.0f} × {contract_years}yr")

        model_key = f"{fa_status}_{pos_group}"
        if model_key not in knn_models:
            print(f"    No KNN model for {model_key}, skipping")
            ax.set_title(f"{label}\n{name}\n(no model)", fontsize=9)
            continue

        try:
            projection = project_contract_value(
                player_id      = pid,
                contract_years = min(contract_years, 8),
                contracted_aav = contracted_aav,
                fa_status      = fa_status,
                position_group = pos_group,
                knn_models     = knn_models,
                fitted_aging   = fitted_aging,
                knn_stats      = knn_stats,
            )
        except Exception as e:
            print(f"    Projection failed: {e}")
            ax.set_title(f"{label}\n{name}\n(error)", fontsize=9)
            continue

        years          = [r["year"]                    for r in projection]
        contracted_pct = [r["contracted_cap_pct"]      for r in projection]
        best_pct       = [r["projected_cap_pct_best"]  for r in projection]
        base_pct       = [r["projected_cap_pct_base"]  for r in projection]
        down_pct       = [r["projected_cap_pct_downside"] for r in projection]

        ax.fill_between(years, down_pct, best_pct,
                        alpha=0.20, color="steelblue", label="Best–Downside band")
        ax.fill_between(years, down_pct, base_pct,
                        alpha=0.25, color="steelblue")
        ax.plot(years, base_pct,       color="steelblue", lw=2.0, marker="o",
                ms=5, label="Projected (base)")
        ax.plot(years, best_pct,       color="steelblue", lw=1.0, ls="--",
                alpha=0.7, label="Projected (best)")
        ax.plot(years, down_pct,       color="steelblue", lw=1.0, ls=":",
                alpha=0.7, label="Projected (downside)")
        ax.plot(years, contracted_pct, color="crimson",   lw=2.0, marker="D",
                ms=4, label="Contracted (actual)")

        ages = [r["age"] for r in projection]
        ax.set_xticks(years)
        ax.set_xticklabels([f"Yr {y}\n(age {a:.0f})" for y, a in zip(years, ages)],
                           fontsize=8)
        ax.set_ylabel("Cap Hit %")
        ax.set_title(f"{label}\n{name}\n{fa_status} | ${contracted_aav/1e6:.1f}M AAV",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle("Individual Player Contract Value Projections", fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "validation_individual_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved to {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NHL Aging Model — Validation Suite")
    print("=" * 60)

    data       = fetch_and_build_data()
    knn_merged = data["knn_merged"]
    aging_data = data["aging_data"]
    knn_stats  = data["knn_stats"]

    print("\nBuilding cap projections...")
    build_cap_projections()

    # Fit aging models (loads from cache if available)
    print("\nFitting / loading aging models...")
    fitted_aging = {}
    for pos in ["Forward", "Defense"]:
        print(f"\n--- {pos} ---")
        fitted_aging[pos] = fit_aging_model(aging_data, pos)

    # Build full-data KNN for individual curve plots
    print("\nBuilding KNN models (full data)...")
    knn_models = build_knn_models(knn_merged)

    # ── Plot 1: population curves ─────────────────────────────────────────
    print("\n[1/3] Population aging curves...")
    plot_population_curves(fitted_aging["Forward"], fitted_aging["Defense"], aging_data)

    # ── Plot 2: holdout test ──────────────────────────────────────────────
    print("\n[2/3] Holdout test...")
    errors_df = run_holdout_test(knn_merged)
    plot_holdout_results(errors_df)

    # ── Plot 3: individual player curves ─────────────────────────────────
    print("\n[3/3] Individual player curves...")
    plot_individual_curves(knn_merged, knn_models, fitted_aging, knn_stats)

    print("\nValidation complete.")

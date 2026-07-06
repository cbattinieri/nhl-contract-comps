# Model Review — July 2026

Two questions were investigated against the 2026 free-agent class (186 actual
signings as a real-world holdout):

1. Should platform-year / recent-season production be weighted more heavily,
   given the first significant cap increase since pre-COVID?
2. Does a market-scarcity (local supply density) layer improve pricing?

All experiments ran offline against a clone; nothing here changed the live
pipeline. Results are recorded for the file.

---

## 1. Platform-year reweighting — supported, worth adopting

**Change tested:** add L2 (2-year) recent-production features to both RFA and UFA
models, add recent features to the RFA model (previously had none), and rebalance
weights so recent production competes with career-to-date stats instead of being
suppressed (old L3 weights of 0.25 were ~40–120× smaller than age/career weights).

**Sign convention:** error = actual − predicted. Positive = model came in low.

### Table A — leave-one-season-out backtest MAE (historical)

| Group | n | MAE now → new | Median AE now → new |
|---|---|---|---|
| RFA Forward | 205 | 2.32 → 1.89 ✅ | 1.26 → 1.13 |
| RFA Defense | 116 | 2.21 → 2.33 ⚠️ | 1.73 → 1.92 |
| UFA Forward | 391 | 2.39 → 2.01 ✅ | 1.91 → 1.50 |
| UFA Defense | 207 | 2.53 → 2.26 ✅ | 2.24 → 2.00 |

### Table B — 2026 real-world holdout (error = actual − predicted)

| Group | n | Bias now → new | MAE now → new |
|---|---|---|---|
| RFA Forward | 35 | +0.32 → +0.19 ✅ | 1.00 → 0.91 ✅ |
| RFA Defense | 25 | −0.51 → −0.37 ✅ | 1.05 → 0.97 ✅ |
| UFA Forward | 76 | −1.05 → −0.80 ✅ | 1.44 → 1.17 ✅ |
| UFA Defense | 38 | −1.13 → −0.81 ✅ | 1.70 → 1.42 ✅ |

**Read-out:** MAE improves in 7 of 8 cells; bias magnitude shrinks toward zero in
all four holdout groups. The lone regression (RFA Defense backtest) still improves
on the live holdout and is likely LOSO noise on a thin, survivorship-limited class.

**Caveats before shipping:**
- Re-tune RFA-Defense weights (only blemish).
- Add a games-played floor to recent features — small-sample breakouts get
  over-weighted (e.g. Simon Nemec moved the wrong way).

**Aggregate-bias nuance:** the class mean is a *net over-estimate*, driven by
decline-phase veterans who sign cheap despite strong career stats (Benn, Perry,
Zuccarello). The rising-cap "signs for more than predicted" effect is real but
concentrated in breakout/platform-year players, not the aggregate. The reweighting
recovers a fraction of those specific misses (Tuch ~46%, McMann ~29%), not all.

---

## 2. Market-scarcity layer — does NOT hold up on current data; do not ship

**Hypothesis:** players with few comparable alternatives on the board sign above
prediction. Thin market → positive residual.

**Tested two operationalizations** on the 2026 class, in the reweighted space:

- **Density in performance-feature space:** wrong sign, r(nearest-alt, resid) =
  −0.27 overall. The "most isolated" players were Ovechkin, Malkin, Burns, Perry,
  Carlson — isolated because their résumés are unique, not because their market was
  thin, and they sign cheap. The metric measured résumé uniqueness, not scarcity.
- **Density by projected value tier** (substitutes within ±1% cap hit of the KNN
  estimate — closer to the real mechanism): r = −0.15 overall, right sign but
  trivial, inconsistent across groups (UFA Forward +0.19, wrong sign), and the one
  respectable cell (UFA Defense −0.49) collapses to −0.15 once decline-phase vets
  are removed. Same confound.

**Root cause:** one clean class (see Data Constraint below). Split four ways →
cells of 25–76, mostly in the dense middle of the board. No statistical room to
detect an effect that only fires on a handful of isolated players per summer, and
the naive distance metric is confounded by résumé uniqueness and age decline —
both of which correlate with *lower* pay, flipping the sign.

**Recommendation:**
1. Do not ship a scarcity adjustment or a density-derived premium field.
2. Bank a daily board snapshot each offseason (already writing `fa_class_YYYY.json`);
   re-run these probes with proper LOSO across classes in 2–3 summers.
3. Real unlock: confirm whether a data source exposes *expired* historical
   contracts. That would give complete past classes and allow validation now.

---

## Data Constraint (Phase 0 finding) — read before extending the model

The KNN model's history is built entirely from PuckPedia's **currently-active**
contracts; expired deals are cleared. Past "classes" are reconstructed by
back-dating active contracts, so older classes retain only their still-running
(long-term, star-biased) deals. This is fine for a *similarity* metric but breaks
any *density* metric, which needs complete classes. Only 2026 has a genuine full
board snapshot in `fa_class_YYYY.json`. Usable near-complete classes: ~2022–2025.

---

## Scarcity module design (spec only — for when data supports it)

`scarcity_model.py`, mirroring `aging_model.py`: standalone, reuses
`build_knn_models`' fitted scaler + weighted feature space per (fa, pos), embeds
the **live unsigned pool**, and emits a labeled `marketDensity` block per player
(`poolSize`, `valueBandSubstitutes`, `neighborCounts` over a D-sweep anchored to
the group's median comp distance, plain-language `thin/normal/deep`) to a separate
`scarcity.json` — never folded into `estimatedCapHitPct`. Premium calibration
stays disabled until LOSO-across-classes earns it.

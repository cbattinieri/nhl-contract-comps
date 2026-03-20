# NHL Contract Comparables Tool

**Batt Analytics**

A KNN-based tool that finds the 5 most statistically similar free agents from the prior 3 signing classes to estimate contract values for pending NHL free agents. Built as a static GitHub Pages site — no server, no hosting costs.

**Live site:** https://cbattinieri.github.io/nhl-contract-comps/

## Architecture

```
generate_data.py              → Python pipeline: fetches data, runs KNN, outputs JSON
data/comps.json               → Pre-computed results (auto-updated daily via GitHub Actions)
index.html                    → Static frontend (loads JSON, no server needed)
.github/workflows/            → GitHub Action for daily data refresh + auto-deploy
feature_study.ipynb           → Jupyter notebook for model validation & feature analysis
test_local.py                 → Local preview server (not deployed)
```

Originally built as a Python Shiny app, rebuilt from scratch as a static site for reliability, zero hosting cost, and portability.

## How It Works

1. `generate_data.py` pulls 22 seasons of stats from the NHL API and contract data from PuckPedia
2. Four independent KNN models run (RFA Forwards, RFA Defense, UFA Forwards, UFA Defense)
3. For each pending free agent, the model finds the 5 most similar players from the prior 3 FA classes
4. Cap hit % is estimated using inverse-distance weighting with a confidence interval
5. Results are written to `data/comps.json` and the static frontend renders them

The GitHub Action runs this pipeline daily and commits updated results automatically.

## Model Details

### Features (validated via ablation study)

All four models use 12 features: 6 career-to-date + 6 L3 rolling averages.

| Feature | Category | Description |
|---------|----------|-------------|
| `july_1_age` | Career | Age on July 1 of the contract year |
| `ctd_gamesPlayed` | Career | Cumulative career games played |
| `ctd_p_pg` | Career | Career points per game (prior seasons) |
| `ctd_ev_p_pg` | Career | Career even-strength points per game (prior seasons) |
| `pct_gp` | Career | Career games played as % of possible (82/season) |
| `ctd_toi_avg` | Career | Career average TOI per game (prior seasons) |
| `gamesPlayed_L3` | L3 Avg | Average GP over last 3 seasons |
| `points_L3` | L3 Avg | Average points over last 3 seasons |
| `pointsPerGame_L3` | L3 Avg | Average P/G over last 3 seasons |
| `evPoints_L3` | L3 Avg | Average EV points over last 3 seasons |
| `evPointsPerGame_L3` | L3 Avg | Average EV P/G over last 3 seasons |
| `timeOnIcePerGame_L3` | L3 Avg | Average TOI/G over last 3 seasons |

**RFA models:** No feature weights. Career-to-date and recent trajectory treated equally.

**UFA models:** L3 features weighted at 2x. Recent production is slightly more predictive for UFAs since the open market prices current form more heavily.

### Why These Features?

A full ablation study (`feature_study.ipynb`) tested 264 configurations across all 4 segments, including: feature group isolation (CTD vs Platform vs L2 vs L3), drop-one importance analysis, platform year vs L2 vs L3 head-to-head comparisons, 13 weight schemes, K-neighbor sweeps (K=3 to K=15), and draft position as a feature.

Key findings:
- Adding L3 rolling averages to the CTD base was the single largest MAE improvement across every segment
- The original age-heavy weights (age 30x, career 10x) were consistently one of the worst performing configs — they forced age-matching over production-matching
- L2 and L3 averages perform similarly; Platform year is slightly noisier
- Draft position (overall pick and bucketed) showed no meaningful MAE improvement despite clear raw correlation with cap hit — career stats already capture the signal
- K=5 neighbors is stable across all segments; K=3 shows marginal improvement for RFA Forwards but with higher variance
- Kitchen-sink models (all 27 features) won some segments but inconsistently, suggesting overfitting risk with only ~100 backtest samples

### Estimation Method

Cap hit % estimates use **inverse-distance weighting (IDW)** rather than fixed weights. Each comp's influence is proportional to `1/distance` — so near-identical comps get nearly equal weight, while distant comps contribute minimally.

Each estimate includes:
- **Weighted mean** — the IDW cap hit % estimate
- **±1σ confidence interval** — weighted standard deviation of the comps' cap hit percentages
- **Comp weights** — the actual IDW weight each comparable received (shown in the app as visual bars)

### Backtest Accuracy

Each model is validated via leave-one-season-out cross-validation: every historical FA class is held out, predicted using only prior years, and the error is measured against actual signings.

After the ablation study optimizations:

| Model | Previous MAE | Current MAE | Improvement |
|-------|-------------|-------------|-------------|
| RFA Forwards | ~1.91% | ~1.33% | -0.58pp |
| RFA Defense | ~2.01% | ~1.83% | -0.18pp |
| UFA Forwards | ~1.99% | ~1.54% | -0.45pp |
| UFA Defense | ~2.17% | ~1.78% | -0.39pp |

MAE is in percentage points of cap hit. On the current $95.5M cap, 1.5% MAE ≈ ~$1.4M AAV.

## Setup (GitHub — no terminal needed)

1. Create a new repo on GitHub named `nhl-contract-comps`, check "Add a README"
2. Upload `generate_data.py`, `index.html`, `requirements.txt` at the root level
3. Create `data/comps.json` via Add File → Create New File (type `data/comps.json`, paste contents)
4. Create `.github/workflows/update_data.yml` via Add File → Create New File
5. Add your PuckPedia API key: **Settings → Secrets and variables → Actions → New repository secret** → name it `PUCKPEDIA_API_KEY`
6. Run the pipeline: **Actions → Update Data & Deploy → Run workflow**
7. Enable GitHub Pages: **Settings → Pages → Source → Deploy from branch → `main`**

Site goes live at `https://<username>.github.io/nhl-contract-comps/`

## Season Switch Checklist

Everything is dynamic except the salary cap numbers. Each offseason, update **one thing** in `generate_data.py`:

### 1. Update `CAP_LIMITS` (required)

Find the `CAP_LIMITS` dictionary near the top. Add the new season's confirmed cap and update projections:

```python
CAP_LIMITS = {
    ...
    20252026: 95_500_000,   # confirmed
    20262027: 104_000_000,  # update once confirmed (may be ~$107M)
    20272028: 113_500_000,  # update once confirmed
    20282029: ???_000_000,  # add when announced
}
```

**When:** After the NHL/NHLPA announces the official cap ceiling, typically January-June before the new season.

**Where to find it:** [NHL.com](https://www.nhl.com) or [PuckPedia](https://puckpedia.com)

### 2. Re-run the Action (required)

After updating the cap, go to **Actions → Update Data & Deploy → Run workflow**.

### That's it

Everything else is automatic:
- **Season detection** — computed from the current date (year ≥ July → new season)
- **Season list** — auto-extends from 2003-04 through the current season
- **July 1st age** — uses the end year of the season (when FA opens)
- **Free agent detection** — derives from PuckPedia contract expiry data
- **Daily refresh** — GitHub Action runs at 8 AM UTC

## Data Sources

- **Stats & bios:** NHL public API (api.nhle.com) — 22 seasons, ~19,700 player-season rows
- **Contracts:** PuckPedia API (requires API key stored as `PUCKPEDIA_API_KEY` repo secret) — ~1,150 contract rows
- **Cap limits:** Manually maintained in `generate_data.py` from official NHL/NHLPA announcements

## App Features

- UFA/RFA and Forward/Defense toggle filters
- Player search with autocomplete
- **Contract & Stats tab** — player overview, estimate banner with cap hit %, AAV, and ±1σ range, confidence pill (tight/moderate/wide), backtest MAE badge, comparable table with IDW weight bars, career stat chart by age, cap hit % future projection chart
- **Stats Detail tab** — full stat lines for player and comps
- **Contract tab** — contract details, AAV range visualization, cap hit % vs term scatter, comp cap hit % bar chart
- **Bio tab** — height, weight, shoots, draft info
- **Info & Notes tab** — model methodology explanation

## Historical Cap Reference

| Season | Upper Limit |
|--------|-------------|
| 2003-04 | $39.0M |
| 2005-06 | $39.0M |
| 2006-07 | $44.0M |
| 2007-08 | $50.3M |
| 2008-09 | $56.7M |
| 2009-10 | $56.8M |
| 2010-11 | $59.4M |
| 2011-12 | $64.3M |
| 2012-13 | $70.2M |
| 2013-14 | $64.3M |
| 2014-15 | $69.0M |
| 2015-16 | $71.4M |
| 2016-17 | $73.0M |
| 2017-18 | $75.0M |
| 2018-19 | $79.5M |
| 2019-20 | $81.5M |
| 2020-21 | $81.5M |
| 2021-22 | $81.5M |
| 2022-23 | $82.5M |
| 2023-24 | $83.5M |
| 2024-25 | $88.0M |
| 2025-26 | $95.5M ✓ |
| 2026-27 | $104.0M (agreed, may adjust to ~$107M) |
| 2027-28 | $113.5M (agreed, subject to minor adjustment) |

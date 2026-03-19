# NHL Contract Comparables Tool

A KNN-based tool that finds the 5 most statistically similar free agents from the prior 3 signing classes to estimate contract values for pending NHL free agents.

**Live site:** `https://cbattinieri.github.io/nhl-contract-comps/`

## Architecture

Static GitHub Pages site — no server, no hosting costs:

```
generate_data.py          → Python pipeline: fetches data, runs KNN, outputs JSON
data/comps.json           → Pre-computed results (auto-updated daily)
index.html                → Static frontend (loads JSON, no server needed)
.github/workflows/        → GitHub Action for daily data refresh
test_local.py             → Local preview server (not deployed)
```

## Season Switch Checklist

Everything is dynamic except the salary cap numbers. Each offseason, update **one thing** in `generate_data.py`:

### 1. Update `CAP_LIMITS` (required)

Open `generate_data.py` and find the `CAP_LIMITS` dictionary near the top. Add the new season's confirmed cap and update any projections:

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

**Where to find it:** [NHL.com salary cap announcements](https://www.nhl.com) or [PuckPedia](https://puckpedia.com).

### 2. Re-run the Action (required)

After updating the cap, go to **Actions → Update Data & Deploy → Run workflow**. This regenerates all data with the new cap numbers.

### That's it

Everything else is automatic:
- **Season detection** — computed from the current date (`year >= July → new season`)
- **Season list** — auto-extends from 2003-04 through the current season
- **Free agent detection** — derives from PuckPedia contract expiry data
- **Daily refresh** — GitHub Action runs at 8 AM UTC during the season

## Model Details

Four KNN models run independently:

| Model | Features | Notes |
|-------|----------|-------|
| RFA Forwards | Age, career GP, career P/G, career EV P/G, GP%, career TOI | Pure career-to-date metrics |
| RFA Defense | Same as RFA Forwards | |
| UFA Forwards | Above + L3 GP avg, L3 P/G avg, L3 EV P/G avg | Age & career stats heavily weighted; platform years at 0.25x |
| UFA Defense | Same as UFA Forwards | |

### Estimation Method

Cap hit % estimates use **inverse-distance weighting (IDW)** — each comp's influence is proportional to `1/distance`. Closer comps dominate; distant comps contribute minimally. Each estimate includes a ±1σ confidence interval.

### Backtest Accuracy

Each model is backtested via leave-one-season-out cross-validation. Typical MAE is ~1.9% of cap hit (~$1.8M AAV on the current cap). The backtest metrics are shown in the app header and regenerated with each data refresh.

## Data Sources

- **Stats & bios:** NHL public API (api.nhle.com)
- **Contracts:** PuckPedia API (requires API key stored as `PUCKPEDIA_API_KEY` repo secret)
- **Cap limits:** Manually maintained in `generate_data.py` from official NHL/NHLPA announcements

## Historical Cap Reference

For convenience, here are all confirmed NHL salary caps:

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

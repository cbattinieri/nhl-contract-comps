# NHL Contract Comparables Tool

A KNN-based tool that finds the 5 most statistically similar free agents from the prior 3 signing classes to estimate contract values for pending NHL free agents.

**Live site:** `https://<your-username>.github.io/nhl-contract-comps/`

## Architecture

This was rebuilt from a Python Shiny app into a **static GitHub Pages site** for reliability and zero hosting cost:

```
generate_data.py          → Python pipeline: fetches data, runs KNN, outputs JSON
data/comps.json           → Pre-computed results (auto-updated daily)
index.html                → Static frontend (loads JSON, no server needed)
.github/workflows/        → GitHub Action for daily data refresh
```

## Key Fixes from Original

- **Dynamic seasons** — season list auto-extends based on current date (no more hardcoded list)
- **In-season signings** — properly handles `signing_status` vs `expiry_status` for current-year FAs
- **No hardcoded column indices** — all references by column name, not position
- **No code duplication** — single `run_knn_model()` function for all 4 model variants
- **Cap ceiling projections** — configurable cap limits dict instead of scattered magic numbers

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/nhl-contract-comps.git
cd nhl-contract-comps
pip install -r requirements.txt
```

### 2. Generate data locally

```bash
export PUCKPEDIA_API_KEY="your_api_key_here"
python generate_data.py
```

This creates `data/comps.json` — all the model results in one file.

### 3. Preview locally

Open `index.html` in a browser, or use a local server:

```bash
python -m http.server 8000
# then visit http://localhost:8000
```

### 4. Deploy to GitHub Pages

1. Push to GitHub
2. Go to **Settings → Pages → Source → Deploy from branch → `main`**
3. Add your PuckPedia API key as a **Repository Secret** named `PUCKPEDIA_API_KEY`
4. The GitHub Action will auto-update `data/comps.json` daily

## Model Details

Four KNN models run independently:

| Model | Features | Notes |
|-------|----------|-------|
| RFA Forwards | Age, career GP, career P/G, career EV P/G, GP%, career TOI | Pure career-to-date metrics |
| RFA Defense | Same as RFA Forwards | |
| UFA Forwards | Above + L3 GP avg, L3 P/G avg, L3 EV P/G avg | Age & career stats heavily weighted; platform years at 0.25x |
| UFA Defense | Same as UFA Forwards | |

Cap Hit % estimate uses weighted averaging: 30% / 25% / 20% / 15% / 10% across the 5 nearest neighbors.

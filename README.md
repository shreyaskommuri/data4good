# WAVE — Workforce Analytics & Vulnerability Engine

WAVE is a tract-level (neighborhood-scale) predictive policy platform designed to prevent economic displacement caused by climate shocks in Santa Barbara County.

When a major storm or flood hits, workers don't just miss shifts — some leave permanently. WAVE answers three questions county planners currently have no tool to answer: **Which neighborhoods lose workers?** **How long does recovery take?** **How much relief needs to be pre-positioned?** It does this in real time, with 100% real data, using a mathematical model that runs across all 109 census tracts in the county simultaneously.

---

## What It Does

| Feature | Description |
|---|---|
| **Interactive Map** | 109 real census tracts colored by displacement risk. Click any tract for a full profile: population, income, EJ burden, coastal job exposure, flood zone status, and exodus probability. |
| **Climate Shock Simulator** | Dial in storm severity and duration. The ODE model instantly recalculates recovery curves, resilience scores, labor flight percentage, and estimated emergency relief costs across the county. |
| **Recovery Chart** | Side-by-side recovery curves for Low / Average / High / Extreme EJ burden tracts — shows exactly how much longer vulnerable communities take to recover from the same shock. |
| **Workforce Panel** | Real workforce data from ~2,575 worker records. Shows which industries are climate-sensitive, where workers go when displaced, and projects post-shock industry shifts using the Markov model. |
| **Housing Pressure Index** | Tracks housing affordability stress across 9 SB County jurisdictions. High housing pressure + high displacement risk = permanent worker exodus. |
| **Economic Impact Scoring** | Each of the 109 tracts gets a vulnerability score (0–100) computed from 12 simulated shock scenarios, with P10–P90 confidence intervals. |
| **AI Policy Chatbot** | Ask natural language questions about any tract, city, or scenario. The chatbot has the full dataset in its context window and gives policy-specific answers. |
| **PDF Export** | One-click export of the full dashboard state as a PDF report. |

---

## The Problem We're Solving

Santa Barbara County employs roughly **220,000 workers**, many in coastal-sensitive jobs: hospitality, tourism, agriculture, and fishing. These industries are the first to collapse in a climate shock and the last to recover — and they disproportionately employ low-income and minority workers who have the fewest buffers.

Existing tools (FEMA's National Risk Index, CalEnviroScreen) score static vulnerability. They tell you how exposed a place *is* — not how its workforce *responds over time*. WAVE models the trajectory: the shape of the recovery curve, the structural reasons some neighborhoods bounce back in weeks while others take years, and the dollar cost of failing to intervene early.

---

## Mathematical Models

### 1. ODE Labor Resilience Engine

The core of WAVE is a **logistic growth differential equation with an Environmental Justice friction term**:

```
dL/dt = rL(1 − L/K) − β(EJ)
```

| Symbol | Meaning |
|---|---|
| `L(t)` | Labor force participation at time t (0–1 scale) |
| `r` | Recovery rate — how fast jobs return (default: 0.10/day) |
| `K` | Carrying capacity — maximum sustainable employment (default: 0.95) |
| `β(EJ)` | Friction coefficient, scaled by the tract's EJ burden percentile |

**Why logistic growth?** Employment recovery is not linear — it accelerates when there is lots of slack and tapers as the economy approaches full capacity. The logistic model captures this S-shaped behavior naturally, the same way ecologists model population recovery after a disaster.

**Why the β friction term?** Communities with higher poverty rates, minority burden, and language barriers face structural headwinds (weaker safety nets, less savings, fewer job connections) that slow recovery independent of the shock itself. β encodes this as a permanent drag on the recovery curve — not a one-time hit, but a sustained disadvantage.

**Climate shock mechanics**: During the shock window, `K` is reduced by `0.3 × severity` (the economy's ceiling drops) and `β` is increased by `0.02 × severity` (friction spikes). Both snap back when the shock ends and the curve climbs again. Solved numerically using `scipy.integrate.solve_ivp` with RK45.

---

### 2. Markov Chain Workforce Transition Model

While the ODE tracks *how many* people are employed, the Markov chain tracks *where workers go*. It models workforce movement across four labor states:

| State | Meaning |
|---|---|
| **COASTAL** | Climate-sensitive jobs (hospitality, tourism, fishing, agriculture) |
| **INLAND** | Climate-resilient jobs (healthcare, tech, education, government) |
| **UNEMPLOYED** | Out of the workforce |
| **TRANSITIONING** | Actively job-searching or in retraining |

Under baseline conditions, a coastal worker has an **85% chance** of staying in their coastal job each period. A climate shock scales the off-diagonal transitions — pushing more workers toward unemployment and transition states. The severity slider directly scales these probabilities.

---

### 3. Scenario-Ensemble Economic Impact Model

For each of the 109 census tracts, WAVE runs a **4 × 3 grid of scenarios** (4 shock severities × 3 durations = 12 simulations per tract), combining ODE stress and Markov displacement pressure:

```
vulnerability_score = 0.70 × ODE_stress + 0.30 × Markov_displacement_pressure
```

The result is a mean vulnerability score (0–100) with P10/P90 confidence intervals — showing not just the expected outcome but the range of plausible outcomes.

---

### 4. Resilience Score (0–100)

```
score = 0.3 × (recovery speed) + 0.3 × (1 − drop depth) + 0.4 × (equilibrium retention)
```

| Score | Status |
|---|---|
| ≥ 60 | RESILIENT |
| 35–59 | AT RISK |
| < 35 | VULNERABLE |

---

### 5. Exodus Probability (per tract)

```
vulnerability = 0.6 × (EJ percentile / 100) + 0.4 × (coastal jobs / 25, capped at 1)
exodus_prob = 0.02 + 0.80 × (vulnerability × shock_multiplier)
```

Where `shock_multiplier = 0.5 + severity × 1.5`. Tracts are then classified into Low / Moderate / High / Critical using quantile bins so all four risk tiers are always represented.

---

## Data Sources (100% Real)

Every number in WAVE comes from a real federal or regional dataset — no synthetic data is used in production.

| Source | What It Provides | Records |
|---|---|---|
| **Census Bureau ACS** | Tract demographics: income, poverty %, minority %, limited English % | 109 tracts |
| **Census TIGERweb** | Actual tract polygon boundaries + centroid coordinates | 109 tracts |
| **NOAA Tides & Currents** | Live 7-day water level observations (station 9411340, SB Harbor) | Rolling 168h |
| **FEMA NFHL** | Flood zone classification per tract centroid | 109 lookups |
| **BLS QCEW** | County-level coastal-sensitive employment share | County-level |
| **Live Data Technologies** | Worker records with full job histories (JSONL) | ~2,575 workers |
| **CA HCD APR 2024** | Housing production, RHNA progress, affordability by jurisdiction | 9,220 records |

All external API data is disk-cached for 24 hours in `data/processed/` and pre-warmed at server startup — so the dashboard loads instantly and is immune to API outages during a demo.

---

## Architecture

```
Browser (React + Vite)  :5173
  │
  │  /api/* (Vite proxy in dev, served directly in prod)
  │
FastAPI  (api.py)  :8000
  │
  ├── src/models/resilience_ode.py     ← ODE engine (scipy RK45)
  ├── src/models/markov_chain.py       ← Markov transition matrix
  ├── src/models/displacement_model.py ← XGBoost risk classifier (17 features)
  ├── src/data/real_data_fetcher.py    ← Census / NOAA / FEMA / BLS clients
  ├── src/data/live_data_loader.py     ← Live Data Technologies JSONL loader
  └── src/data/housing_loader.py       ← CA HCD APR Excel loader
```

**Frontend components:**

| Component | Role |
|---|---|
| `TractMap.jsx` | MapLibre GL choropleth map with 3D extrusion, flood overlays, click handlers |
| `RecoveryChart.jsx` | Recharts recovery curve with shock region annotation |
| `KPIHeader.jsx` | Resilience score badge + 4 live KPI cards |
| `ControlPanel.jsx` | Scenario parameter sliders (severity, duration, r, K) |
| `WorkforcePanel.jsx` | Industry distribution + transition Sankey diagram |
| `HousingPanel.jsx` | Housing Pressure Index bar chart by jurisdiction |
| `MarkovPanel.jsx` | Markov state transition breakdown |
| `EconomicImpactPanel.jsx` | Scenario-ensemble vulnerability scores |
| `ChatPanel.jsx` | AI chatbot with full dataset context injection |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/tracts?severity=` | 109 census tracts with demographics, EJ scores, exodus probabilities |
| `GET /api/simulation?severity=&duration=&r=&K=` | ODE simulation: resilience score, recovery time, labor flight %, emergency fund |
| `GET /api/simulation/compare?...` | Recovery curves for 4 EJ burden profiles (Low / Average / High / Extreme) |
| `GET /api/noaa` | Live NOAA water level data (168h window) |
| `GET /api/workforce` | Industry distribution, job transitions, climate sensitivity breakdown |
| `GET /api/workforce/projected?severity=&duration=` | Markov-projected post-shock industry shifts |
| `GET /api/housing` | Housing Pressure Index by SB County jurisdiction |
| `GET /api/markov?severity=&duration=` | Markov state transition probabilities under shock |
| `GET /api/economic-impact` | Scenario-ensemble vulnerability scores for all 109 tracts |
| `GET /api/tract-boundaries?severity=` | Full polygon GeoJSON with all metrics merged as feature properties |
| `GET /api/city-boundaries` | Convex hull city/community boundaries with aggregate stats |
| `GET /api/county-outline` | Outer boundary of Santa Barbara County |
| `POST /api/chat` | AI policy chatbot (GPT-4o mini via GitHub Models API) |

---

## Key Metrics Glossary

| Metric | Definition |
|---|---|
| **Resilience Score** | 0–100 composite: 30% recovery speed + 30% shock depth + 40% long-run equilibrium retention |
| **Recovery Time** | Days from shock start until employment returns to 95% of pre-shock level |
| **Labor Flight %** | Percentage of the workforce that leaves during the shock trough |
| **Minimum Employment** | Lowest employment level reached during the shock (as % of workforce) |
| **Exodus Probability** | Per-tract probability of permanent worker displacement; combines EJ burden and coastal job dependency |
| **EJ Percentile** | Environmental Justice burden: `0.4 × poverty% + 0.3 × minority% + 0.3 × limited_english%` |
| **Housing Pressure Index** | RHNA gap + affordability gap + ADU dependence + rental pressure, per jurisdiction |
| **Emergency Fund** | Estimated stabilization cost: `affected_workers × ($2,500 + $1,500 × severity) × duration_factor` |

---

## Quick Start

### 1. Install Backend Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> The venv folder is named `venv`, not `.venv`.

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

> Node.js 20.19+ required (Vite v7 compatibility).

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set:

```env
# Required for the AI Chatbot
# Create a free classic PAT at https://github.com/settings/tokens
# Select "Classic token" with NO scopes checked
GITHUB_TOKEN=ghp_your_token_here
```

> The chatbot uses the [GitHub Models API](https://github.com/marketplace/models) — free at 150 requests/day, powered by GPT-4o mini.

### 4. Run

Open two terminals:

**Terminal 1 — Backend**
```bash
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

> On first load, the backend pre-warms all data caches in a background thread. The dashboard will be fully populated within ~10 seconds.

### 5. Validate Data Sources

```bash
python scripts/check_data.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| Frontend | React 18, Vite 7 |
| Map | MapLibre GL + CARTO Dark Matter tiles (free, no API key) |
| Charts | Recharts, Nivo |
| ODE Solver | scipy `solve_ivp` (RK45) |
| ML Model | XGBoost (17-feature displacement risk classifier, ROC AUC 0.87) |
| AI Chatbot | GPT-4o mini via GitHub Models API |
| PDF Export | jsPDF + html2canvas |
| Data | pandas, numpy |

---

## Reference

- **County FIPS**: 06083 (Santa Barbara)
- **NOAA Station**: 9411340 (Santa Barbara Harbor)
- **Census Tracts**: 109 (all of Santa Barbara County)
- **Baseline Labor Force (L₀)**: 0.92 (92% employment — consistent with SB County BLS data)
- **ODE Initial Recovery Rate (r)**: 0.10/day
- **ODE Carrying Capacity (K)**: 0.95

---

## License

MIT License

# WAVE — Workforce Analytics & Vulnerability Engine

Real-time modeling of coastal labor resilience for Santa Barbara County. This dashboard combines live environmental hazards, workforce transitions, housing pressure, and demographic vulnerability to simulate how climate shocks impact employment recovery.

## What It Does

- Models labor force recovery under climate shocks using ODEs and Markov transitions
- Uses 100% real data from federal and regional sources
- Explains resilience gaps by EJ burden, housing pressure, and coastal exposure
- Interactive map of 109 census tracts with click popups, flood zone overlays, and vulnerability coloring
- Dynamic recovery charts that auto-trim to meaningful time ranges
- Housing Pressure Index across 9 Santa Barbara County jurisdictions

## Data Sources (All Real)

- **Census Bureau ACS**: Tract demographics, income, poverty, minority %
- **Census TIGERweb**: Tract centroid coordinates
- **NOAA Tides & Currents**: Water level observations (station 9411340)
- **FEMA NFHL**: Flood zone classifications
- **BLS QCEW**: Coastal-sensitive employment share (county-level)
- **Live Data Technologies**: Workforce profiles and transitions (~2,575 records)
- **CA HCD APR (via SBCAG)**: Housing production and affordability (APR_Download_2024.xlsx, 9,220 records)

## Quick Start

### 1) Install Backend Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The venv folder is named `venv`, not `.venv`.

### 2) Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3) Configure Environment Variables

Copy `.env` and fill in your credentials:

```bash
cp .env .env.local   # optional, .env works directly
```

Open `.env` and set:

```env
# Required for the AI Chatbot
# Create a free classic PAT at https://github.com/settings/tokens
# Select "Classic token" with NO scopes checked
GITHUB_TOKEN=ghp_your_token_here
```

> **Chatbot** uses the [GitHub Models API](https://github.com/marketplace/models) (free, 150 req/day) — powered by GPT-4o mini via your GitHub token.

### 4) Install PDF Export Dependencies (Frontend)

The PDF export button requires two extra packages:

```bash
cd frontend
npm install jspdf html2canvas
cd ..
```

### 5) Run

Open two terminals:

**Terminal 1 — Backend (FastAPI)**
```bash
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Frontend (React/Vite)**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

## Architecture

- **Backend**: FastAPI (`api.py`) serving REST endpoints on port 8000
- **Frontend**: React + Vite (`frontend/`) with Vite proxy forwarding `/api` to the backend
- **Map**: MapLibre GL + CARTO Dark Matter tiles (free, no API key)
- **Charts**: Recharts
- **ODE Model**: `dL/dt = rL(1−L/K) − β(EJ)` across 109 census tracts

## Repository Layout

```
data4good/
├── api.py                     # FastAPI backend (all API endpoints)
├── APR_Download_2024.xlsx     # SBCAG APR housing data export
├── check_data.py              # Real data validation script
├── frontend/
│   ├── src/
│   │   ├── main.jsx           # React entry point
│   │   ├── App.jsx            # Main layout
│   │   ├── api.js             # API client
│   │   ├── hooks.js           # useApi hook
│   │   ├── index.css          # Design system (dark mode)
│   │   ├── ErrorBoundary.jsx  # Error boundary
│   │   └── components/
│   │       ├── KPIHeader.jsx      # Resilience score + KPI cards
│   │       ├── ControlPanel.jsx   # Scenario sliders
│   │       ├── TractMap.jsx       # Interactive MapLibre map
│   │       ├── RecoveryChart.jsx  # Employment recovery curves
│   │       ├── MarkovPanel.jsx    # Worker flow visualization
│   │       ├── HousingPanel.jsx   # Housing Pressure Index
│   │       ├── WorkforcePanel.jsx # Live Data workforce intel
│   │       ├── NoaaPanel.jsx      # NOAA sea level data
│   │       ├── PolicySection.jsx  # Auto-generated recommendations
│   │       ├── ChatPanel.jsx      # AI policy chatbot (GitHub Models)
│   │       └── PDFExportButton.jsx # PDF report export
│   ├── vite.config.js         # Vite config with API proxy
│   └── package.json
├── src/
│   ├── data/
│   │   ├── real_data_fetcher.py   # Census, NOAA, FEMA, BLS clients
│   │   ├── live_data_loader.py    # Live Data Technologies loader
│   │   └── housing_loader.py      # APR housing data + pressure index
│   └── models/
│       ├── resilience_ode.py      # ODE labor resilience model
│       └── markov_chain.py        # Workforce transition model
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/tracts?severity=` | 109 census tracts with vulnerability scores |
| `GET /api/simulation?...` | ODE simulation results + resilience score |
| `GET /api/simulation/compare?...` | Recovery curves by EJ burden level |
| `GET /api/noaa` | NOAA water level observations |
| `GET /api/workforce` | Live Data Technologies workforce data |
| `GET /api/workforce/projected?severity=&duration=` | Projected workforce shifts under shock |
| `GET /api/housing` | Housing Pressure Index by jurisdiction |
| `GET /api/markov?severity=&duration=` | Markov chain transition probabilities |
| `POST /api/chat` | AI policy chatbot (GitHub Models / GPT-4o mini) |

## Key Metrics

- **Resilience Index**: 0–1 composite of depth of drop + recovery speed
- **Recovery Time**: Days to return to ~95% of baseline employment
- **Minimum Employment**: Lowest employment during shock
- **Housing Pressure Index**: RHNA gap + affordability gap + ADU dependence + rental pressure
- **Exodus Probability**: Per-tract flight risk based on EJ burden + coastal dependency

## Validate Data Sources

```bash
python3 check_data.py
```

## Notes

- County FIPS: 06083 (Santa Barbara)
- NOAA station: 9411340 (Santa Barbara)
- 109 census tracts with real coordinates from TIGERweb
- Node.js 20.19+ recommended (Vite v7 compatibility)

## License

MIT License

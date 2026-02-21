# Coastal Labor-Resilience Engine

Real-time modeling of coastal labor resilience for Santa Barbara County. This dashboard combines live environmental hazards, workforce transitions, housing pressure, and demographic vulnerability to simulate how climate shocks impact employment recovery.

## What It Does

- Models labor force recovery under climate shocks using ODEs and Markov transitions
- Uses 100% real data from federal and regional sources
- Explains resilience gaps by EJ burden, housing pressure, and coastal exposure
- Visualizes tract-level vulnerability, workforce intelligence, and hazard signals

## Data Sources (All Real)

- **Census Bureau ACS**: Tract demographics, income, poverty, minority %
- **Census TIGERweb**: Tract centroid coordinates
- **NOAA Tides & Currents**: Water level observations (station 9411340)
- **FEMA NFHL**: Flood zone classifications
- **BLS QCEW**: Coastal-sensitive employment share (county-level)
- **Live Data Technologies**: Workforce profiles and transitions
- **CA HCD APR (via SBCAG)**: Housing production and affordability (APR_Download_2024.xlsx)

## Quick Start

### 1) Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Add Environment Variables

```bash
cp .env.example .env
```

### 3) Run the Dashboard

```bash
streamlit run app.py
```

## Repository Layout

```
data4good/
├── app.py                     # Streamlit dashboard
├── APR_Download_2024.xlsx     # SBCAG APR housing data export
├── check_data.py              # Real data validation script
├── config/
├── data/
├── notebooks/
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

## Key Metrics

- **Resilience Index**: 0–1 composite of depth of drop + recovery speed
- **Recovery Time**: Days to return to ~95% of baseline employment
- **Minimum Employment**: Lowest employment during shock
- **Housing Pressure Index**: RHNA gap + affordability gap + ADU dependence + rental pressure

## Validate Data Sources

```bash
python3 check_data.py
```

## Notes

- County FIPS: 06083 (Santa Barbara)
- NOAA station: 9411340 (Santa Barbara)

## License

MIT License

# Coastal Labor-Resilience Engine

A data science project to analyze the relationship between coastal environmental events and workforce resilience in the Santa Barbara region.

## 🎯 Project Overview

This project combines multiple data sources to understand how coastal environmental shocks (storms, high tides, flooding) impact local workforce dynamics. By mapping environmental events to employment changes and overlaying demographic vulnerability data, we can better understand and predict community resilience patterns.

## 📁 Project Structure

```
coastal-labor-resilience/
├── config/                    # Configuration settings
│   ├── __init__.py
│   └── settings.py           # API keys, paths, constants
├── data/
│   ├── raw/                  # Original, immutable data
│   │   ├── ejscreen_data.csv # EPA EJScreen demographics
│   │   └── ...
│   └── processed/            # Cleaned, aligned data
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── data/                 # Data ingestion scripts
│   │   ├── __init__.py
│   │   ├── load_data.py      # Main data loading functions
│   │   ├── auth.py           # API authentication
│   │   ├── workforce.py      # Live Data API client
│   │   ├── noaa_baseline.py  # NOAA environmental data
│   │   ├── demographic_overlay.py  # EJScreen integration
│   │   └── cleaning.py       # Data cleaning & alignment
│   └── models/               # Markov & ODE models (Phase 2)
├── .env                      # Environment variables (not in git)
├── .env.example              # Template for environment variables
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API credentials
# - LIVE_DATA_API_KEY
# - LIVE_DATA_API_SECRET
```

### 3. Download Required Data

**EPA EJScreen Data:**
1. Visit https://www.epa.gov/ejscreen/download-ejscreen-data
2. Download the CSV for California
3. Save as `data/raw/ejscreen_data.csv`

### 4. Test Data Loading

```bash
# Test NOAA data fetch
python -m src.data.load_data

# Test environmental baselining
python -m src.data.noaa_baseline

# Test demographic overlay
python -m src.data.demographic_overlay
```

## 📊 Phase 1 - Data Foundation

### Step 1: API Authentication
```python
from src.data.auth import TokenManager

# Authenticate with Live Data Technologies
token_mgr = TokenManager()
token = token_mgr.access_token
```

### Step 2: Workforce Scoping
```python
from src.data.workforce import fetch_santa_barbara_workforce

# Fetch workforce profiles for Santa Barbara
workforce_df = fetch_santa_barbara_workforce(
    industries=["Tourism", "Hospitality", "Fishing"],
    max_profiles=1000
)
```

### Step 3: Environmental Baselining
```python
from src.data.noaa_baseline import (
    NOAAEnvironmentalClient,
    get_shock_dates
)

# Get historical extreme water events
client = NOAAEnvironmentalClient(station_id="9411340")  # Santa Barbara
extreme_events = client.identify_highest_observed_tides(years_back=10)

# Identify "shock dates" - extreme coastal events
shock_dates = get_shock_dates(threshold_percentile=99, years_back=10)
```

### Step 4: Demographic Overlay
```python
from src.data.demographic_overlay import overlay_demographics

# Merge workforce with EPA EJScreen vulnerability data
merged_df = overlay_demographics(workforce_df)

# Access social vulnerability indicators
print(merged_df[['first_name', 'zip_code', 'DEMOGIDX', 'LOWINCPCT']])
```

### Step 5: Data Cleaning & Alignment
```python
from src.data.cleaning import align_all_datasets

# Align environmental events with workforce changes
aligned_data = align_all_datasets(
    workforce_df=workforce_df,
    events_df=extreme_events,
    window_days=30
)

# Access aligned data
aligned_events = aligned_data['aligned']
timeline = aligned_data['timeline']
```

## 🔧 Data Sources

| Data Source | Description | API/Format |
|-------------|-------------|------------|
| Live Data Technologies | Professional profiles, employment history | REST API |
| NOAA Tides & Currents | Water levels, tide predictions, extreme events | REST API |
| EPA EJScreen | Environmental justice indicators, demographics | CSV |

## 📝 Key Metrics

- **NOAA Station**: 9411340 (Santa Barbara, CA)
- **Santa Barbara County FIPS**: 06083
- **Key ZIP Codes**: 93101-93199

## 🔮 Next Steps (Phase 2)

- [ ] Implement Markov chain models for workforce state transitions
- [ ] Build ODE models for resilience dynamics
- [ ] Create predictive models for vulnerability assessment
- [ ] Develop interactive visualizations

## 📜 License

MIT License

## 🤝 Contributing

This project is part of a hackathon ("Data for Good"). Contributions welcome!

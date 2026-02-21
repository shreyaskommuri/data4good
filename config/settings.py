"""
Configuration settings for the Coastal Labor-Resilience Engine.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================================
# API Configuration - Live Data Technologies
# ============================================================================
LIVE_DATA_API_BASE_URL = "https://api.livedata.io/v1"
LIVE_DATA_API_KEY = os.getenv("LIVE_DATA_API_KEY")
LIVE_DATA_API_SECRET = os.getenv("LIVE_DATA_API_SECRET")

# ============================================================================
# API Configuration - NOAA Tides and Currents
# ============================================================================
NOAA_API_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_STATION_SANTA_BARBARA = "9411340"

# NOAA API parameters
NOAA_DEFAULT_PARAMS = {
    "station": NOAA_STATION_SANTA_BARBARA,
    "product": "water_level",
    "datum": "MLLW",  # Mean Lower Low Water
    "units": "metric",
    "time_zone": "gmt",
    "format": "json",
    "application": "coastal_labor_resilience_engine"
}

# ============================================================================
# EPA EJScreen Configuration
# ============================================================================
EPA_EJSCREEN_DATA_PATH = RAW_DATA_DIR / "ejscreen_data.csv"

# Santa Barbara County FIPS code
SANTA_BARBARA_COUNTY_FIPS = "06083"

# ============================================================================
# Geographic Scope
# ============================================================================
SANTA_BARBARA_ZIP_CODES = [
    "93101", "93102", "93103", "93105", "93106", "93107",
    "93108", "93109", "93110", "93111", "93117", "93118",
    "93120", "93121", "93130", "93140", "93150", "93160",
    "93190", "93199"
]

# ============================================================================
# Data Processing Settings
# ============================================================================
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIMEZONE = "America/Los_Angeles"

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def validate_config():
    """Validate that required configuration is present."""
    missing = []
    
    if not LIVE_DATA_API_KEY:
        missing.append("LIVE_DATA_API_KEY")
    if not LIVE_DATA_API_SECRET:
        missing.append("LIVE_DATA_API_SECRET")
    
    if missing:
        print(f"Warning: Missing environment variables: {', '.join(missing)}")
        print("Some functionality may be limited.")
        return False
    
    return True


if __name__ == "__main__":
    # Quick validation check
    validate_config()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")

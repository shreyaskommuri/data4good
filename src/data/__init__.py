"""
Data ingestion and preprocessing module for Coastal Labor-Resilience Engine.

This module provides:
- Live Data Technologies API client for workforce data
- NOAA Tides and Currents API client for environmental data
- EPA EJScreen data loading for demographics
- Data cleaning and alignment utilities

Usage:
    from src.data import (
        LiveDataAPIClient,
        NOAADataClient,
        load_ejscreen_data,
        align_all_datasets
    )
"""

from .load_data import (
    LiveDataAPIClient,
    NOAADataClient,
    load_ejscreen_data,
    load_all_data
)

from .auth import (
    TokenManager,
    get_bearer_token,
    AuthenticationError
)

from .workforce import (
    WorkforceClient,
    fetch_santa_barbara_workforce,
    SANTA_BARBARA_ZIP_CODES,
    COASTAL_INDUSTRIES
)

from .noaa_baseline import (
    NOAAEnvironmentalClient,
    get_extreme_water_events,
    get_shock_dates,
    save_baseline_data
)

from .demographic_overlay import (
    DemographicOverlay,
    overlay_demographics,
    create_vulnerability_report
)

from .cleaning import (
    DataAligner,
    DataCleaner,
    align_all_datasets
)

__all__ = [
    # Main data loaders
    "LiveDataAPIClient",
    "NOAADataClient",
    "load_ejscreen_data",
    "load_all_data",
    
    # Authentication
    "TokenManager",
    "get_bearer_token",
    "AuthenticationError",
    
    # Workforce
    "WorkforceClient",
    "fetch_santa_barbara_workforce",
    "SANTA_BARBARA_ZIP_CODES",
    "COASTAL_INDUSTRIES",
    
    # NOAA/Environmental
    "NOAAEnvironmentalClient",
    "get_extreme_water_events",
    "get_shock_dates",
    "save_baseline_data",
    
    # Demographics
    "DemographicOverlay",
    "overlay_demographics",
    "create_vulnerability_report",
    
    # Cleaning/Alignment
    "DataAligner",
    "DataCleaner",
    "align_all_datasets"
]

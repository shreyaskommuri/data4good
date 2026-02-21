"""
Data Loading Module for Coastal Labor-Resilience Engine

This module provides unified data ingestion from multiple sources:
1. Live Data Technologies People API (workforce data)
2. NOAA Tides and Currents API (environmental data)
3. EPA EJScreen (demographic/environmental justice data)

Author: Coastal Labor-Resilience Engine Team
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
LIVE_DATA_API_BASE_URL = "https://api.livedata.io/v1"
NOAA_API_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
SANTA_BARBARA_STATION_ID = "9411340"


class LiveDataAPIClient:
    """
    Client for interacting with Live Data Technologies People API.
    
    This client handles authentication and provides methods for
    searching professional profiles in the Santa Barbara area.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the Live Data API client.
        
        Args:
            api_key: API key for authentication. If None, reads from LIVE_DATA_API_KEY env var.
            api_secret: API secret for authentication. If None, reads from LIVE_DATA_API_SECRET env var.
        """
        self.api_key = api_key or os.getenv("LIVE_DATA_API_KEY")
        self.api_secret = api_secret or os.getenv("LIVE_DATA_API_SECRET")
        self.base_url = LIVE_DATA_API_BASE_URL
        self._bearer_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        if not self.api_key or not self.api_secret:
            logger.warning("Live Data API credentials not configured. Set LIVE_DATA_API_KEY and LIVE_DATA_API_SECRET.")
    
    def authenticate(self) -> str:
        """
        Authenticate with Live Data Technologies API and retrieve Bearer token.
        
        Returns:
            str: Bearer token for API requests.
            
        Raises:
            requests.HTTPError: If authentication fails.
        """
        logger.info("Authenticating with Live Data Technologies API...")
        
        auth_url = f"{self.base_url}/auth/token"
        
        payload = {
            "api_key": self.api_key,
            "api_secret": self.api_secret
        }
        
        try:
            response = requests.post(auth_url, json=payload, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self._bearer_token = token_data.get("access_token")
            
            # Set token expiry (typically 1 hour, but check response)
            expires_in = token_data.get("expires_in", 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info("Successfully authenticated with Live Data API.")
            return self._bearer_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    @property
    def bearer_token(self) -> str:
        """Get valid bearer token, refreshing if necessary."""
        if not self._bearer_token or (self._token_expiry and datetime.now() >= self._token_expiry):
            self.authenticate()
        return self._bearer_token
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authorization for API requests."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
    
    def search_workforce(
        self,
        location: str = "Santa Barbara, CA",
        zip_codes: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Search for professional profiles in specified location.
        
        Args:
            location: City/region to search (default: Santa Barbara, CA)
            zip_codes: Optional list of ZIP codes to filter by
            industries: Optional list of industries to filter by
            limit: Maximum number of results to return
            
        Returns:
            DataFrame containing workforce profile data.
        """
        logger.info(f"Searching workforce data for {location}...")
        
        search_url = f"{self.base_url}/search"
        
        # Build search query
        query = {
            "location": location,
            "limit": limit
        }
        
        if zip_codes:
            query["zip_codes"] = zip_codes
        if industries:
            query["industries"] = industries
        
        try:
            response = requests.post(
                search_url,
                headers=self._get_headers(),
                json=query,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            profiles = data.get("results", [])
            
            logger.info(f"Retrieved {len(profiles)} workforce profiles.")
            return pd.DataFrame(profiles)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Workforce search failed: {e}")
            raise


class NOAADataClient:
    """
    Client for fetching data from NOAA Tides and Currents API.
    
    Provides methods for retrieving water levels, tide predictions,
    and historical extreme water events for coastal monitoring.
    """
    
    def __init__(self, station_id: str = SANTA_BARBARA_STATION_ID):
        """
        Initialize NOAA data client.
        
        Args:
            station_id: NOAA station identifier (default: Santa Barbara 9411340)
        """
        self.station_id = station_id
        self.base_url = NOAA_API_BASE_URL
        
    def fetch_water_levels(
        self,
        begin_date: str,
        end_date: str,
        datum: str = "MLLW",
        interval: str = "h"
    ) -> pd.DataFrame:
        """
        Fetch water level data for the station.
        
        Args:
            begin_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            datum: Vertical datum (MLLW, MSL, NAVD, etc.)
            interval: Data interval ('h' for hourly, '6' for 6-min)
            
        Returns:
            DataFrame with water level observations.
        """
        logger.info(f"Fetching water levels from {begin_date} to {end_date}...")
        
        params = {
            "station": self.station_id,
            "begin_date": begin_date,
            "end_date": end_date,
            "product": "water_level",
            "datum": datum,
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
            "interval": interval,
            "application": "coastal_labor_resilience_engine"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                logger.error(f"NOAA API error: {data['error']}")
                raise ValueError(data["error"]["message"])
            
            observations = data.get("data", [])
            df = pd.DataFrame(observations)
            
            if not df.empty:
                df["t"] = pd.to_datetime(df["t"])
                df["v"] = pd.to_numeric(df["v"], errors="coerce")
                df.rename(columns={"t": "timestamp", "v": "water_level_m"}, inplace=True)
            
            logger.info(f"Retrieved {len(df)} water level observations.")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch water levels: {e}")
            raise
    
    def fetch_high_low_tides(
        self,
        begin_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch high/low tide data to identify extreme events.
        
        Args:
            begin_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            
        Returns:
            DataFrame with high/low tide observations.
        """
        logger.info(f"Fetching high/low tides from {begin_date} to {end_date}...")
        
        params = {
            "station": self.station_id,
            "begin_date": begin_date,
            "end_date": end_date,
            "product": "high_low",
            "datum": "MLLW",
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
            "application": "coastal_labor_resilience_engine"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if "error" in data:
                logger.error(f"NOAA API error: {data['error']}")
                raise ValueError(data["error"]["message"])
            
            observations = data.get("data", [])
            df = pd.DataFrame(observations)
            
            if not df.empty:
                df["t"] = pd.to_datetime(df["t"])
                df["v"] = pd.to_numeric(df["v"], errors="coerce")
                df.rename(columns={
                    "t": "timestamp",
                    "v": "water_level_m",
                    "ty": "tide_type"
                }, inplace=True)
            
            logger.info(f"Retrieved {len(df)} high/low tide observations.")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch high/low tides: {e}")
            raise
    
    def identify_extreme_events(
        self,
        df: pd.DataFrame,
        threshold_percentile: float = 99.0
    ) -> pd.DataFrame:
        """
        Identify extreme water level events (potential coastal shocks).
        
        Args:
            df: DataFrame with water level data
            threshold_percentile: Percentile threshold for extreme events
            
        Returns:
            DataFrame with extreme events only.
        """
        if df.empty:
            return df
        
        threshold = df["water_level_m"].quantile(threshold_percentile / 100)
        extreme_events = df[df["water_level_m"] >= threshold].copy()
        
        logger.info(f"Identified {len(extreme_events)} extreme events above {threshold:.2f}m")
        return extreme_events


def load_ejscreen_data(
    file_path: Optional[str] = None,
    county_fips: str = "06083"  # Santa Barbara County
) -> pd.DataFrame:
    """
    Load EPA EJScreen data from local CSV file.
    
    EJScreen provides environmental justice indices including:
    - Demographic indicators (minority %, low income %, etc.)
    - Environmental indicators (air quality, proximity to hazards)
    - Social vulnerability indices
    
    Args:
        file_path: Path to EJScreen CSV file. If None, uses default location.
        county_fips: County FIPS code to filter data (default: Santa Barbara)
        
    Returns:
        DataFrame with EJScreen data for specified county.
    """
    if file_path is None:
        file_path = Path(__file__).parent.parent.parent / "data" / "raw" / "ejscreen_data.csv"
    
    logger.info(f"Loading EJScreen data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, dtype={"GEOID": str})
        
        # Filter to county if GEOID contains FIPS
        if "GEOID" in df.columns:
            # Census tract GEOID format: SSCCCTTTTTT (State, County, Tract)
            df = df[df["GEOID"].str.startswith(county_fips[:5])]
        
        logger.info(f"Loaded {len(df)} census tract records.")
        return df
        
    except FileNotFoundError:
        logger.error(f"EJScreen file not found: {file_path}")
        logger.info("Download EJScreen data from: https://www.epa.gov/ejscreen/download-ejscreen-data")
        raise
    except Exception as e:
        logger.error(f"Failed to load EJScreen data: {e}")
        raise


# ============================================================================
# Main Data Loading Function
# ============================================================================

def load_all_data(
    begin_date: str,
    end_date: str,
    ejscreen_path: Optional[str] = None,
    include_workforce: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load all data sources for the Coastal Labor-Resilience Engine.
    
    Args:
        begin_date: Start date for NOAA data (YYYYMMDD)
        end_date: End date for NOAA data (YYYYMMDD)
        ejscreen_path: Optional path to EJScreen CSV
        include_workforce: Whether to fetch Live Data workforce profiles
        
    Returns:
        Dictionary containing all loaded DataFrames:
        - 'water_levels': NOAA water level observations
        - 'high_low_tides': NOAA high/low tide data
        - 'ejscreen': EPA EJScreen demographic data
        - 'workforce': Live Data workforce profiles (if included)
    """
    data = {}
    
    # Load NOAA data
    noaa_client = NOAADataClient()
    data["water_levels"] = noaa_client.fetch_water_levels(begin_date, end_date)
    data["high_low_tides"] = noaa_client.fetch_high_low_tides(begin_date, end_date)
    
    # Load EJScreen data
    try:
        data["ejscreen"] = load_ejscreen_data(ejscreen_path)
    except FileNotFoundError:
        logger.warning("EJScreen data not available. Skipping.")
        data["ejscreen"] = pd.DataFrame()
    
    # Load workforce data (if API credentials available)
    if include_workforce:
        try:
            live_data_client = LiveDataAPIClient()
            data["workforce"] = live_data_client.search_workforce()
        except Exception as e:
            logger.warning(f"Workforce data not available: {e}")
            data["workforce"] = pd.DataFrame()
    
    return data


if __name__ == "__main__":
    # Example usage
    print("Coastal Labor-Resilience Engine - Data Loading Module")
    print("=" * 60)
    
    # Test NOAA data fetch (last 7 days)
    end_date = datetime.now().strftime("%Y%m%d")
    begin_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    
    print(f"\nFetching NOAA data from {begin_date} to {end_date}...")
    
    noaa_client = NOAADataClient()
    water_levels = noaa_client.fetch_water_levels(begin_date, end_date)
    
    if not water_levels.empty:
        print(f"\nWater Level Statistics:")
        print(f"  Records: {len(water_levels)}")
        print(f"  Min: {water_levels['water_level_m'].min():.3f} m")
        print(f"  Max: {water_levels['water_level_m'].max():.3f} m")
        print(f"  Mean: {water_levels['water_level_m'].mean():.3f} m")

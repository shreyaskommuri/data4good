"""
NOAA Environmental Baselining Module

Step 3: Environmental Baselining
A script that pulls the "Highest Observed Tide" or "Max Tide Date" from NOAA
for the Santa Barbara station to identify historical shock dates.

Author: Coastal Labor-Resilience Engine Team
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NOAA API Configuration
NOAA_API_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
SANTA_BARBARA_STATION_ID = "9411340"

# NOAA Products for environmental analysis
NOAA_PRODUCTS = {
    "water_level": "Verified 6-minute water level data",
    "hourly_height": "Verified hourly water level data",
    "high_low": "Verified high/low water level data",
    "predictions": "Tide predictions",
    "datums": "Tidal datums",
    "air_temperature": "Air temperature",
    "water_temperature": "Water temperature",
    "wind": "Wind speed and direction"
}


class NOAAEnvironmentalClient:
    """
    Client for fetching environmental data from NOAA Tides and Currents.
    
    Specialized for identifying historical extreme events and
    establishing environmental baselines for coastal resilience analysis.
    """
    
    def __init__(self, station_id: str = SANTA_BARBARA_STATION_ID):
        """
        Initialize the NOAA Environmental Client.
        
        Args:
            station_id: NOAA station identifier. Default is Santa Barbara (9411340).
        """
        self.station_id = station_id
        self.base_url = NOAA_API_BASE_URL
        self._datums: Optional[Dict] = None
    
    def _make_request(
        self,
        params: Dict,
        timeout: int = 60
    ) -> Dict:
        """
        Make a request to the NOAA API.
        
        Args:
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            JSON response data
            
        Raises:
            ValueError: If API returns an error
            requests.RequestException: If request fails
        """
        params["application"] = "coastal_labor_resilience_engine"
        params["format"] = "json"
        
        response = requests.get(self.base_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            raise ValueError(f"NOAA API error: {error_msg}")
        
        return data
    
    def get_station_datums(self) -> Dict[str, float]:
        """
        Retrieve tidal datums for the station.
        
        Datums include:
        - MHHW: Mean Higher High Water
        - MHW: Mean High Water
        - MTL: Mean Tide Level
        - MSL: Mean Sea Level
        - MLW: Mean Low Water
        - MLLW: Mean Lower Low Water
        - HAT: Highest Astronomical Tide
        - LAT: Lowest Astronomical Tide
        
        Returns:
            Dict mapping datum names to their values in meters.
        """
        if self._datums is not None:
            return self._datums
        
        logger.info("Fetching station datums...")
        
        params = {
            "station": self.station_id,
            "product": "datums",
            "units": "metric"
        }
        
        data = self._make_request(params)
        
        # Parse datums
        self._datums = {}
        for datum in data.get("datums", []):
            name = datum.get("n")
            value = float(datum.get("v", 0))
            self._datums[name] = value
        
        logger.info(f"Retrieved {len(self._datums)} datums.")
        return self._datums
    
    def fetch_high_low_data(
        self,
        begin_date: str,
        end_date: str,
        datum: str = "MLLW"
    ) -> pd.DataFrame:
        """
        Fetch high/low tide data for the station.
        
        Args:
            begin_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            datum: Vertical datum (MLLW, MSL, etc.)
            
        Returns:
            DataFrame with high/low tide observations including:
            - timestamp: Date/time of observation
            - water_level_m: Water level in meters
            - tide_type: 'H' for high, 'L' for low,
                        'HH' for higher high, 'LL' for lower low
        """
        logger.info(f"Fetching high/low data: {begin_date} to {end_date}")
        
        params = {
            "station": self.station_id,
            "product": "high_low",
            "begin_date": begin_date,
            "end_date": end_date,
            "datum": datum,
            "units": "metric",
            "time_zone": "gmt"
        }
        
        data = self._make_request(params)
        
        df = pd.DataFrame(data.get("data", []))
        
        if not df.empty:
            df["t"] = pd.to_datetime(df["t"])
            df["v"] = pd.to_numeric(df["v"], errors="coerce")
            df = df.rename(columns={
                "t": "timestamp",
                "v": "water_level_m",
                "ty": "tide_type"
            })
        
        logger.info(f"Retrieved {len(df)} high/low observations.")
        return df
    
    def identify_highest_observed_tides(
        self,
        years_back: int = 10,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify the highest observed tide levels over a historical period.
        
        This method retrieves high/low data and identifies extreme high
        water events that may correspond to coastal "shocks".
        
        Args:
            years_back: How many years of history to analyze
            top_n: Number of top extreme events to return
            
        Returns:
            DataFrame with top extreme high water events:
            - timestamp: Date/time of event
            - water_level_m: Water level in meters
            - anomaly_m: Amount above Mean Higher High Water (MHHW)
            - tide_type: Type of tide observation
        """
        logger.info(f"Identifying highest observed tides ({years_back} years)...")
        
        # Get datums for reference
        datums = self.get_station_datums()
        mhhw = datums.get("MHHW", 0)
        
        # Calculate date range
        end_date = datetime.now()
        begin_date = end_date - timedelta(days=years_back * 365)
        
        all_data = []
        
        # NOAA limits requests to ~31 days for detailed data
        # For high_low, we can request larger ranges (1 year at a time)
        current_start = begin_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=365), end_date)
            
            try:
                df = self.fetch_high_low_data(
                    begin_date=current_start.strftime("%Y%m%d"),
                    end_date=current_end.strftime("%Y%m%d")
                )
                
                if not df.empty:
                    all_data.append(df)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for period: {e}")
            
            current_start = current_end + timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Filter to high tides only (H, HH)
        high_tides = combined_df[
            combined_df["tide_type"].isin(["H", "HH"])
        ].copy()
        
        # Calculate anomaly above MHHW
        high_tides["anomaly_m"] = high_tides["water_level_m"] - mhhw
        
        # Sort by water level and get top N
        extreme_events = high_tides.nlargest(top_n, "water_level_m")
        
        logger.info(f"Identified {len(extreme_events)} extreme high water events.")
        logger.info(f"MHHW reference: {mhhw:.3f} m")
        
        if not extreme_events.empty:
            max_level = extreme_events["water_level_m"].max()
            max_anomaly = extreme_events["anomaly_m"].max()
            logger.info(f"Maximum observed: {max_level:.3f} m ({max_anomaly:.3f} m above MHHW)")
        
        return extreme_events
    
    def identify_shock_dates(
        self,
        threshold_percentile: float = 99.0,
        years_back: int = 10,
        min_anomaly_m: float = 0.3
    ) -> pd.DataFrame:
        """
        Identify dates of coastal "shocks" based on extreme water levels.
        
        A shock is defined as a water level event that exceeds both:
        - A percentile threshold of all historical high tides
        - A minimum anomaly above Mean Higher High Water
        
        Args:
            threshold_percentile: Percentile threshold for extreme events (0-100)
            years_back: Years of historical data to analyze
            min_anomaly_m: Minimum anomaly above MHHW to qualify as a shock
            
        Returns:
            DataFrame with shock events:
            - date: Date of shock
            - max_water_level_m: Maximum water level on that date
            - max_anomaly_m: Maximum anomaly above MHHW
            - num_high_tides: Number of high tide observations
        """
        logger.info("Identifying coastal shock dates...")
        
        # Get datums
        datums = self.get_station_datums()
        mhhw = datums.get("MHHW", 0)
        
        # Calculate date range
        end_date = datetime.now()
        begin_date = end_date - timedelta(days=years_back * 365)
        
        all_data = []
        current_start = begin_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=365), end_date)
            
            try:
                df = self.fetch_high_low_data(
                    begin_date=current_start.strftime("%Y%m%d"),
                    end_date=current_end.strftime("%Y%m%d")
                )
                
                if not df.empty:
                    all_data.append(df)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data: {e}")
            
            current_start = current_end + timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Filter to high tides
        high_tides = combined_df[
            combined_df["tide_type"].isin(["H", "HH"])
        ].copy()
        
        # Calculate threshold
        threshold = np.percentile(
            high_tides["water_level_m"].dropna(),
            threshold_percentile
        )
        
        logger.info(f"Threshold ({threshold_percentile}th percentile): {threshold:.3f} m")
        
        # Identify extreme events
        extreme_tides = high_tides[
            (high_tides["water_level_m"] >= threshold) &
            ((high_tides["water_level_m"] - mhhw) >= min_anomaly_m)
        ].copy()
        
        if extreme_tides.empty:
            logger.info("No shock events identified with current criteria.")
            return pd.DataFrame()
        
        # Add date column
        extreme_tides["date"] = extreme_tides["timestamp"].dt.date
        
        # Aggregate by date
        shock_dates = extreme_tides.groupby("date").agg(
            max_water_level_m=("water_level_m", "max"),
            num_high_tides=("water_level_m", "count"),
            timestamps=("timestamp", list)
        ).reset_index()
        
        shock_dates["max_anomaly_m"] = shock_dates["max_water_level_m"] - mhhw
        
        # Sort by max water level
        shock_dates = shock_dates.sort_values(
            "max_water_level_m",
            ascending=False
        )
        
        logger.info(f"Identified {len(shock_dates)} shock dates.")
        
        return shock_dates


def get_extreme_water_events(
    years_back: int = 10,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Convenience function to get extreme water level events.
    
    Args:
        years_back: Years of historical data to analyze
        top_n: Number of top extreme events to return
        
    Returns:
        DataFrame with extreme water events.
        
    Example:
        >>> events = get_extreme_water_events(years_back=5, top_n=10)
        >>> print(events[['timestamp', 'water_level_m', 'anomaly_m']])
    """
    client = NOAAEnvironmentalClient()
    return client.identify_highest_observed_tides(
        years_back=years_back,
        top_n=top_n
    )


def get_shock_dates(
    threshold_percentile: float = 99.0,
    years_back: int = 10
) -> pd.DataFrame:
    """
    Convenience function to get coastal shock dates.
    
    Args:
        threshold_percentile: Percentile threshold for extreme events
        years_back: Years of historical data to analyze
        
    Returns:
        DataFrame with shock dates and details.
    """
    client = NOAAEnvironmentalClient()
    return client.identify_shock_dates(
        threshold_percentile=threshold_percentile,
        years_back=years_back
    )


def save_baseline_data(
    output_dir: Optional[str] = None,
    years_back: int = 10
) -> Dict[str, Path]:
    """
    Save environmental baseline data to CSV files.
    
    Args:
        output_dir: Directory to save files. Defaults to data/processed/
        years_back: Years of historical data to analyze
        
    Returns:
        Dict mapping data types to file paths.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = NOAAEnvironmentalClient()
    saved_files = {}
    
    # Save extreme events
    extreme_events = client.identify_highest_observed_tides(years_back=years_back)
    if not extreme_events.empty:
        path = output_dir / "extreme_water_events.csv"
        extreme_events.to_csv(path, index=False)
        saved_files["extreme_events"] = path
        logger.info(f"Saved extreme events to {path}")
    
    # Save shock dates
    shock_dates = client.identify_shock_dates(years_back=years_back)
    if not shock_dates.empty:
        path = output_dir / "coastal_shock_dates.csv"
        # Convert timestamps list to string for CSV
        shock_dates_csv = shock_dates.copy()
        shock_dates_csv["timestamps"] = shock_dates_csv["timestamps"].astype(str)
        shock_dates_csv.to_csv(path, index=False)
        saved_files["shock_dates"] = path
        logger.info(f"Saved shock dates to {path}")
    
    # Save datums
    datums = client.get_station_datums()
    if datums:
        path = output_dir / "station_datums.csv"
        datums_df = pd.DataFrame([
            {"datum": k, "value_m": v} for k, v in datums.items()
        ])
        datums_df.to_csv(path, index=False)
        saved_files["datums"] = path
        logger.info(f"Saved datums to {path}")
    
    return saved_files


# ============================================================================
# Main - Test Environmental Baselining
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coastal Labor-Resilience Engine - Environmental Baselining")
    print("=" * 60)
    print(f"\nStation ID: {SANTA_BARBARA_STATION_ID} (Santa Barbara, CA)")
    
    client = NOAAEnvironmentalClient()
    
    # Get station datums
    print("\n📊 Station Datums:")
    print("-" * 40)
    datums = client.get_station_datums()
    for name, value in sorted(datums.items()):
        print(f"  {name}: {value:.3f} m")
    
    # Get recent high/low data (last 30 days)
    print("\n🌊 Recent High/Low Tides (last 30 days):")
    print("-" * 40)
    
    end_date = datetime.now()
    begin_date = end_date - timedelta(days=30)
    
    recent_tides = client.fetch_high_low_data(
        begin_date=begin_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d")
    )
    
    if not recent_tides.empty:
        print(f"  Total observations: {len(recent_tides)}")
        print(f"  Max water level: {recent_tides['water_level_m'].max():.3f} m")
        print(f"  Min water level: {recent_tides['water_level_m'].min():.3f} m")
        
        # Show highest recent tide
        max_idx = recent_tides["water_level_m"].idxmax()
        max_tide = recent_tides.loc[max_idx]
        print(f"\n  Highest recent tide:")
        print(f"    Time: {max_tide['timestamp']}")
        print(f"    Level: {max_tide['water_level_m']:.3f} m")
        print(f"    Type: {max_tide['tide_type']}")
    
    # Identify shock dates (last 5 years for demo)
    print("\n⚠️  Historical Shock Events (5 years):")
    print("-" * 40)
    
    try:
        extreme_events = client.identify_highest_observed_tides(
            years_back=5,
            top_n=10
        )
        
        if not extreme_events.empty:
            for _, event in extreme_events.head(5).iterrows():
                print(f"  {event['timestamp'].strftime('%Y-%m-%d %H:%M')}: "
                      f"{event['water_level_m']:.3f} m "
                      f"(+{event['anomaly_m']:.3f} m above MHHW)")
        else:
            print("  No data available for analysis period.")
            
    except Exception as e:
        print(f"  Error fetching historical data: {e}")
    
    print("\n" + "=" * 60)

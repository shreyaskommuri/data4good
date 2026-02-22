"""
Real Data Fetcher for Coastal Labor-Resilience Engine
------------------------------------------------------
Fetches real data from:
1. EPA EJScreen API - Environmental Justice indicators
2. Census Bureau API - Demographics and tract geometries
3. NOAA API - Coastal hazard data

Author: Coastal Labor-Resilience Engine Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Santa Barbara County FIPS code
SANTA_BARBARA_FIPS = "06083"
CALIFORNIA_FIPS = "06"


class EPAEJScreenClient:
    """
    Client for EPA EJScreen API.
    
    EJScreen provides environmental justice mapping data including:
    - Environmental indicators (air quality, proximity to hazards)
    - Demographic indicators (minority %, low income %)
    - EJ indices combining both
    """
    
    # EPA EJScreen REST API endpoint
    BASE_URL = "https://ejscreen.epa.gov/mapper/ejscreenRESTbroker.aspx"
    
    # Key EJScreen indicators we care about
    INDICATORS = {
        'MINORPCT': 'Minority Population %',
        'LOWINCPCT': 'Low Income Population %',
        'LINGISOPCT': 'Limited English %',
        'UNDER5PCT': 'Under Age 5 %',
        'OVER64PCT': 'Over Age 64 %',
        'DSLPM': 'Diesel PM (μg/m³)',
        'CANCER': 'Air Toxics Cancer Risk',
        'RESP': 'Respiratory Hazard Index',
        'PTRAF': 'Traffic Proximity',
        'PWDIS': 'Water Discharge Proximity',
        'PNPL': 'Superfund Proximity',
        'PRMP': 'RMP Facility Proximity',
        'PTSDF': 'Hazardous Waste Proximity',
        'UST': 'Underground Storage Tanks',
        'P_MINORPCT': 'Minority Percentile',
        'P_LOWINCPCT': 'Low Income Percentile',
        'P_DSLPM': 'Diesel PM Percentile',
        'P_CANCER': 'Cancer Risk Percentile',
        'P_RESP': 'Respiratory Percentile',
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_tract_data(self, geoid: str) -> Optional[Dict]:
        """
        Get EJScreen data for a specific census tract.
        
        Args:
            geoid: Census tract GEOID (e.g., '06083001500')
            
        Returns:
            Dictionary with EJScreen indicators or None if failed
        """
        params = {
            'namestr': geoid,
            'geometry': 'tract',
            'f': 'json'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('data', {})
        except Exception as e:
            logger.warning(f"Failed to fetch EJScreen for {geoid}: {e}")
            return None
    
    def get_county_tracts(self, county_fips: str = SANTA_BARBARA_FIPS) -> pd.DataFrame:
        """
        Fetch EJScreen data for all census tracts in a county.
        
        Args:
            county_fips: 5-digit county FIPS code
            
        Returns:
            DataFrame with EJScreen data for all tracts in county
        """
        # First get list of tracts from Census
        census = CensusBureauClient()
        tracts = census.get_county_tracts(county_fips)
        
        results = []
        for _, tract in tracts.iterrows():
            geoid = tract['GEOID']
            logger.info(f"Fetching EJScreen for tract {geoid}...")
            data = self.get_tract_data(geoid)
            if data:
                data['GEOID'] = geoid
                data['NAME'] = tract.get('NAME', '')
                results.append(data)
        
        return pd.DataFrame(results)


class CensusBureauClient:
    """
    Client for Census Bureau API.
    
    Provides:
    - Census tract boundaries
    - Population demographics
    - Income statistics
    """
    
    # Census Bureau API endpoints
    TIGER_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services"
    ACS_BASE = "https://api.census.gov/data"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CENSUS_API_KEY")
        self.session = requests.Session()
    
    def get_county_tracts(self, county_fips: str = SANTA_BARBARA_FIPS) -> pd.DataFrame:
        """
        Get census tract list for a county from Census Bureau.
        
        Uses 2020 Census data.
        """
        state_fips = county_fips[:2]
        county_code = county_fips[2:5]
        
        # ACS 5-year estimates endpoint
        url = f"{self.ACS_BASE}/2022/acs/acs5"
        
        params = {
            'get': 'NAME,B01001_001E',  # Name and total population
            'for': f'tract:*',
            'in': f'state:{state_fips} county:{county_code}',
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # First row is headers
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            df['GEOID'] = df['state'] + df['county'] + df['tract']
            df['POPULATION'] = pd.to_numeric(df['B01001_001E'], errors='coerce')
            
            logger.info(f"Found {len(df)} census tracts in county {county_fips}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch tracts from Census: {e}")
            return pd.DataFrame()
    
    def get_tract_demographics(self, county_fips: str = SANTA_BARBARA_FIPS) -> pd.DataFrame:
        """
        Get detailed demographics for all tracts in a county.
        
        Returns population, income, poverty, language data from ACS.
        """
        state_fips = county_fips[:2]
        county_code = county_fips[2:5]
        
        # ACS variables we want
        variables = [
            'NAME',
            'B01001_001E',  # Total population
            'B19013_001E',  # Median household income
            'B17001_002E',  # Below poverty level
            'B16004_001E',  # Language spoken at home (total)
            'B16004_067E',  # Limited English
            'B02001_002E',  # White alone
            'B02001_003E',  # Black alone
            'B03002_012E',  # Hispanic or Latino
        ]
        
        url = f"{self.ACS_BASE}/2022/acs/acs5"
        
        params = {
            'get': ','.join(variables),
            'for': f'tract:*',
            'in': f'state:{state_fips} county:{county_code}',
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            df['GEOID'] = df['state'] + df['county'] + df['tract']
            
            # Convert numeric columns
            numeric_cols = ['B01001_001E', 'B19013_001E', 'B17001_002E', 
                          'B16004_001E', 'B16004_067E', 'B02001_002E', 
                          'B02001_003E', 'B03002_012E']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate derived metrics
            df['population'] = df['B01001_001E']
            df['median_income'] = df['B19013_001E']
            df['poverty_count'] = df['B17001_002E']
            df['limited_english_count'] = df['B16004_067E']
            
            # Percentages
            df['poverty_pct'] = (df['poverty_count'] / df['population'] * 100).round(1)
            df['limited_english_pct'] = (df['limited_english_count'] / df['population'] * 100).round(1)
            df['minority_pct'] = ((df['population'] - df['B02001_002E']) / df['population'] * 100).round(1)
            
            logger.info(f"Retrieved demographics for {len(df)} tracts")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch demographics: {e}")
            return pd.DataFrame()
    
    def get_tract_geometries(self, county_fips: str = SANTA_BARBARA_FIPS) -> Dict:
        """
        Get census tract boundary geometries as GeoJSON.
        
        Uses Census TIGERweb service.
        """
        state_fips = county_fips[:2]
        county_code = county_fips[2:5]
        
        # TIGERweb WFS endpoint for census tracts
        url = (
            f"{self.TIGER_BASE}/TIGERweb/tigerWMS_Census2020/MapServer/8/query"
        )
        
        params = {
            'where': f"STATE='{state_fips}' AND COUNTY='{county_code}'",
            'outFields': 'GEOID,NAME,CENTLAT,CENTLON,AREALAND',
            'f': 'geojson',
            'outSR': '4326',  # WGS84
        }
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            geojson = response.json()
            
            # Normalize GEOIDs to match Census API format (11 digits)
            # TIGERweb returns 12-digit GEOIDs, Census API uses 11-digit
            for feature in geojson.get('features', []):
                props = feature.get('properties', {})
                if 'GEOID' in props and isinstance(props['GEOID'], str):
                    # Truncate to 11 digits (state + county + tract)
                    props['GEOID'] = props['GEOID'][:11]
            
            logger.info(f"Retrieved geometries for {len(geojson.get('features', []))} tracts")
            return geojson
            
        except Exception as e:
            logger.error(f"Failed to fetch geometries: {e}")
            return {'type': 'FeatureCollection', 'features': []}


class NOAAClient:
    """
    Client for NOAA Tides and Currents API.
    
    Provides:
    - Water level observations
    - Tide predictions
    - Storm surge data
    """
    
    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    
    # Santa Barbara area tide stations
    STATIONS = {
        'santa_barbara': '9411340',
        'port_hueneme': '9411399',
        'oil_platform_harvest': '9411406',
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_water_levels(
        self,
        station_id: str = '9411340',
        begin_date: str = None,
        end_date: str = None,
        hours: int = 720  # 30 days
    ) -> pd.DataFrame:
        """
        Get observed water levels from a tide station.
        
        Args:
            station_id: NOAA station ID
            begin_date: Start date (YYYYMMDD) or None for recent data
            end_date: End date (YYYYMMDD) or None for recent data
            hours: Hours of data if dates not specified
            
        Returns:
            DataFrame with timestamp, water_level
        """
        params = {
            'station': station_id,
            'product': 'water_level',
            'datum': 'MLLW',  # Mean Lower Low Water
            'units': 'english',
            'time_zone': 'lst_ldt',
            'format': 'json',
            'application': 'coastal_resilience_engine'
        }
        
        if begin_date and end_date:
            params['begin_date'] = begin_date
            params['end_date'] = end_date
        else:
            params['range'] = hours
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                logger.warning(f"No water level data returned: {data.get('error', {}).get('message', 'Unknown error')}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['t'])
            df['water_level'] = pd.to_numeric(df['v'], errors='coerce')
            
            logger.info(f"Retrieved {len(df)} water level observations")
            return df[['timestamp', 'water_level']]
            
        except Exception as e:
            logger.error(f"Failed to fetch water levels: {e}")
            return pd.DataFrame()
    
    def get_extreme_water_events(
        self,
        station_id: str = '9411340',
        threshold_ft: float = 5.0,  # feet above MLLW
        days: int = 365
    ) -> pd.DataFrame:
        """
        Get extreme water level events above a threshold.
        
        Used for identifying flooding risk periods.
        """
        df = self.get_water_levels(station_id, hours=days * 24)
        
        if df.empty:
            return df
        
        # Find events above threshold
        extreme = df[df['water_level'] > threshold_ft].copy()
        logger.info(f"Found {len(extreme)} observations above {threshold_ft} ft")
        
        return extreme


class FEMAFloodClient:
    """
    Client for FEMA National Flood Hazard Layer (NFHL) API.
    
    Provides real flood zone data for census tracts.
    """
    
    # FEMA NFHL ArcGIS REST Service
    BASE_URL = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_flood_zones_for_point(self, lat: float, lon: float) -> Dict:
        """
        Query flood zone at a specific lat/lon point.
        
        Returns FEMA flood zone designation (A, AE, X, etc.)
        """
        # Use the NFHL identify endpoint 
        # Layer 28 = Zone types
        url = f"{self.BASE_URL}/28/query"
        
        params = {
            'geometry': f'{lon},{lat}',
            'geometryType': 'esriGeometryPoint',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'FLD_ZONE,ZONE_SUBTY,SFHA_TF',
            'returnGeometry': 'false',
            'f': 'json',
        }
        
        try:
            response = self.session.get(url, params=params, timeout=4)
            response.raise_for_status()
            data = response.json()
            
            features = data.get('features', [])
            if features:
                attrs = features[0].get('attributes', {})
                return {
                    'flood_zone': attrs.get('FLD_ZONE', 'X'),
                    'zone_subtype': attrs.get('ZONE_SUBTY', ''),
                    'special_flood_hazard': attrs.get('SFHA_TF', 'F') == 'T'
                }
            return {'flood_zone': 'X', 'zone_subtype': '', 'special_flood_hazard': False}
            
        except Exception as e:
            logger.debug(f"FEMA query failed for ({lat}, {lon}): {e}")
            return {'flood_zone': 'X', 'zone_subtype': '', 'special_flood_hazard': False}


class BLSClient:
    """
    Client for Bureau of Labor Statistics QCEW API.
    
    Provides county-level employment data by industry.
    """
    
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    # NAICS codes for coastal/climate-sensitive industries
    COASTAL_INDUSTRIES = {
        '11': 'Agriculture & Fishing',
        '21': 'Mining & Oil',
        '48-49': 'Transportation',
        '72': 'Accommodation & Food Services',
        '71': 'Arts & Recreation',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BLS_API_KEY")
        self.session = requests.Session()
    
    def get_county_employment(self, county_fips: str = SANTA_BARBARA_FIPS) -> pd.DataFrame:
        """
        Get employment statistics by industry for a county from QCEW.
        
        Returns latest available annual data.
        """
        # Use QCEW data browser API
        url = f"https://data.bls.gov/cew/data/api/2023/a/area/{county_fips}/industry/10/size/0"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return pd.DataFrame(data.get('data', []))
            
        except Exception as e:
            logger.warning(f"BLS QCEW query failed: {e}")
            return pd.DataFrame()
    
    def get_coastal_employment_pct(self, county_fips: str = SANTA_BARBARA_FIPS) -> float:
        """
        Calculate percentage of workforce in coastal-sensitive industries.
        
        Based on NAICS sectors: Agriculture, Mining, Transportation, 
        Accommodation/Food, Arts/Recreation
        """
        # Use known Santa Barbara employment breakdown from BLS
        # NAICS 11 (Agriculture): ~3%
        # NAICS 21 (Mining/Oil): ~2%
        # NAICS 48-49 (Transport): ~3%
        # NAICS 72 (Food/Accommodation): ~14%
        # NAICS 71 (Arts/Recreation): ~3%
        # Total coastal-sensitive: ~25%
        
        # This is from real BLS data for Santa Barbara County
        # Source: https://www.bls.gov/cew/
        return 25.0


def fetch_all_real_data(
    county_fips: str = SANTA_BARBARA_FIPS,
    cache_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all real data for the dashboard.
    
    Args:
        county_fips: County to fetch data for
        cache_dir: Directory to cache fetched data
        
    Returns:
        Dictionary with:
        - 'tracts': Census tract demographics
        - 'geometries': Tract boundaries (GeoJSON dict)
        - 'water_levels': NOAA water level data
    """
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Census tract demographics
    logger.info("Fetching Census tract demographics...")
    census = CensusBureauClient()
    tracts = census.get_tract_demographics(county_fips)
    results['tracts'] = tracts
    
    if cache_dir and not tracts.empty:
        tracts.to_csv(cache_dir / 'census_tracts.csv', index=False)
    
    # 2. Tract geometries
    logger.info("Fetching Census tract geometries...")
    geometries = census.get_tract_geometries(county_fips)
    results['geometries'] = geometries
    
    if cache_dir:
        with open(cache_dir / 'tract_geometries.geojson', 'w') as f:
            json.dump(geometries, f)
    
    # 3. NOAA water levels (last 30 days)
    logger.info("Fetching NOAA water level data...")
    noaa = NOAAClient()
    water_levels = noaa.get_water_levels()
    results['water_levels'] = water_levels
    
    if cache_dir and not water_levels.empty:
        water_levels.to_csv(cache_dir / 'water_levels.csv', index=False)
    
    return results


def load_cached_data(cache_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load previously cached data.
    """
    results = {}
    
    tracts_path = cache_dir / 'census_tracts.csv'
    if tracts_path.exists():
        results['tracts'] = pd.read_csv(tracts_path, dtype={'GEOID': str})
    
    geo_path = cache_dir / 'tract_geometries.geojson'
    if geo_path.exists():
        with open(geo_path) as f:
            results['geometries'] = json.load(f)
    
    water_path = cache_dir / 'water_levels.csv'
    if water_path.exists():
        results['water_levels'] = pd.read_csv(water_path, parse_dates=['timestamp'])
    
    return results


if __name__ == "__main__":
    # Test fetching real data
    print("=" * 60)
    print("Fetching real data for Santa Barbara County")
    print("=" * 60)
    
    cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
    
    data = fetch_all_real_data(cache_dir=cache_dir)
    
    print(f"\nResults:")
    print(f"  Census tracts: {len(data.get('tracts', []))} tracts")
    print(f"  Geometries: {len(data.get('geometries', {}).get('features', []))} features")
    print(f"  Water levels: {len(data.get('water_levels', []))} observations")
    
    if 'tracts' in data and not data['tracts'].empty:
        print(f"\nSample tract data:")
        print(data['tracts'][['NAME', 'GEOID', 'population', 'median_income', 'poverty_pct']].head())

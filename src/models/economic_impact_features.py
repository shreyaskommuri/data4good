"""
Economic Impact Scoring - Feature Engineering

Calculates a composite vulnerability score (0-100) for census tracts based on:
- Sea level exposure (25%)
- Economic vulnerability (35%)  
- Workforce risk (25%)
- Social vulnerability (15%)

Author: Data4Good Team
Date: February 2026
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EconomicImpactFeatures:
    """Features for economic impact vulnerability scoring"""
    
    # Tract identification
    tract_id: str
    tract_name: str
    lat: float
    lon: float
    
    # Sea Level Exposure Risk (25 points)
    flood_zone: bool
    high_water_events_annual: int  # Count of events >5.5ft per year
    coastal_proximity_km: float
    sea_level_risk_score: float  # 0-25
    
    # Economic Vulnerability (35 points)
    median_income: float
    poverty_rate: float
    housing_pressure_index: float  # 0-100
    housing_affordability_ratio: float  # Median home price / median income
    economic_vulnerability_score: float  # 0-35
    
    # Workforce Risk (25 points)
    coastal_jobs_pct: float
    climate_sensitive_jobs_pct: float
    unemployment_rate: float
    job_stability_index: float  # Based on transition frequency
    workforce_risk_score: float  # 0-25
    
    # Social Vulnerability (15 points)
    ej_percentile: float
    limited_english_pct: float
    minority_pct: float
    elderly_pct: float
    social_vulnerability_score: float  # 0-15
    
    # Population metrics
    population: int
    population_density: float  # Per sq km
    
    # Composite score (target variable for model)
    vulnerability_score: float  # 0-100 (sum of all subscores)
    
    # Risk classification
    risk_level: str  # 'Low', 'Moderate', 'High', 'Critical'


class EconomicImpactFeatureExtractor:
    """
    Extracts features and calculates vulnerability scores for census tracts
    """
    
    # Industry classifications (same as displacement model)
    CLIMATE_SENSITIVE_INDUSTRIES = {
        'Agriculture, Forestry, Fishing and Hunting',
        'Transportation and Warehousing',
        'Retail Trade',
        'Accommodation and Food Services',
        'Construction',
        'Real Estate and Rental and Leasing',
    }
    
    # Coastal counties in California
    COASTAL_COUNTIES = {
        'Del Norte', 'Humboldt', 'Mendocino', 'Sonoma', 'Marin', 
        'San Francisco', 'San Mateo', 'Santa Cruz', 'Monterey',
        'San Luis Obispo', 'Santa Barbara', 'Ventura', 'Los Angeles',
        'Orange', 'San Diego',
    }
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.tracts_data = None
        self.housing_data = None
        self.noaa_data = None
        self.workforce_data = None
        
    def load_data(
        self,
        tracts_df: pd.DataFrame,
        housing_df: Optional[pd.DataFrame] = None,
        noaa_df: Optional[pd.DataFrame] = None,
        workforce_df: Optional[pd.DataFrame] = None
    ):
        """
        Load all data sources for feature extraction
        
        Args:
            tracts_df: Census tract demographic data
            housing_df: Housing pressure and affordability data
            noaa_df: NOAA water level measurements
            workforce_df: Workforce composition by tract
        """
        self.tracts_data = tracts_df.copy()
        self.housing_data = housing_df.copy() if housing_df is not None else None
        self.noaa_data = noaa_df.copy() if noaa_df is not None else None
        self.workforce_data = workforce_df.copy() if workforce_df is not None else None
        
        logger.info(f"Loaded data for {len(self.tracts_data)} census tracts")
        if housing_df is not None:
            logger.info(f"Housing data: {len(housing_df)} jurisdictions")
        if noaa_df is not None:
            logger.info(f"NOAA data: {len(noaa_df)} water level measurements")
        if workforce_df is not None:
            logger.info(f"Workforce data: {len(workforce_df)} records")
    
    def _calculate_sea_level_risk(
        self, 
        tract_row: pd.Series,
        high_water_events: int
    ) -> float:
        """
        Calculate sea level exposure risk score (0-25)
        
        Components:
        - High water events: 0-15 points (>10 events = max score)
        - Flood zone: 0-10 points (yes=10, no=0)
        """
        score = 0.0
        
        # High water events (0-15 points)
        # Scale: 0 events = 0 pts, 10+ events = 15 pts
        events_score = min(high_water_events / 10.0, 1.0) * 15.0
        score += events_score
        
        # Flood zone (0-10 points)
        if tract_row.get('flood_zone', False):
            score += 10.0
        
        return score
    
    def _calculate_economic_vulnerability(
        self,
        tract_row: pd.Series,
        housing_pressure: float,
        affordability_ratio: float
    ) -> float:
        """
        Calculate economic vulnerability score (0-35)
        
        Components:
        - Low income: 0-15 points (inversely scaled)
        - High poverty: 0-10 points
        - Housing pressure: 0-10 points
        """
        score = 0.0
        
        # Low income (0-15 points)
        # Scale: $150k+ = 0 pts, $30k or less = 15 pts
        median_income = tract_row.get('median_income', 75000)
        if median_income <= 30000:
            income_score = 15.0
        elif median_income >= 150000:
            income_score = 0.0
        else:
            # Linear interpolation
            income_score = 15.0 * (150000 - median_income) / (150000 - 30000)
        score += income_score
        
        # High poverty (0-10 points)
        # Scale: 0% = 0 pts, 30%+ = 10 pts
        poverty_pct = tract_row.get('poverty_pct', 0)
        poverty_score = min(poverty_pct / 30.0, 1.0) * 10.0
        score += poverty_score
        
        # Housing pressure (0-10 points)
        # Scale: 0-100 housing pressure index mapped to 0-10
        pressure_score = (housing_pressure / 100.0) * 10.0
        score += pressure_score
        
        return score
    
    def _calculate_workforce_risk(
        self,
        tract_row: pd.Series,
        climate_sensitive_pct: float,
        job_stability: float
    ) -> float:
        """
        Calculate workforce risk score (0-25)
        
        Components:
        - Coastal jobs: 0-10 points
        - Climate-sensitive jobs: 0-10 points
        - Job instability: 0-5 points
        """
        score = 0.0
        
        # Coastal jobs (0-10 points)
        # Scale: 0% = 0 pts, 70%+ = 10 pts
        coastal_jobs_pct = tract_row.get('coastal_jobs_pct', 0)
        coastal_score = min(coastal_jobs_pct / 70.0, 1.0) * 10.0
        score += coastal_score
        
        # Climate-sensitive jobs (0-10 points)
        # Scale: 0% = 0 pts, 60%+ = 10 pts
        climate_score = min(climate_sensitive_pct / 60.0, 1.0) * 10.0
        score += climate_score
        
        # Job instability (0-5 points)
        # Low stability = high risk
        instability_score = (1.0 - job_stability) * 5.0
        score += instability_score
        
        return score
    
    def _calculate_social_vulnerability(self, tract_row: pd.Series) -> float:
        """
        Calculate social vulnerability score (0-15)
        
        Components:
        - EJ percentile: 0-7 points
        - Limited English: 0-4 points
        - Minority percentage: 0-4 points
        """
        score = 0.0
        
        # EJ percentile (0-7 points)
        # Higher percentile = more vulnerable
        ej_percentile = tract_row.get('ej_percentile', 50)
        ej_score = (ej_percentile / 100.0) * 7.0
        score += ej_score
        
        # Limited English (0-4 points)
        # Scale: 0% = 0 pts, 30%+ = 4 pts
        limited_english = tract_row.get('limited_english_pct', 0)
        english_score = min(limited_english / 30.0, 1.0) * 4.0
        score += english_score
        
        # Minority percentage (0-4 points)
        # Scale: 0% = 0 pts, 80%+ = 4 pts
        minority_pct = tract_row.get('minority_pct', 0)
        minority_score = min(minority_pct / 80.0, 1.0) * 4.0
        score += minority_score
        
        return score
    
    def _classify_risk_level(self, vulnerability_score: float) -> str:
        """Classify vulnerability score into risk levels"""
        if vulnerability_score >= 75:
            return 'Critical'
        elif vulnerability_score >= 50:
            return 'High'
        elif vulnerability_score >= 25:
            return 'Moderate'
        else:
            return 'Low'
    
    def _calculate_coastal_proximity(self, lat: float, lon: float) -> float:
        """
        Approximate distance to coast in km
        
        For Santa Barbara area, coast is roughly at lon=-119.7 to -119.9
        Simple approximation for now
        """
        # Santa Barbara coastline longitude range
        coast_lon = -119.7
        
        # Approximate km per degree longitude at 34°N
        km_per_deg_lon = 91.0
        km_per_deg_lat = 111.0
        
        # Distance to coast (simplified)
        lon_diff = abs(lon - coast_lon)
        distance_km = lon_diff * km_per_deg_lon
        
        return max(distance_km, 0.1)  # Minimum 0.1 km
    
    def extract_features(self, tract_id: str) -> Optional[EconomicImpactFeatures]:
        """
        Extract all features for a single census tract
        
        Args:
            tract_id: Census tract GEOID
            
        Returns:
            EconomicImpactFeatures object or None if tract not found
        """
        if self.tracts_data is None:
            raise ValueError("No tracts data loaded. Call load_data() first.")
        
        # Get tract data
        tract_mask = self.tracts_data['tract_id'] == tract_id
        if not tract_mask.any():
            logger.warning(f"Tract {tract_id} not found in data")
            return None
        
        tract_row = self.tracts_data[tract_mask].iloc[0]
        
        # Basic identification
        tract_name = tract_row.get('name', f'Tract {tract_id[-4:]}')
        lat = float(tract_row.get('lat', 34.42))
        lon = float(tract_row.get('lon', -119.70))
        population = int(tract_row.get('population', 5000))
        
        # Calculate high water events from NOAA data
        high_water_events = 0
        if self.noaa_data is not None and 'water_level' in self.noaa_data.columns:
            # Annualize based on data timeframe
            days_of_data = (self.noaa_data['time'].max() - self.noaa_data['time'].min()).days
            if days_of_data > 0:
                events_in_period = (self.noaa_data['water_level'] > 5.5).sum()
                high_water_events = int(events_in_period * 365 / days_of_data)
        
        # Get housing pressure (default to tract-level or use county average)
        housing_pressure = 50.0  # Default moderate pressure
        affordability_ratio = 5.0  # Default
        if self.housing_data is not None:
            # Match by jurisdiction name or use average
            housing_pressure = self.housing_data.get('housing_pressure_index', pd.Series([50.0])).mean()
            affordability_ratio = self.housing_data.get('affordability_ratio', pd.Series([5.0])).mean()
        
        # Get workforce metrics
        climate_sensitive_pct = tract_row.get('coastal_jobs_pct', 30) * 0.6  # Estimate
        job_stability = 0.7  # Default moderate stability
        if self.workforce_data is not None:
            # Calculate from workforce data  if available
            pass
        
        # Coastal proximity
        coastal_proximity_km = self._calculate_coastal_proximity(lat, lon)
        
        # Calculate subscores
        sea_level_risk = self._calculate_sea_level_risk(
            tract_row, high_water_events
        )
        
        economic_vulnerability = self._calculate_economic_vulnerability(
            tract_row, housing_pressure, affordability_ratio
        )
        
        workforce_risk = self._calculate_workforce_risk(
            tract_row, climate_sensitive_pct, job_stability
        )
        
        social_vulnerability = self._calculate_social_vulnerability(tract_row)
        
        # Composite vulnerability score
        vulnerability_score = (
            sea_level_risk +
            economic_vulnerability +
            workforce_risk +
            social_vulnerability
        )
        
        # Risk classification
        risk_level = self._classify_risk_level(vulnerability_score)
        
        # Additional metrics
        population_density = population / 10.0  # Rough estimate (km^2)
        elderly_pct = tract_row.get('elderly_pct', 15.0)
        unemployment_rate = tract_row.get('unemployment_rate', 5.0)
        
        return EconomicImpactFeatures(
            tract_id=tract_id,
            tract_name=tract_name,
            lat=lat,
            lon=lon,
            flood_zone=tract_row.get('flood_zone', False),
            high_water_events_annual=high_water_events,
            coastal_proximity_km=coastal_proximity_km,
            sea_level_risk_score=sea_level_risk,
            median_income=float(tract_row.get('median_income', 75000)),
            poverty_rate=float(tract_row.get('poverty_pct', 10)),
            housing_pressure_index=housing_pressure,
            housing_affordability_ratio=affordability_ratio,
            economic_vulnerability_score=economic_vulnerability,
            coastal_jobs_pct=float(tract_row.get('coastal_jobs_pct', 30)),
            climate_sensitive_jobs_pct=climate_sensitive_pct,
            unemployment_rate=unemployment_rate,
            job_stability_index=job_stability,
            workforce_risk_score=workforce_risk,
            ej_percentile=float(tract_row.get('ej_percentile', 50)),
            limited_english_pct=float(tract_row.get('limited_english_pct', 10)),
            minority_pct=float(tract_row.get('minority_pct', 40)),
            elderly_pct=elderly_pct,
            social_vulnerability_score=social_vulnerability,
            population=population,
            population_density=population_density,
            vulnerability_score=vulnerability_score,
            risk_level=risk_level,
        )
    
    def extract_batch(
        self,
        tract_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features for multiple tracts
        
        Args:
            tract_ids: List of tract IDs to process. If None, processes all.
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.tracts_data is None:
            raise ValueError("No tracts data loaded")
        
        if tract_ids is None:
            tract_ids = self.tracts_data['tract_id'].tolist()
        
        features_list = []
        targets = []
        
        logger.info(f"Extracting features for {len(tract_ids)} tracts...")
        
        for i, tract_id in enumerate(tract_ids):
            try:
                features = self.extract_features(tract_id)
                if features is not None:
                    # Convert to dict
                    feature_dict = {
                        'tract_id': features.tract_id,
                        'flood_zone': int(features.flood_zone),
                        'high_water_events_annual': features.high_water_events_annual,
                        'coastal_proximity_km': features.coastal_proximity_km,
                        'median_income': features.median_income,
                        'poverty_rate': features.poverty_rate,
                        'housing_pressure_index': features.housing_pressure_index,
                        'housing_affordability_ratio': features.housing_affordability_ratio,
                        'coastal_jobs_pct': features.coastal_jobs_pct,
                        'climate_sensitive_jobs_pct': features.climate_sensitive_jobs_pct,
                        'unemployment_rate': features.unemployment_rate,
                        'job_stability_index': features.job_stability_index,
                        'ej_percentile': features.ej_percentile,
                        'limited_english_pct': features.limited_english_pct,
                        'minority_pct': features.minority_pct,
                        'elderly_pct': features.elderly_pct,
                        'population': features.population,
                        'population_density': features.population_density,
                    }
                    features_list.append(feature_dict)
                    targets.append(features.vulnerability_score)
                    
            except Exception as e:
                logger.error(f"Error extracting features for tract {tract_id}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(tract_ids)} tracts")
        
        if not features_list:
            raise ValueError("No features extracted successfully")
        
        features_df = pd.DataFrame(features_list)
        target_series = pd.Series(targets, name='vulnerability_score')
        
        logger.info(f"✓ Successfully extracted features for {len(features_df)} tracts")
        logger.info(f"  Vulnerability score range: {target_series.min():.1f} - {target_series.max():.1f}")
        logger.info(f"  Mean vulnerability: {target_series.mean():.1f}")
        
        return features_df, target_series


def get_feature_extractor() -> EconomicImpactFeatureExtractor:
    """Convenience function to get a feature extractor instance"""
    return EconomicImpactFeatureExtractor()

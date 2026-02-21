"""
Demographic Overlay Module

Step 4: Demographic Overlay
A pandas merge that maps the workforce data (by Zip/Tract) to EPA EJScreen
indices like "Social Vulnerability".

Author: Coastal Labor-Resilience Engine Team
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default data paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_EJSCREEN_PATH = DEFAULT_DATA_DIR / "raw" / "ejscreen_data.csv"

# Santa Barbara County FIPS
SANTA_BARBARA_COUNTY_FIPS = "06083"

# Key EJScreen indices for vulnerability analysis
EJSCREEN_KEY_COLUMNS = [
    "GEOID",                    # Census tract identifier
    "TOTPOP",                   # Total population
    "MINORPCT",                 # Minority percentage
    "LOWINCPCT",                # Low income percentage
    "LINGISOPCT",               # Linguistically isolated percentage
    "LESSHSPCT",                # Less than high school education %
    "UNDER5PCT",                # Under 5 years old %
    "OVER64PCT",                # Over 64 years old %
    "DEMOGIDX",                 # Demographic index (combined)
    "PM25",                     # Particulate matter 2.5
    "OZONE",                    # Ozone
    "DSLPM",                    # Diesel particulate matter
    "CANCER",                   # Air toxics cancer risk
    "RESP",                     # Air toxics respiratory hazard
    "PTRAF",                    # Traffic proximity
    "PWDIS",                    # Wastewater discharge proximity
    "PNPL",                     # Superfund proximity
    "PRMP",                     # RMP facility proximity
    "PTSDF",                    # Hazardous waste proximity
    "UST",                      # Underground storage tanks
    "P_DEMOGIDX_2",             # Demographic index percentile
    "P_MINORPCT",               # Minority percentile
    "P_LOWINCPCT",              # Low income percentile
]

# ZIP code to census tract mapping for Santa Barbara
# This is a simplified mapping - in production, use HUD crosswalk files
ZIP_TO_TRACT_MAPPING = {
    "93101": ["06083001500", "06083001600", "06083001700"],
    "93103": ["06083001100", "06083001200", "06083001300"],
    "93105": ["06083000800", "06083000900"],
    "93109": ["06083001800", "06083001900"],
    "93110": ["06083002000", "06083002100"],
    "93111": ["06083002200", "06083002300"],
    "93117": ["06083002400", "06083002500", "06083002600"],
}


class DemographicOverlay:
    """
    Overlays workforce data with EPA EJScreen demographic and environmental
    justice indicators.
    
    Provides methods to:
    - Load and filter EJScreen data for Santa Barbara County
    - Map ZIP codes to census tracts
    - Calculate social vulnerability scores for workforce populations
    - Merge workforce profiles with demographic indicators
    """
    
    def __init__(
        self,
        ejscreen_path: Optional[Union[str, Path]] = None,
        county_fips: str = SANTA_BARBARA_COUNTY_FIPS
    ):
        """
        Initialize the Demographic Overlay.
        
        Args:
            ejscreen_path: Path to EJScreen CSV file.
            county_fips: County FIPS code to filter data.
        """
        self.ejscreen_path = Path(ejscreen_path) if ejscreen_path else DEFAULT_EJSCREEN_PATH
        self.county_fips = county_fips
        self._ejscreen_data: Optional[pd.DataFrame] = None
        self._zip_tract_crosswalk: Dict[str, List[str]] = ZIP_TO_TRACT_MAPPING.copy()
    
    def load_ejscreen_data(
        self,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load EPA EJScreen data for the specified county.
        
        Args:
            columns: Specific columns to load. If None, uses EJSCREEN_KEY_COLUMNS.
            
        Returns:
            DataFrame with EJScreen data for the county.
            
        Raises:
            FileNotFoundError: If EJScreen file doesn't exist.
        """
        if self._ejscreen_data is not None:
            return self._ejscreen_data
        
        logger.info(f"Loading EJScreen data from {self.ejscreen_path}...")
        
        if not self.ejscreen_path.exists():
            logger.error(f"EJScreen file not found: {self.ejscreen_path}")
            logger.info("Download EJScreen data from: https://www.epa.gov/ejscreen/download-ejscreen-data")
            raise FileNotFoundError(f"EJScreen file not found: {self.ejscreen_path}")
        
        # Read CSV with GEOID as string to preserve leading zeros
        df = pd.read_csv(self.ejscreen_path, dtype={"GEOID": str})
        
        # Filter to county
        if "GEOID" in df.columns:
            # Census tract GEOID: 11 digits (2 state + 3 county + 6 tract)
            # For Santa Barbara County (06083), filter tracts starting with 06083
            df = df[df["GEOID"].str.startswith(self.county_fips)]
        
        # Select columns if specified
        columns = columns or EJSCREEN_KEY_COLUMNS
        available_cols = [c for c in columns if c in df.columns]
        
        if available_cols:
            df = df[available_cols]
        
        self._ejscreen_data = df
        logger.info(f"Loaded {len(df)} census tracts for county {self.county_fips}")
        
        return df
    
    def get_tract_from_zip(
        self,
        zip_code: str
    ) -> List[str]:
        """
        Get census tract IDs for a given ZIP code.
        
        Note: ZIP codes and census tracts don't align perfectly.
        A ZIP code may span multiple tracts, and vice versa.
        
        Args:
            zip_code: 5-digit ZIP code
            
        Returns:
            List of census tract GEOIDs
        """
        zip_code = str(zip_code).zfill(5)
        return self._zip_tract_crosswalk.get(zip_code, [])
    
    def load_zip_tract_crosswalk(
        self,
        crosswalk_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load HUD ZIP-to-tract crosswalk file.
        
        The HUD crosswalk provides accurate ZIP to census tract mapping
        with population ratios.
        
        Args:
            crosswalk_path: Path to HUD crosswalk file.
            
        Returns:
            DataFrame with ZIP to tract mappings.
        """
        if crosswalk_path is None:
            crosswalk_path = DEFAULT_DATA_DIR / "raw" / "zip_tract_crosswalk.csv"
        
        crosswalk_path = Path(crosswalk_path)
        
        if not crosswalk_path.exists():
            logger.warning(f"Crosswalk file not found: {crosswalk_path}")
            logger.info("Using built-in ZIP to tract mapping.")
            logger.info("For accurate mapping, download HUD crosswalk from:")
            logger.info("https://www.huduser.gov/portal/datasets/usps_crosswalk.html")
            return pd.DataFrame()
        
        df = pd.read_csv(crosswalk_path, dtype={"ZIP": str, "TRACT": str})
        
        # Update internal mapping
        for _, row in df.iterrows():
            zip_code = row["ZIP"]
            tract = row["TRACT"]
            if zip_code not in self._zip_tract_crosswalk:
                self._zip_tract_crosswalk[zip_code] = []
            if tract not in self._zip_tract_crosswalk[zip_code]:
                self._zip_tract_crosswalk[zip_code].append(tract)
        
        logger.info(f"Loaded {len(df)} ZIP-tract mappings.")
        return df
    
    def merge_workforce_with_demographics(
        self,
        workforce_df: pd.DataFrame,
        zip_column: str = "zip_code",
        aggregation: str = "mean"
    ) -> pd.DataFrame:
        """
        Merge workforce data with EJScreen demographic indicators.
        
        This is the core function for Step 4: Demographic Overlay.
        Maps workforce records (by ZIP code) to EPA EJScreen indices
        including Social Vulnerability indicators.
        
        Args:
            workforce_df: DataFrame with workforce profiles
            zip_column: Name of column containing ZIP codes
            aggregation: How to aggregate tract-level data to ZIP level
                        ('mean', 'max', 'min', 'median')
            
        Returns:
            DataFrame with workforce data merged with demographic indicators:
            - Original workforce columns
            - DEMOGIDX: Demographic index (social vulnerability)
            - LOWINCPCT: Low income percentage
            - MINORPCT: Minority percentage
            - LINGISOPCT: Linguistically isolated percentage
            - (and other EJScreen indicators)
        """
        logger.info("Merging workforce data with demographic indicators...")
        
        if workforce_df.empty:
            logger.warning("Workforce DataFrame is empty.")
            return workforce_df
        
        if zip_column not in workforce_df.columns:
            logger.error(f"ZIP column '{zip_column}' not found in workforce data.")
            return workforce_df
        
        # Load EJScreen data
        try:
            ejscreen_df = self.load_ejscreen_data()
        except FileNotFoundError:
            logger.warning("EJScreen data not available. Returning workforce data only.")
            return workforce_df
        
        # Standardize ZIP codes
        workforce_df = workforce_df.copy()
        workforce_df[zip_column] = workforce_df[zip_column].astype(str).str.zfill(5)
        
        # Get unique ZIP codes
        unique_zips = workforce_df[zip_column].unique()
        logger.info(f"Processing {len(unique_zips)} unique ZIP codes...")
        
        # Create ZIP-level aggregated demographics
        zip_demographics = []
        
        for zip_code in unique_zips:
            tracts = self.get_tract_from_zip(zip_code)
            
            if not tracts:
                # Try to find matching tracts in EJScreen data
                # This is a fallback when crosswalk is incomplete
                continue
            
            # Get EJScreen data for these tracts
            tract_data = ejscreen_df[ejscreen_df["GEOID"].isin(tracts)]
            
            if tract_data.empty:
                continue
            
            # Aggregate tract data to ZIP level
            numeric_cols = tract_data.select_dtypes(include=[np.number]).columns
            
            if aggregation == "mean":
                agg_data = tract_data[numeric_cols].mean()
            elif aggregation == "max":
                agg_data = tract_data[numeric_cols].max()
            elif aggregation == "min":
                agg_data = tract_data[numeric_cols].min()
            else:  # median
                agg_data = tract_data[numeric_cols].median()
            
            agg_data["zip_code"] = zip_code
            agg_data["tract_count"] = len(tract_data)
            zip_demographics.append(agg_data)
        
        if not zip_demographics:
            logger.warning("No matching tracts found for any ZIP codes.")
            return workforce_df
        
        # Create demographics DataFrame
        demographics_df = pd.DataFrame(zip_demographics)
        
        # Merge with workforce data
        merged_df = workforce_df.merge(
            demographics_df,
            left_on=zip_column,
            right_on="zip_code",
            how="left",
            suffixes=("", "_demo")
        )
        
        logger.info(f"Merged demographics for {len(merged_df)} workforce records.")
        logger.info(f"Records with demographic data: {merged_df['DEMOGIDX'].notna().sum()}")
        
        return merged_df
    
    def calculate_social_vulnerability_score(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate a composite social vulnerability score.
        
        Combines multiple EJScreen indicators into a single vulnerability
        score for resilience analysis.
        
        Args:
            df: DataFrame with EJScreen indicators
            weights: Optional dict of indicator weights. Defaults to equal weights.
            
        Returns:
            DataFrame with added 'social_vulnerability_score' column.
        """
        default_weights = {
            "LOWINCPCT": 0.25,
            "MINORPCT": 0.20,
            "LINGISOPCT": 0.15,
            "LESSHSPCT": 0.15,
            "UNDER5PCT": 0.10,
            "OVER64PCT": 0.15
        }
        
        weights = weights or default_weights
        
        df = df.copy()
        
        # Calculate weighted score
        score = 0
        total_weight = 0
        
        for indicator, weight in weights.items():
            if indicator in df.columns:
                # Normalize indicator to 0-1 range
                col_min = df[indicator].min()
                col_max = df[indicator].max()
                
                if col_max > col_min:
                    normalized = (df[indicator] - col_min) / (col_max - col_min)
                else:
                    normalized = 0
                
                score = score + (normalized * weight)
                total_weight += weight
        
        # Normalize final score
        if total_weight > 0:
            df["social_vulnerability_score"] = score / total_weight * 100
        else:
            df["social_vulnerability_score"] = np.nan
        
        logger.info("Calculated social vulnerability scores.")
        
        return df
    
    def get_vulnerability_summary(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate a summary of social vulnerability by ZIP code.
        
        Args:
            df: Merged workforce/demographic DataFrame
            
        Returns:
            DataFrame with vulnerability metrics by ZIP code.
        """
        if "zip_code" not in df.columns:
            return pd.DataFrame()
        
        vulnerability_cols = [
            "DEMOGIDX", "LOWINCPCT", "MINORPCT", "LINGISOPCT",
            "social_vulnerability_score"
        ]
        
        available_cols = [c for c in vulnerability_cols if c in df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        summary = df.groupby("zip_code")[available_cols].agg(["mean", "max"]).round(3)
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.reset_index()
        
        # Add workforce count
        workforce_counts = df.groupby("zip_code").size()
        summary["workforce_count"] = summary["zip_code"].map(workforce_counts)
        
        return summary


def overlay_demographics(
    workforce_df: pd.DataFrame,
    ejscreen_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to overlay demographics on workforce data.
    
    Args:
        workforce_df: DataFrame with workforce profiles
        ejscreen_path: Optional path to EJScreen CSV
        
    Returns:
        Merged DataFrame with demographic indicators.
        
    Example:
        >>> workforce_df = fetch_santa_barbara_workforce()
        >>> merged_df = overlay_demographics(workforce_df)
        >>> print(merged_df[['first_name', 'zip_code', 'DEMOGIDX', 'LOWINCPCT']])
    """
    overlay = DemographicOverlay(ejscreen_path=ejscreen_path)
    return overlay.merge_workforce_with_demographics(workforce_df)


def create_vulnerability_report(
    workforce_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comprehensive vulnerability report for workforce population.
    
    Args:
        workforce_df: DataFrame with workforce profiles
        output_path: Optional path to save report CSV
        
    Returns:
        DataFrame with vulnerability summary by ZIP code.
    """
    overlay = DemographicOverlay()
    
    # Merge with demographics
    merged_df = overlay.merge_workforce_with_demographics(workforce_df)
    
    # Calculate vulnerability scores
    merged_df = overlay.calculate_social_vulnerability_score(merged_df)
    
    # Generate summary
    summary = overlay.get_vulnerability_summary(merged_df)
    
    if output_path and not summary.empty:
        summary.to_csv(output_path, index=False)
        logger.info(f"Saved vulnerability report to {output_path}")
    
    return summary


# ============================================================================
# Main - Test Demographic Overlay
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coastal Labor-Resilience Engine - Demographic Overlay")
    print("=" * 60)
    
    # Create sample workforce data for testing
    sample_workforce = pd.DataFrame({
        "first_name": ["Alice", "Bob", "Carlos", "Diana", "Elena"],
        "last_name": ["Smith", "Johnson", "Garcia", "Chen", "Rodriguez"],
        "job_title": ["Server", "Fisher", "Manager", "Nurse", "Driver"],
        "company": ["Beachside Cafe", "SB Fisheries", "Hotel ABC", "SB Hospital", "Uber"],
        "industry": ["Hospitality", "Fishing", "Hospitality", "Healthcare", "Transportation"],
        "zip_code": ["93101", "93103", "93101", "93105", "93109"]
    })
    
    print("\n📊 Sample Workforce Data:")
    print("-" * 40)
    print(sample_workforce[["first_name", "job_title", "zip_code"]].to_string(index=False))
    
    # Test demographic overlay
    overlay = DemographicOverlay()
    
    print("\n🗺️  ZIP to Census Tract Mapping:")
    print("-" * 40)
    for zip_code in sample_workforce["zip_code"].unique():
        tracts = overlay.get_tract_from_zip(zip_code)
        print(f"  {zip_code}: {tracts}")
    
    # Try to load EJScreen data
    print("\n📈 Loading EJScreen Data:")
    print("-" * 40)
    try:
        ejscreen_df = overlay.load_ejscreen_data()
        print(f"  Loaded {len(ejscreen_df)} census tracts")
        print(f"  Columns: {list(ejscreen_df.columns)[:5]}...")
        
        # Merge demo
        merged_df = overlay.merge_workforce_with_demographics(sample_workforce)
        
        print("\n✅ Merged Data Sample:")
        print("-" * 40)
        display_cols = ["first_name", "zip_code", "DEMOGIDX", "LOWINCPCT"]
        available = [c for c in display_cols if c in merged_df.columns]
        print(merged_df[available].head().to_string(index=False))
        
    except FileNotFoundError:
        print("  ❌ EJScreen data file not found.")
        print("     Download from: https://www.epa.gov/ejscreen/download-ejscreen-data")
        print("     Save as: data/raw/ejscreen_data.csv")
    
    print("\n" + "=" * 60)

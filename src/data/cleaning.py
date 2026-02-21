"""
Data Cleaning and Alignment Module

Step 5: Cleaning & Alignment
Standardize all timestamps across datasets so "Storm Event A" aligns perfectly
with "Job Change B".

Author: Coastal Labor-Resilience Engine Team
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Standard timezone for all data
STANDARD_TIMEZONE = "America/Los_Angeles"  # Pacific Time
UTC_TIMEZONE = "UTC"

# Date format standards
ISO_DATE_FORMAT = "%Y-%m-%d"
ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
NOAA_DATE_FORMAT = "%Y%m%d"

# Event alignment windows (for matching events across datasets)
DEFAULT_ALIGNMENT_WINDOW_DAYS = 7


class DataAligner:
    """
    Aligns and standardizes temporal data across multiple datasets.
    
    Ensures that environmental events (storms, tides) can be precisely
    correlated with workforce changes (job transitions, unemployment).
    """
    
    def __init__(
        self,
        standard_tz: str = STANDARD_TIMEZONE,
        alignment_window_days: int = DEFAULT_ALIGNMENT_WINDOW_DAYS
    ):
        """
        Initialize the Data Aligner.
        
        Args:
            standard_tz: Standard timezone for all timestamps
            alignment_window_days: Default window for event matching
        """
        self.standard_tz = pytz.timezone(standard_tz)
        self.utc_tz = pytz.UTC
        self.alignment_window_days = alignment_window_days
    
    def standardize_timestamps(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
        source_tz: Optional[str] = None,
        infer_format: bool = True
    ) -> pd.DataFrame:
        """
        Standardize timestamps in a DataFrame to a common timezone.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_column: Name of the timestamp column
            source_tz: Source timezone. If None, assumes UTC or naive timestamps
            infer_format: Whether to infer datetime format automatically
            
        Returns:
            DataFrame with standardized timestamps
        """
        if timestamp_column not in df.columns:
            logger.warning(f"Column '{timestamp_column}' not found in DataFrame.")
            return df
        
        df = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(
                df[timestamp_column],
                infer_datetime_format=infer_format,
                errors="coerce"
            )
        
        # Handle timezone
        col = df[timestamp_column]
        
        if col.dt.tz is None:
            # Naive datetime - localize to source timezone
            source_tz = source_tz or "UTC"
            df[timestamp_column] = col.dt.tz_localize(source_tz)
        
        # Convert to standard timezone
        df[timestamp_column] = df[timestamp_column].dt.tz_convert(self.standard_tz)
        
        logger.info(f"Standardized {len(df)} timestamps to {self.standard_tz}")
        
        return df
    
    def standardize_noaa_data(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Standardize NOAA data timestamps (typically in GMT/UTC).
        
        Args:
            df: NOAA DataFrame
            timestamp_column: Name of timestamp column
            
        Returns:
            DataFrame with Pacific Time timestamps
        """
        logger.info("Standardizing NOAA data...")
        return self.standardize_timestamps(
            df,
            timestamp_column=timestamp_column,
            source_tz="UTC"  # NOAA uses GMT
        )
    
    def standardize_workforce_data(
        self,
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Standardize workforce data dates (job start dates, transitions).
        
        Args:
            df: Workforce DataFrame
            date_columns: List of date columns to standardize
            
        Returns:
            DataFrame with standardized dates
        """
        logger.info("Standardizing workforce data...")
        
        df = df.copy()
        
        # Default date columns in workforce data
        date_columns = date_columns or [
            "job_start_date",
            "job_end_date",
            "hire_date",
            "termination_date",
            "last_updated"
        ]
        
        for col in date_columns:
            if col in df.columns:
                df = self.standardize_timestamps(df, col)
        
        return df
    
    def extract_date_components(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Extract useful date components for analysis.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_column: Name of timestamp column
            prefix: Prefix for new column names
            
        Returns:
            DataFrame with added date component columns
        """
        if timestamp_column not in df.columns:
            return df
        
        df = df.copy()
        ts = df[timestamp_column]
        
        # Extract components
        prefix = f"{prefix}_" if prefix else ""
        
        df[f"{prefix}date"] = ts.dt.date
        df[f"{prefix}year"] = ts.dt.year
        df[f"{prefix}month"] = ts.dt.month
        df[f"{prefix}week"] = ts.dt.isocalendar().week
        df[f"{prefix}day_of_week"] = ts.dt.dayofweek
        df[f"{prefix}day_of_year"] = ts.dt.dayofyear
        df[f"{prefix}quarter"] = ts.dt.quarter
        
        return df
    
    def identify_event_windows(
        self,
        events_df: pd.DataFrame,
        event_timestamp_col: str,
        window_before_days: int = 7,
        window_after_days: int = 30
    ) -> pd.DataFrame:
        """
        Create event windows for correlation analysis.
        
        For each event (e.g., storm), creates a window of time
        before and after to search for correlated effects.
        
        Args:
            events_df: DataFrame with event timestamps
            event_timestamp_col: Name of event timestamp column
            window_before_days: Days before event to include
            window_after_days: Days after event to include
            
        Returns:
            DataFrame with window_start and window_end columns
        """
        df = events_df.copy()
        
        df["window_start"] = df[event_timestamp_col] - pd.Timedelta(days=window_before_days)
        df["window_end"] = df[event_timestamp_col] + pd.Timedelta(days=window_after_days)
        
        logger.info(f"Created event windows: -{window_before_days} to +{window_after_days} days")
        
        return df
    
    def align_events_to_workforce(
        self,
        events_df: pd.DataFrame,
        workforce_df: pd.DataFrame,
        event_timestamp_col: str = "timestamp",
        workforce_date_col: str = "job_start_date",
        window_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Align environmental events with workforce changes.
        
        Finds workforce events (job changes) that occurred within
        a time window of environmental events (storms, high tides).
        
        Args:
            events_df: Environmental events DataFrame
            workforce_df: Workforce DataFrame
            event_timestamp_col: Event timestamp column name
            workforce_date_col: Workforce date column name
            window_days: Days to search around each event
            
        Returns:
            DataFrame with aligned event-workforce pairs
        """
        logger.info("Aligning environmental events with workforce changes...")
        
        window_days = window_days or self.alignment_window_days
        
        # Ensure standardized timezones
        events_df = self.standardize_timestamps(events_df, event_timestamp_col)
        workforce_df = self.standardize_timestamps(workforce_df, workforce_date_col)
        
        # Create event windows
        events_df = self.identify_event_windows(
            events_df,
            event_timestamp_col,
            window_before_days=0,
            window_after_days=window_days
        )
        
        alignments = []
        
        for _, event in events_df.iterrows():
            event_time = event[event_timestamp_col]
            window_start = event["window_start"]
            window_end = event["window_end"]
            
            # Find workforce changes in window
            mask = (
                (workforce_df[workforce_date_col] >= window_start) &
                (workforce_df[workforce_date_col] <= window_end)
            )
            
            matching_workforce = workforce_df[mask].copy()
            
            if not matching_workforce.empty:
                # Add event information
                matching_workforce["event_timestamp"] = event_time
                matching_workforce["event_id"] = event.name if hasattr(event, "name") else None
                matching_workforce["days_after_event"] = (
                    matching_workforce[workforce_date_col] - event_time
                ).dt.days
                
                alignments.append(matching_workforce)
        
        if alignments:
            aligned_df = pd.concat(alignments, ignore_index=True)
            logger.info(f"Found {len(aligned_df)} event-workforce alignments.")
            return aligned_df
        else:
            logger.info("No alignments found between events and workforce changes.")
            return pd.DataFrame()
    
    def create_unified_timeline(
        self,
        datasets: Dict[str, Tuple[pd.DataFrame, str]],
        resolution: str = "D"
    ) -> pd.DataFrame:
        """
        Create a unified timeline combining all datasets.
        
        Args:
            datasets: Dict mapping dataset names to (DataFrame, timestamp_col) tuples
            resolution: Time resolution ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with unified timeline and event counts per dataset
        """
        logger.info(f"Creating unified timeline with {resolution} resolution...")
        
        # Find date range across all datasets
        min_date = None
        max_date = None
        
        for name, (df, ts_col) in datasets.items():
            if ts_col in df.columns and not df[ts_col].isna().all():
                df_min = df[ts_col].min()
                df_max = df[ts_col].max()
                
                if min_date is None or df_min < min_date:
                    min_date = df_min
                if max_date is None or df_max > max_date:
                    max_date = df_max
        
        if min_date is None or max_date is None:
            logger.warning("No valid dates found in datasets.")
            return pd.DataFrame()
        
        # Create timeline index
        timeline = pd.date_range(
            start=min_date.normalize(),
            end=max_date.normalize(),
            freq=resolution
        )
        
        timeline_df = pd.DataFrame({"date": timeline})
        
        # Add event counts per dataset
        for name, (df, ts_col) in datasets.items():
            if ts_col in df.columns:
                # Convert to same frequency
                df_dates = df[ts_col].dt.to_period(resolution)
                counts = df_dates.value_counts().sort_index()
                
                # Map to timeline
                timeline_df[f"{name}_count"] = 0
                for period, count in counts.items():
                    mask = timeline_df["date"].dt.to_period(resolution) == period
                    timeline_df.loc[mask, f"{name}_count"] = count
        
        logger.info(f"Created timeline from {min_date.date()} to {max_date.date()}")
        
        return timeline_df


class DataCleaner:
    """
    Cleans and validates data from multiple sources.
    """
    
    @staticmethod
    def clean_workforce_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean workforce data for analysis.
        
        - Removes duplicates
        - Standardizes text fields
        - Validates required fields
        
        Args:
            df: Raw workforce DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning workforce data...")
        
        df = df.copy()
        original_len = len(df)
        
        # Remove duplicates
        if "linkedin_url" in df.columns:
            df = df.drop_duplicates(subset=["linkedin_url"])
        elif "first_name" in df.columns and "last_name" in df.columns:
            df = df.drop_duplicates(subset=["first_name", "last_name", "company"])
        
        # Standardize text fields
        text_cols = ["first_name", "last_name", "job_title", "company", "industry"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()
        
        # Standardize ZIP codes
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
            # Remove invalid ZIPs
            df = df[df["zip_code"].str.match(r"^\d{5}$", na=False)]
        
        removed = original_len - len(df)
        logger.info(f"Cleaned workforce data: {removed} records removed, {len(df)} remaining.")
        
        return df
    
    @staticmethod
    def clean_noaa_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NOAA environmental data.
        
        - Removes records with missing water levels
        - Filters out flagged/suspect data
        - Removes obvious outliers
        
        Args:
            df: Raw NOAA DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning NOAA data...")
        
        df = df.copy()
        original_len = len(df)
        
        # Remove null water levels
        if "water_level_m" in df.columns:
            df = df.dropna(subset=["water_level_m"])
            
            # Remove extreme outliers (likely errors)
            q1 = df["water_level_m"].quantile(0.001)
            q99 = df["water_level_m"].quantile(0.999)
            margin = (q99 - q1) * 2
            
            df = df[
                (df["water_level_m"] >= q1 - margin) &
                (df["water_level_m"] <= q99 + margin)
            ]
        
        # Remove flagged data if quality flag exists
        if "f" in df.columns or "flags" in df.columns:
            flag_col = "f" if "f" in df.columns else "flags"
            # Keep only clean data (no flag or '0' flag)
            df = df[(df[flag_col].isna()) | (df[flag_col].isin(["0", 0, ""]))]
        
        removed = original_len - len(df)
        logger.info(f"Cleaned NOAA data: {removed} records removed, {len(df)} remaining.")
        
        return df
    
    @staticmethod
    def clean_ejscreen_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean EPA EJScreen data.
        
        Args:
            df: Raw EJScreen DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning EJScreen data...")
        
        df = df.copy()
        
        # Ensure GEOID is string with leading zeros
        if "GEOID" in df.columns:
            df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        
        # Handle missing values in percentage columns
        pct_cols = [c for c in df.columns if c.endswith("PCT")]
        for col in pct_cols:
            if col in df.columns:
                # Fill missing with median (for analysis purposes)
                df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"Cleaned EJScreen data: {len(df)} records.")
        
        return df


def align_all_datasets(
    workforce_df: pd.DataFrame,
    events_df: pd.DataFrame,
    ejscreen_df: Optional[pd.DataFrame] = None,
    window_days: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to align all datasets for analysis.
    
    Args:
        workforce_df: Workforce profiles DataFrame
        events_df: Environmental events DataFrame
        ejscreen_df: Optional EJScreen demographics DataFrame
        window_days: Days to search for event correlations
        
    Returns:
        Dict with cleaned and aligned DataFrames:
        - 'workforce': Cleaned workforce data
        - 'events': Cleaned event data  
        - 'aligned': Event-workforce alignments
        - 'timeline': Unified timeline
    """
    # Initialize processors
    cleaner = DataCleaner()
    aligner = DataAligner()
    
    # Clean data
    clean_workforce = cleaner.clean_workforce_data(workforce_df)
    clean_events = cleaner.clean_noaa_data(events_df)
    
    # Standardize timestamps
    clean_workforce = aligner.standardize_workforce_data(clean_workforce)
    clean_events = aligner.standardize_noaa_data(clean_events)
    
    # Align events with workforce
    aligned = aligner.align_events_to_workforce(
        clean_events,
        clean_workforce,
        window_days=window_days
    )
    
    # Create unified timeline
    datasets = {
        "events": (clean_events, "timestamp"),
        "workforce": (clean_workforce, "job_start_date")
    }
    timeline = aligner.create_unified_timeline(datasets)
    
    result = {
        "workforce": clean_workforce,
        "events": clean_events,
        "aligned": aligned,
        "timeline": timeline
    }
    
    if ejscreen_df is not None:
        result["ejscreen"] = cleaner.clean_ejscreen_data(ejscreen_df)
    
    return result


# ============================================================================
# Main - Test Data Alignment
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coastal Labor-Resilience Engine - Data Cleaning & Alignment")
    print("=" * 60)
    
    # Create sample data for testing
    
    # Sample environmental events (storm dates)
    events = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-15 08:30:00",
            "2024-02-20 14:00:00",
            "2024-03-10 22:15:00"
        ]),
        "water_level_m": [2.45, 2.38, 2.52],
        "event_type": ["storm", "king_tide", "storm"]
    })
    
    # Sample workforce changes
    workforce = pd.DataFrame({
        "first_name": ["Alice", "Bob", "Carlos", "Diana", "Elena", "Frank"],
        "job_start_date": pd.to_datetime([
            "2024-01-18",  # 3 days after storm
            "2024-01-25",  # 10 days after storm
            "2024-02-22",  # 2 days after king tide
            "2024-03-12",  # 2 days after storm
            "2024-03-25",  # 15 days after storm
            "2024-04-01"   # No related event
        ]),
        "job_title": ["Server", "Fisher", "Manager", "Nurse", "Driver", "Clerk"],
        "zip_code": ["93101", "93103", "93101", "93105", "93109", "93111"]
    })
    
    print("\n📊 Sample Environmental Events:")
    print("-" * 40)
    print(events.to_string(index=False))
    
    print("\n👥 Sample Workforce Changes:")
    print("-" * 40)
    print(workforce[["first_name", "job_start_date", "job_title"]].to_string(index=False))
    
    # Initialize aligner
    aligner = DataAligner()
    
    # Standardize timestamps
    print("\n⏰ Standardizing Timestamps to Pacific Time:")
    print("-" * 40)
    events_std = aligner.standardize_timestamps(events, "timestamp", source_tz="UTC")
    print(f"  Original (UTC): {events['timestamp'].iloc[0]}")
    print(f"  Standardized (PT): {events_std['timestamp'].iloc[0]}")
    
    # Align events with workforce
    print("\n🔗 Aligning Events with Workforce Changes (7-day window):")
    print("-" * 40)
    aligned = aligner.align_events_to_workforce(
        events,
        workforce,
        event_timestamp_col="timestamp",
        workforce_date_col="job_start_date",
        window_days=7
    )
    
    if not aligned.empty:
        print(aligned[[
            "first_name", "job_start_date", "event_timestamp", "days_after_event"
        ]].to_string(index=False))
        
        print(f"\n  ✅ Found {len(aligned)} workforce changes within 7 days of events")
    else:
        print("  No alignments found")
    
    # Create unified timeline
    print("\n📅 Unified Timeline (monthly resolution):")
    print("-" * 40)
    datasets = {
        "events": (events, "timestamp"),
        "workforce": (workforce, "job_start_date")
    }
    timeline = aligner.create_unified_timeline(datasets, resolution="M")
    print(timeline.to_string(index=False))
    
    print("\n" + "=" * 60)

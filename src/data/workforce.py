"""
Workforce Scoping Module

Step 2: Workforce Scoping
A function using the Live Data /search endpoint to pull professional profiles
specifically for the Santa Barbara area.

Author: Coastal Labor-Resilience Engine Team
"""

import os
import logging
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv

from .auth import TokenManager, AuthenticationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API Configuration
LIVE_DATA_API_BASE_URL = "https://api.livedata.io/v1"

# Santa Barbara Area ZIP Codes
SANTA_BARBARA_ZIP_CODES = [
    "93101", "93102", "93103", "93105", "93106", "93107",
    "93108", "93109", "93110", "93111", "93117", "93118",
    "93120", "93121", "93130", "93140", "93150", "93160",
    "93190", "93199"
]

# Industry categories relevant to coastal resilience
COASTAL_INDUSTRIES = [
    "Agriculture",
    "Fishing",
    "Tourism",
    "Hospitality",
    "Marine Services",
    "Oil and Gas",
    "Construction",
    "Transportation",
    "Real Estate",
    "Insurance",
    "Healthcare",
    "Emergency Services"
]


class WorkforceClient:
    """
    Client for fetching and analyzing workforce data from Live Data Technologies.
    
    Provides methods to search, filter, and analyze professional profiles
    in the Santa Barbara coastal region.
    """
    
    def __init__(self, token_manager: Optional[TokenManager] = None):
        """
        Initialize the Workforce Client.
        
        Args:
            token_manager: TokenManager instance for authentication.
                          If None, creates a new one using environment variables.
        """
        self.token_manager = token_manager or TokenManager()
        self.base_url = LIVE_DATA_API_BASE_URL
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication for API requests."""
        headers = self.token_manager.get_auth_header()
        headers["Content-Type"] = "application/json"
        return headers
    
    def search_profiles(
        self,
        location: str = "Santa Barbara, CA",
        zip_codes: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        job_titles: Optional[List[str]] = None,
        company_names: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search for professional profiles using Live Data /search endpoint.
        
        Args:
            location: Primary location to search (city, state)
            zip_codes: List of ZIP codes to filter by
            industries: List of industries to filter by
            job_titles: List of job titles to search
            company_names: List of company names to filter
            limit: Maximum results per request (1-1000)
            offset: Pagination offset
            
        Returns:
            Dict containing:
            - 'results': List of profile dictionaries
            - 'total': Total matching profiles
            - 'has_more': Whether more results exist
        """
        logger.info(f"Searching profiles in {location}...")
        
        search_url = f"{self.base_url}/search"
        
        query = {
            "limit": min(limit, 1000),
            "offset": offset
        }
        
        # Build location filter
        location_filter = {"location": location}
        if zip_codes:
            location_filter["zip_codes"] = zip_codes
        
        query["filters"] = {"location": location_filter}
        
        # Add optional filters
        if industries:
            query["filters"]["industries"] = industries
        if job_titles:
            query["filters"]["job_titles"] = job_titles
        if company_names:
            query["filters"]["company_names"] = company_names
        
        try:
            response = requests.post(
                search_url,
                json=query,
                headers=self._get_headers(),
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            logger.info(f"Found {data.get('total', 0)} matching profiles.")
            
            return {
                "results": data.get("results", []),
                "total": data.get("total", 0),
                "has_more": data.get("has_more", False)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Profile search failed: {e}")
            raise
    
    def fetch_santa_barbara_workforce(
        self,
        industries: Optional[List[str]] = None,
        max_profiles: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch all workforce profiles for the Santa Barbara area.
        
        Handles pagination to retrieve up to max_profiles results.
        
        Args:
            industries: Optional list of industries to filter.
                       Defaults to COASTAL_INDUSTRIES.
            max_profiles: Maximum total profiles to retrieve.
            
        Returns:
            DataFrame with workforce profile data.
        """
        industries = industries or COASTAL_INDUSTRIES
        
        logger.info(f"Fetching Santa Barbara workforce data...")
        logger.info(f"ZIP codes: {len(SANTA_BARBARA_ZIP_CODES)}")
        logger.info(f"Industries: {len(industries)}")
        
        all_profiles = []
        offset = 0
        page_size = 100
        
        while len(all_profiles) < max_profiles:
            result = self.search_profiles(
                location="Santa Barbara, CA",
                zip_codes=SANTA_BARBARA_ZIP_CODES,
                industries=industries,
                limit=page_size,
                offset=offset
            )
            
            profiles = result.get("results", [])
            
            if not profiles:
                break
            
            all_profiles.extend(profiles)
            offset += len(profiles)
            
            logger.info(f"Retrieved {len(all_profiles)} / {result.get('total', '?')} profiles")
            
            if not result.get("has_more", False):
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_profiles)
        
        if not df.empty:
            df = self._normalize_profile_data(df)
        
        logger.info(f"Total profiles retrieved: {len(df)}")
        return df
    
    def _normalize_profile_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and clean profile data for analysis.
        
        Args:
            df: Raw profile DataFrame
            
        Returns:
            Cleaned DataFrame with standardized columns.
        """
        # Standardize column names
        column_mapping = {
            "first_name": "first_name",
            "last_name": "last_name",
            "current_title": "job_title",
            "current_company": "company",
            "location": "location",
            "zip_code": "zip_code",
            "industry": "industry",
            "start_date": "job_start_date",
            "linkedin_url": "linkedin_url"
        }
        
        # Rename columns that exist
        for old, new in column_mapping.items():
            if old in df.columns and old != new:
                df = df.rename(columns={old: new})
        
        # Extract ZIP code from location if not present
        if "zip_code" not in df.columns and "location" in df.columns:
            df["zip_code"] = df["location"].str.extract(r"(\d{5})")
        
        # Standardize ZIP codes
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
        
        # Parse dates
        if "job_start_date" in df.columns:
            df["job_start_date"] = pd.to_datetime(
                df["job_start_date"], 
                errors="coerce"
            )
        
        return df
    
    def get_industry_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate industry distribution for workforce analysis.
        
        Args:
            df: Workforce DataFrame
            
        Returns:
            DataFrame with industry counts and percentages.
        """
        if "industry" not in df.columns:
            return pd.DataFrame()
        
        distribution = df["industry"].value_counts().reset_index()
        distribution.columns = ["industry", "count"]
        distribution["percentage"] = (
            distribution["count"] / distribution["count"].sum() * 100
        ).round(2)
        
        return distribution
    
    def get_geographic_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate geographic distribution by ZIP code.
        
        Args:
            df: Workforce DataFrame
            
        Returns:
            DataFrame with ZIP code counts and percentages.
        """
        if "zip_code" not in df.columns:
            return pd.DataFrame()
        
        distribution = df["zip_code"].value_counts().reset_index()
        distribution.columns = ["zip_code", "count"]
        distribution["percentage"] = (
            distribution["count"] / distribution["count"].sum() * 100
        ).round(2)
        
        return distribution


def fetch_santa_barbara_workforce(
    industries: Optional[List[str]] = None,
    max_profiles: int = 1000
) -> pd.DataFrame:
    """
    Convenience function to fetch Santa Barbara workforce data.
    
    Args:
        industries: Optional list of industries to filter.
        max_profiles: Maximum profiles to retrieve.
        
    Returns:
        DataFrame with workforce profiles.
        
    Example:
        >>> df = fetch_santa_barbara_workforce(industries=["Tourism", "Hospitality"])
        >>> print(df.head())
    """
    client = WorkforceClient()
    return client.fetch_santa_barbara_workforce(
        industries=industries,
        max_profiles=max_profiles
    )


# ============================================================================
# Main - Test Workforce Scoping
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coastal Labor-Resilience Engine - Workforce Scoping")
    print("=" * 60)
    
    try:
        client = WorkforceClient()
        
        # Test with a small sample
        print("\nFetching sample workforce data...")
        
        result = client.search_profiles(
            location="Santa Barbara, CA",
            zip_codes=SANTA_BARBARA_ZIP_CODES[:5],  # First 5 ZIP codes
            industries=["Tourism", "Hospitality", "Agriculture"],
            limit=10
        )
        
        print(f"\n✅ Search successful!")
        print(f"   Total matching profiles: {result['total']}")
        print(f"   Profiles retrieved: {len(result['results'])}")
        print(f"   Has more results: {result['has_more']}")
        
        if result['results']:
            df = pd.DataFrame(result['results'])
            print(f"\n   Sample columns: {list(df.columns)[:5]}")
        
    except AuthenticationError as e:
        print(f"\n❌ Authentication Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

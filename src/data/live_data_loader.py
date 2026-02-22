"""
Live Data Technologies Data Loader
----------------------------------
Loads and processes JSONL.GZ files from Live Data Technologies
for coastal labor market analysis.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator, Optional, Union, List, Dict
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime


@dataclass
class JobRecord:
    """Represents a single job from employment history."""
    title: Optional[str]
    level: Optional[str]
    function: Optional[str]
    company_name: Optional[str]
    industry: Optional[str]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    location: Optional[str]
    county: Optional[str]
    region: Optional[str]
    msa: Optional[str]
    is_current: bool = False
    duration_months: Optional[int] = None


@dataclass
class PersonRecord:
    """Represents a person from Live Data Technologies."""
    id: str
    current_title: Optional[str]
    current_company: Optional[str]
    current_industry: Optional[str]
    current_location: Optional[str]
    current_county: Optional[str]
    current_region: Optional[str]
    current_msa: Optional[str]
    employment_status: Optional[str]
    job_history: list[JobRecord] = field(default_factory=list)
    

class LiveDataLoader:
    """Loads and processes Live Data Technologies JSONL.GZ files."""
    
    # California coastal counties for filtering
    COASTAL_COUNTIES = {
        "Santa Barbara County",
        "Ventura County", 
        "Los Angeles County",
        "Orange County",
        "San Diego County",
        "San Luis Obispo County",
        "Monterey County",
        "Santa Cruz County",
        "San Mateo County",
        "San Francisco County",
        "Marin County",
        "Sonoma County",
        "Mendocino County",
        "Humboldt County",
        "Del Norte County",
    }
    
    # Climate-sensitive industries (coastal/outdoor/agriculture)
    CLIMATE_SENSITIVE_INDUSTRIES = {
        "Hospitality",
        "Restaurants",
        "Food & Beverages",
        "Food and Beverage Services",
        "Food Production",
        "Leisure, Travel & Tourism",
        "Travel Arrangements",
        "Airlines/Aviation",
        "Aviation & Aerospace",
        "Aviation and Aerospace Component Manufacturing",
        "Gambling & Casinos",
        "Performing Arts",
        "Spectator Sports",
        "Sports",
        "Sporting Goods",
        "Recreational Facilities and Services",
        "Construction",
        "Building Materials",
        "Oil & Energy",
        "Utilities",
        "Transportation/Trucking/Railroad",
        "Logistics and Supply Chain",
        "Environmental Services",
        "Renewables & Environment",
        "Services for Renewable Energy",
        "Solar Electric Power Generation",
        "Glass, Ceramics & Concrete",
        "Events Services",
        "Wholesale",
        "Retail",
        "Apparel & Fashion",
        "Automotive",
    }
    
    def __init__(self, data_dir: str | Path):
        """Initialize loader with data directory path."""
        self.data_dir = Path(data_dir)
        
    def list_files(self) -> list[Path]:
        """List all JSONL.GZ files in the data directory."""
        return list(self.data_dir.glob("*.jsonl.gz"))
    
    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    
    def _extract_location_details(self, loc_details: Optional[dict]) -> dict:
        """Extract location details from nested dict."""
        if not loc_details:
            return {"county": None, "region": None, "msa": None, "locality": None}
        return {
            "county": loc_details.get("county"),
            "region": loc_details.get("region"),
            "msa": loc_details.get("msa"),
            "locality": loc_details.get("locality"),
        }
    
    def _parse_job(self, job_data: dict, is_current: bool = False) -> JobRecord:
        """Parse a job record from raw JSON."""
        company = job_data.get("company", {}) or {}
        loc_details = self._extract_location_details(job_data.get("location_details"))
        
        return JobRecord(
            title=job_data.get("title"),
            level=job_data.get("level"),
            function=job_data.get("function"),
            company_name=company.get("name"),
            industry=company.get("industry"),
            started_at=self._parse_datetime(job_data.get("started_at")),
            ended_at=self._parse_datetime(job_data.get("ended_at")),
            location=job_data.get("location"),
            county=loc_details["county"],
            region=loc_details["region"],
            msa=loc_details["msa"],
            is_current=is_current,
            duration_months=job_data.get("duration"),
        )
    
    def _parse_person(self, data: dict) -> PersonRecord:
        """Parse a person record from raw JSON."""
        position = data.get("position", {}) or {}
        company = position.get("company", {}) or {}
        loc_details = self._extract_location_details(position.get("location_details"))
        
        # Parse job history
        jobs = []
        for i, job_data in enumerate(data.get("jobs", [])):
            is_current = (i == 0 and job_data.get("ended_at") is None)
            jobs.append(self._parse_job(job_data, is_current))
        
        return PersonRecord(
            id=data.get("id", ""),
            current_title=position.get("title"),
            current_company=company.get("name"),
            current_industry=company.get("industry"),
            current_location=position.get("location"),
            current_county=loc_details["county"],
            current_region=loc_details["region"],
            current_msa=loc_details["msa"],
            employment_status=data.get("employment_status"),
            job_history=jobs,
        )
    
    def stream_records(
        self, 
        files: Optional[list[Path]] = None,
        max_records: Optional[int] = None,
    ) -> Iterator[PersonRecord]:
        """Stream person records from JSONL.GZ files."""
        files = files or self.list_files()
        count = 0
        
        for filepath in files:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    if max_records and count >= max_records:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield self._parse_person(data)
                        count += 1
                    except json.JSONDecodeError:
                        continue
    
    def filter_california_coastal(
        self,
        records: Iterator[PersonRecord],
    ) -> Iterator[PersonRecord]:
        """Filter records to California coastal counties."""
        for person in records:
            # Check current position
            if person.current_region == "California" and person.current_county in self.COASTAL_COUNTIES:
                yield person
                continue
            
            # Check job history for California coastal work
            for job in person.job_history:
                if job.region == "California" and job.county in self.COASTAL_COUNTIES:
                    yield person
                    break
    
    def load_to_dataframe(
        self,
        filter_coastal: bool = True,
        max_records: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load records into a pandas DataFrame."""
        records = self.stream_records(max_records=max_records)
        
        if filter_coastal:
            records = self.filter_california_coastal(records)
        
        data = []
        for person in records:
            data.append({
                "person_id": person.id,
                "current_title": person.current_title,
                "current_company": person.current_company,
                "current_industry": person.current_industry,
                "current_location": person.current_location,
                "current_county": person.current_county,
                "current_region": person.current_region,
                "current_msa": person.current_msa,
                "employment_status": person.employment_status,
                "job_count": len(person.job_history),
            })
        
        return pd.DataFrame(data)
    
    def get_job_transitions(
        self,
        filter_coastal: bool = True,
        max_records: Optional[int] = None,
    ) -> pd.DataFrame:
        """Extract job-to-job transitions for Markov analysis."""
        records = self.stream_records(max_records=max_records)
        
        if filter_coastal:
            records = self.filter_california_coastal(records)
        
        transitions = []
        for person in records:
            jobs = person.job_history
            for i in range(len(jobs) - 1):
                from_job = jobs[i]
                to_job = jobs[i + 1]
                
                # Skip if missing data
                if not from_job.title or not to_job.title:
                    continue
                
                transitions.append({
                    "person_id": person.id,
                    "from_title": from_job.title,
                    "from_industry": from_job.industry,
                    "from_county": from_job.county,
                    "to_title": to_job.title,
                    "to_industry": to_job.industry,
                    "to_county": to_job.county,
                    "transition_date": to_job.started_at,
                })
        
        return pd.DataFrame(transitions)
    
    def get_industry_distribution(
        self,
        filter_coastal: bool = True,
        max_records: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get industry distribution for coastal workers."""
        records = self.stream_records(max_records=max_records)
        
        if filter_coastal:
            records = self.filter_california_coastal(records)
        
        industries = []
        for person in records:
            if person.current_industry:
                industries.append({
                    "industry": person.current_industry,
                    "county": person.current_county,
                    "is_climate_sensitive": person.current_industry in self.CLIMATE_SENSITIVE_INDUSTRIES,
                })
        
        df = pd.DataFrame(industries)
        return df.groupby(["industry", "is_climate_sensitive"]).size().reset_index(name="count")


def load_live_data(data_path: str | Path, max_records: int = 10000) -> dict:
    """
    Convenience function to load Live Data for dashboard.
    
    Returns dict with:
    - persons_df: DataFrame of persons
    - transitions_df: DataFrame of job transitions
    - industry_df: DataFrame of industry distribution
    """
    loader = LiveDataLoader(data_path)
    
    return {
        "persons_df": loader.load_to_dataframe(filter_coastal=True, max_records=max_records),
        "transitions_df": loader.get_job_transitions(filter_coastal=True, max_records=max_records),
        "industry_df": loader.get_industry_distribution(filter_coastal=True, max_records=max_records),
    }


if __name__ == "__main__":
    # Test loading
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "drive-download-20260221T181111Z-3-001"
    
    loader = LiveDataLoader(data_path)
    print(f"Found {len(loader.list_files())} files")
    
    # Load sample
    df = loader.load_to_dataframe(filter_coastal=True, max_records=1000)
    print(f"\nLoaded {len(df)} California coastal workers")
    print(f"\nTop industries:")
    print(df["current_industry"].value_counts().head(10))
    print(f"\nTop counties:")
    print(df["current_county"].value_counts().head(10))

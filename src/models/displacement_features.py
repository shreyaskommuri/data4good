"""
Worker Displacement Risk - Feature Engineering
----------------------------------------------
Extracts features from Live Data worker records to predict
displacement risk from climate shocks and sea level rise.

Target Variable:
- displaced: Worker left coastal county or climate-sensitive industry

Features:
- Industry characteristics (climate sensitivity, diversity)
- Job history patterns (tenure, transitions, stability)
- Geographic exposure (coastal county, flood risk)
- Employment profile (seniority, function)
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from ..data.live_data_loader import PersonRecord, JobRecord, LiveDataLoader


@dataclass
class DisplacementFeatures:
    """Feature set for displacement risk prediction."""
    person_id: str
    
    # Target variable
    displaced: int  # 1 = displaced, 0 = retained
    
    # Industry features
    current_climate_sensitive: int
    prev_climate_sensitive: int
    industry_transitions: int
    climate_industry_tenure_months: float
    industry_diversity: float  # Unique industries / total jobs
    
    # Geographic features
    current_coastal: int
    prev_coastal: int
    county_changes: int
    left_coastal_county: int
    
    # Job history features
    total_jobs: int
    avg_tenure_months: float
    current_tenure_months: float
    job_stability_score: float  # Avg tenure / total jobs
    recent_transitions: int  # Transitions in last 2 years
    
    # Employment level
    is_senior: int
    is_manager: int
    is_entry: int
    
    def to_dict(self) -> Dict:
        return {
            'person_id': self.person_id,
            'displaced': self.displaced,
            'current_climate_sensitive': self.current_climate_sensitive,
            'prev_climate_sensitive': self.prev_climate_sensitive,
            'industry_transitions': self.industry_transitions,
            'climate_industry_tenure_months': self.climate_industry_tenure_months,
            'industry_diversity': self.industry_diversity,
            'current_coastal': self.current_coastal,
            'prev_coastal': self.prev_coastal,
            'county_changes': self.county_changes,
            'left_coastal_county': self.left_coastal_county,
            'total_jobs': self.total_jobs,
            'avg_tenure_months': self.avg_tenure_months,
            'current_tenure_months': self.current_tenure_months,
            'job_stability_score': self.job_stability_score,
            'recent_transitions': self.recent_transitions,
            'is_senior': self.is_senior,
            'is_manager': self.is_manager,
            'is_entry': self.is_entry,
        }


class DisplacementFeatureExtractor:
    """Extract ML features from worker records."""
    
    CLIMATE_SENSITIVE_INDUSTRIES = {
        "Hospitality", "Restaurants", "Food & Beverages",
        "Food and Beverage Services", "Food Production",
        "Leisure, Travel & Tourism", "Travel Arrangements",
        "Airlines/Aviation", "Aviation & Aerospace",
        "Aviation and Aerospace Component Manufacturing",
        "Gambling & Casinos", "Performing Arts",
        "Spectator Sports", "Sports", "Sporting Goods",
        "Recreational Facilities and Services",
        "Retail", "Retail Apparel and Fashion",
        "Warehousing", "Transportation/Trucking/Railroad",
        "Farming", "Ranching", "Fishery",
        "Construction", "Oil & Energy",
        "Maritime", "Shipbuilding",
    }
    
    COASTAL_COUNTIES = {
        "Santa Barbara County", "Ventura County", 
        "Los Angeles County", "Orange County",
        "San Diego County", "San Luis Obispo County",
        "Monterey County", "Santa Cruz County",
        "San Mateo County", "San Francisco County",
        "Marin County", "Sonoma County",
        "Mendocino County", "Humboldt County",
        "Del Norte County",
    }
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def _is_climate_sensitive(self, industry: Optional[str]) -> bool:
        """Check if industry is climate-sensitive."""
        if not industry:
            return False
        return industry in self.CLIMATE_SENSITIVE_INDUSTRIES
    
    def _is_coastal(self, county: Optional[str]) -> bool:
        """Check if county is coastal."""
        if not county:
            return False
        return county in self.COASTAL_COUNTIES
    
    def _calculate_tenure_months(self, job: JobRecord) -> float:
        """Calculate job tenure in months."""
        if not job.started_at:
            return 0.0
        
        # Make dates timezone-naive for comparison
        started = job.started_at.replace(tzinfo=None) if job.started_at.tzinfo else job.started_at
        now = datetime.now()
        
        if job.ended_at:
            ended = job.ended_at.replace(tzinfo=None) if job.ended_at.tzinfo else job.ended_at
            # Don't allow future end dates
            end_date = min(ended, now)
        else:
            end_date = now
        
        months = (end_date.year - started.year) * 12
        months += end_date.month - started.month
        return max(0.0, float(months))
    
    def _extract_seniority(self, title: Optional[str], level: Optional[str]) -> Tuple[int, int, int]:
        """Extract seniority flags from title and level."""
        if not title and not level:
            return 0, 0, 0
        
        title_lower = (title or "").lower()
        level_lower = (level or "").lower()
        text = f"{title_lower} {level_lower}"
        
        is_senior = int(any(kw in text for kw in [
            "senior", "sr.", "lead", "principal", "staff", "director", "vp", "vice president"
        ]))
        
        is_manager = int(any(kw in text for kw in [
            "manager", "supervisor", "head of", "chief", "ceo", "cto", "cfo"
        ]))
        
        is_entry = int(any(kw in text for kw in [
            "junior", "jr.", "entry", "intern", "trainee", "associate"
        ]))
        
        return is_senior, is_manager, is_entry
    
    def _define_displacement(self, person: PersonRecord) -> int:
        """
        Define if a worker was displaced.
        
        Displacement criteria:
        1. Left a climate-sensitive industry AND
        2. Left a coastal county OR moved to climate-resilient industry
        """
        if not person.job_history or len(person.job_history) < 2:
            return 0
        
        # Get recent history (last 2 jobs)
        recent_jobs = sorted(
            [j for j in person.job_history if j.started_at],
            key=lambda j: j.started_at,
            reverse=True
        )[:2]
        
        if len(recent_jobs) < 2:
            return 0
        
        current_job = recent_jobs[0]
        prev_job = recent_jobs[1]
        
        # Check if left climate-sensitive industry
        left_sensitive = (
            self._is_climate_sensitive(prev_job.industry) and
            not self._is_climate_sensitive(current_job.industry)
        )
        
        # Check if left coastal county
        left_coastal = (
            self._is_coastal(prev_job.county) and
            not self._is_coastal(current_job.county)
        )
        
        # Displaced if left sensitive industry and (left coast OR went to resilient)
        displaced = int(left_sensitive and (left_coastal or not self._is_climate_sensitive(current_job.industry)))
        
        return displaced
    
    def extract_features(self, person: PersonRecord) -> DisplacementFeatures:
        """Extract all features for one person."""
        
        # Target variable
        displaced = self._define_displacement(person)
        
        # Industry features
        current_climate_sensitive = int(self._is_climate_sensitive(person.current_industry))
        
        prev_industry = None
        if person.job_history:
            sorted_jobs = sorted(
                [j for j in person.job_history if j.started_at and not j.is_current],
                key=lambda j: j.started_at,
                reverse=True
            )
            if sorted_jobs:
                prev_industry = sorted_jobs[0].industry
        
        prev_climate_sensitive = int(self._is_climate_sensitive(prev_industry))
        
        # Count industry transitions
        industries = [j.industry for j in person.job_history if j.industry]
        industry_transitions = len([i for i in range(1, len(industries)) if industries[i] != industries[i-1]])
        
        # Industry diversity
        unique_industries = len(set(industries)) if industries else 0
        industry_diversity = unique_industries / max(len(industries), 1)
        
        # Climate-sensitive tenure
        climate_tenure = sum(
            self._calculate_tenure_months(j)
            for j in person.job_history
            if self._is_climate_sensitive(j.industry)
        )
        
        # Geographic features
        current_coastal = int(self._is_coastal(person.current_county))
        
        prev_county = None
        if person.job_history:
            sorted_jobs = sorted(
                [j for j in person.job_history if j.started_at and not j.is_current],
                key=lambda j: j.started_at,
                reverse=True
            )
            if sorted_jobs:
                prev_county = sorted_jobs[0].county
        
        prev_coastal = int(self._is_coastal(prev_county))
        
        # County changes
        counties = [j.county for j in person.job_history if j.county]
        county_changes = len([i for i in range(1, len(counties)) if counties[i] != counties[i-1]])
        
        # Check if left coastal county
        left_coastal_county = int(prev_coastal == 1 and current_coastal == 0)
        
        # Job history features
        total_jobs = len(person.job_history)
        
        tenures = [self._calculate_tenure_months(j) for j in person.job_history]
        avg_tenure = np.mean(tenures) if tenures else 0.0
        
        current_job = next((j for j in person.job_history if j.is_current), None)
        current_tenure = self._calculate_tenure_months(current_job) if current_job else 0.0
        
        job_stability = avg_tenure / max(total_jobs, 1)
        
        # Recent transitions (last 2 years)
        cutoff_date = datetime.now().replace(year=datetime.now().year - 2)
        recent_transitions = len([
            j for j in person.job_history
            if j.started_at and (j.started_at.replace(tzinfo=None) if j.started_at.tzinfo else j.started_at) >= cutoff_date
        ]) - 1  # Subtract 1 because current job doesn't count as transition
        recent_transitions = max(0, recent_transitions)
        
        # Seniority features
        is_senior, is_manager, is_entry = self._extract_seniority(
            person.current_title,
            current_job.level if current_job else None
        )
        
        return DisplacementFeatures(
            person_id=person.id,
            displaced=displaced,
            current_climate_sensitive=current_climate_sensitive,
            prev_climate_sensitive=prev_climate_sensitive,
            industry_transitions=industry_transitions,
            climate_industry_tenure_months=climate_tenure,
            industry_diversity=industry_diversity,
            current_coastal=current_coastal,
            prev_coastal=prev_coastal,
            county_changes=county_changes,
            left_coastal_county=left_coastal_county,
            total_jobs=total_jobs,
            avg_tenure_months=avg_tenure,
            current_tenure_months=current_tenure,
            job_stability_score=job_stability,
            recent_transitions=recent_transitions,
            is_senior=is_senior,
            is_manager=is_manager,
            is_entry=is_entry,
        )
    
    def extract_batch(
        self,
        data_path: str,
        max_records: Optional[int] = None
    ) -> pd.DataFrame:
        """Extract features for all workers in dataset."""
        loader = LiveDataLoader(data_path)
        records = loader.stream_records(max_records=max_records)
        
        # Filter to coastal workers only
        records = loader.filter_california_coastal(records)
        
        features_list = []
        error_count = 0
        for person in records:
            try:
                features = self.extract_features(person)
                features_list.append(features.to_dict())
            except Exception as e:
                # Track errors but continue
                error_count += 1
                if error_count <= 3:  # Print first 3 errors for debugging
                    print(f"Warning: Failed to extract features for person {person.id}: {e}")
                continue
        
        if not features_list:
            raise ValueError(f"No features extracted! Total errors: {error_count}. Check data format and feature extraction logic.")
        
        if error_count > 0:
            print(f"Note: Skipped {error_count} records due to extraction errors")
        
        return pd.DataFrame(features_list)


def prepare_training_data(
    data_path: str,
    max_records: Optional[int] = None,
    min_jobs: int = 2
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data for XGBoost model.
    
    Args:
        data_path: Path to Live Data JSONL files
        max_records: Maximum records to process
        min_jobs: Minimum job history length to include
    
    Returns:
        (X, y): Features dataframe and target series
    """
    extractor = DisplacementFeatureExtractor()
    df = extractor.extract_batch(data_path, max_records=max_records)
    
    # Filter to workers with sufficient history
    df = df[df['total_jobs'] >= min_jobs].copy()
    
    # Separate features and target
    y = df['displaced']
    X = df.drop(['person_id', 'displaced'], axis=1)
    
    print(f"Training data: {len(df)} workers")
    print(f"Displaced: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Features: {list(X.columns)}")
    
    return X, y


if __name__ == "__main__":
    # Test feature extraction
    import sys
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "live_data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("Extracting displacement features...")
    X, y = prepare_training_data(str(data_dir), max_records=1000)
    
    print("\nFeature statistics:")
    print(X.describe())
    
    print(f"\nTarget distribution:")
    print(y.value_counts())

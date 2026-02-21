"""
Housing Data Loader — APR (Annual Progress Report) Data
--------------------------------------------------------
Loads SBCAG Housing Data Dashboard export (APR_Download_2024.xlsx)

Source: California Department of Housing and Community Development
via Santa Barbara County Association of Governments (SBCAG)

Contains project-level housing data:
- Entitlements, building permits, certificates of occupancy
- By income level (Very Low / Low / Moderate / Above Moderate)
- By jurisdiction (9 cities + unincorporated county)
- By year (2018–2024)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class HousingDataLoader:
    """
    Loads and processes SBCAG APR housing data.
    
    Produces:
    - Jurisdiction-level housing production summaries
    - Affordability gap metrics
    - Housing pressure index for the resilience engine
    """
    
    INCOME_LEVELS = ['Very Low', 'Low', 'Moderate', 'Above Moderate']
    
    # RHNA 6th Cycle targets for Santa Barbara County (2023-2031)
    # Source: https://www.hcd.ca.gov/planning-and-community-development/regional-housing-needs-allocation
    RHNA_TARGETS = {
        'SANTA BARBARA': {'Very Low': 2915, 'Low': 1688, 'Moderate': 1614, 'Above Moderate': 3738},
        'SANTA BARBARA COUNTY': {'Very Low': 1288, 'Low': 756, 'Moderate': 728, 'Above Moderate': 1937},
        'SANTA MARIA': {'Very Low': 1792, 'Low': 1107, 'Moderate': 1004, 'Above Moderate': 2480},
        'GOLETA': {'Very Low': 747, 'Low': 432, 'Moderate': 411, 'Above Moderate': 928},
        'LOMPOC': {'Very Low': 375, 'Low': 228, 'Moderate': 206, 'Above Moderate': 544},
        'CARPINTERIA': {'Very Low': 222, 'Low': 132, 'Moderate': 118, 'Above Moderate': 260},
        'GUADALUPE': {'Very Low': 145, 'Low': 85, 'Moderate': 80, 'Above Moderate': 182},
        'BUELLTON': {'Very Low': 97, 'Low': 58, 'Moderate': 52, 'Above Moderate': 133},
        'SOLVANG': {'Very Low': 95, 'Low': 57, 'Moderate': 52, 'Above Moderate': 130},
    }
    
    def __init__(self, filepath: Union[str, Path] = 'APR_Download_2024.xlsx'):
        self.filepath = Path(filepath)
        self._df: Optional[pd.DataFrame] = None
    
    @property
    def df(self) -> pd.DataFrame:
        """Lazy-load the dataset."""
        if self._df is None:
            self._df = self._load()
        return self._df
    
    def _load(self) -> pd.DataFrame:
        """Load and clean the APR dataset."""
        logger.info(f"Loading housing data from {self.filepath}")
        df = pd.read_excel(self.filepath, sheet_name='APR (2018-2024 Data)')
        
        # Ensure numeric columns
        permit_cols = [c for c in df.columns if 'Permit' in c or 'Entitlement' in c 
                       or 'Cert.' in c or 'Total' in c or 'Units' in c]
        for col in permit_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info(f"Loaded {len(df)} housing records across {df['Year'].nunique()} years")
        return df
    
    def get_production_by_jurisdiction(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Summarize housing production by jurisdiction.
        
        Returns permits issued by income category.
        """
        data = self.df
        if year:
            data = data[data['Year'] == year]
        
        agg = data.groupby('Jurisdiction').agg(
            total_permits=('Total Number of Building Permits', 'sum'),
            total_certs=('Total Number of Units Issued Cert. of Occupancy', 'sum'),
            total_entitlements=('Total Entitlements', 'sum'),
            very_low_permits=('Very Low Income Building Permits', 'sum'),
            low_permits=('Low Income Building Permits', 'sum'),
            moderate_permits=('Moderate Income Building Permits', 'sum'),
            above_moderate_permits=('Above Moderate Income Building Permits', 'sum'),
            n_projects=('APN', 'count'),
        ).reset_index()
        
        # Affordable = Very Low + Low + Moderate
        agg['affordable_permits'] = (
            agg['very_low_permits'] + agg['low_permits'] + agg['moderate_permits']
        )
        agg['affordable_pct'] = (
            agg['affordable_permits'] / agg['total_permits'].replace(0, np.nan) * 100
        ).round(1)
        
        return agg
    
    def get_production_trend(self) -> pd.DataFrame:
        """
        Annual housing production trend for the whole county.
        """
        agg = self.df.groupby('Year').agg(
            total_permits=('Total Number of Building Permits', 'sum'),
            total_certs=('Total Number of Units Issued Cert. of Occupancy', 'sum'),
            very_low_permits=('Very Low Income Building Permits', 'sum'),
            low_permits=('Low Income Building Permits', 'sum'),
            moderate_permits=('Moderate Income Building Permits', 'sum'),
            above_moderate_permits=('Above Moderate Income Building Permits', 'sum'),
            n_projects=('APN', 'count'),
        ).reset_index()
        
        agg['affordable_permits'] = (
            agg['very_low_permits'] + agg['low_permits'] + agg['moderate_permits']
        )
        
        return agg
    
    def get_type_breakdown(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Housing type breakdown (ADU, Single-Family, Multi-Family, etc.)
        """
        data = self.df
        if year:
            data = data[data['Year'] == year]
        
        return data.groupby('Type').agg(
            total_permits=('Total Number of Building Permits', 'sum'),
            n_projects=('APN', 'count'),
        ).sort_values('total_permits', ascending=False).reset_index()
    
    def get_rhna_progress(self) -> pd.DataFrame:
        """
        Calculate RHNA progress for each jurisdiction.
        
        Compares actual permits issued (2023+) against 6th cycle targets.
        """
        # 6th cycle starts 2023
        cycle6 = self.df[self.df['Year'] >= 2023].copy()
        
        rows = []
        for jurisdiction, targets in self.RHNA_TARGETS.items():
            jur_data = cycle6[cycle6['Jurisdiction'] == jurisdiction]
            
            actual = {
                'Very Low': jur_data['Very Low Income Building Permits'].sum(),
                'Low': jur_data['Low Income Building Permits'].sum(),
                'Moderate': jur_data['Moderate Income Building Permits'].sum(),
                'Above Moderate': jur_data['Above Moderate Income Building Permits'].sum(),
            }
            
            total_target = sum(targets.values())
            total_actual = sum(actual.values())
            
            rows.append({
                'jurisdiction': jurisdiction,
                'total_target': total_target,
                'total_actual': int(total_actual),
                'progress_pct': round(total_actual / total_target * 100, 1) if total_target > 0 else 0,
                'affordable_target': targets['Very Low'] + targets['Low'] + targets['Moderate'],
                'affordable_actual': int(actual['Very Low'] + actual['Low'] + actual['Moderate']),
                **{f'target_{k.lower().replace(" ", "_")}': v for k, v in targets.items()},
                **{f'actual_{k.lower().replace(" ", "_")}': int(v) for k, v in actual.items()},
            })
        
        df = pd.DataFrame(rows)
        df['affordable_progress_pct'] = (
            df['affordable_actual'] / df['affordable_target'].replace(0, np.nan) * 100
        ).round(1)
        
        return df
    
    def get_housing_pressure_index(self) -> pd.DataFrame:
        """
        Compute a Housing Pressure Index for each jurisdiction.
        
        Components (0-100 scale, higher = more pressure):
        1. Production Gap: How far behind RHNA targets (40% weight)
        2. Affordability Gap: Shortage of affordable units (30% weight)
        3. ADU Dependence: Over-reliance on ADUs vs multi-family (15% weight)
        4. Rental Pressure: Renter-heavy production (15% weight)
        
        This feeds directly into the resilience model as an additional
        friction coefficient — areas with high housing pressure have
        lower labor resilience.
        """
        rhna = self.get_rhna_progress()
        production = self.get_production_by_jurisdiction()
        
        # Merge
        hpi = rhna.merge(production, left_on='jurisdiction', right_on='Jurisdiction', how='left')
        
        # 1. Production Gap Score (0-100)
        # 0% RHNA progress = 100, 100%+ = 0
        hpi['production_gap_score'] = (100 - hpi['progress_pct']).clip(0, 100)
        
        # 2. Affordability Gap Score (0-100)
        # If affordable progress is much lower than overall, score is higher
        hpi['affordability_gap_score'] = (100 - hpi['affordable_progress_pct'].fillna(0)).clip(0, 100)
        
        # 3. ADU Dependence Score (0-100)
        # County-level ADU stats by jurisdiction
        adu_data = self.df[self.df['Type'] == 'Accessory Dwelling Unit'].groupby('Jurisdiction').agg(
            adu_permits=('Total Number of Building Permits', 'sum')
        ).reset_index()
        hpi = hpi.merge(adu_data, left_on='jurisdiction', right_on='Jurisdiction', 
                        how='left', suffixes=('', '_adu'))
        hpi['adu_permits'] = hpi['adu_permits'].fillna(0)
        hpi['adu_ratio'] = (hpi['adu_permits'] / hpi['total_permits'].replace(0, np.nan)).fillna(0)
        hpi['adu_dependence_score'] = (hpi['adu_ratio'] * 100).clip(0, 100)
        
        # 4. Rental Pressure Score (0-100)
        rental_data = self.df[self.df['Tenure'] == 'Renter'].groupby('Jurisdiction').agg(
            rental_permits=('Total Number of Building Permits', 'sum')
        ).reset_index()
        hpi = hpi.merge(rental_data, left_on='jurisdiction', right_on='Jurisdiction',
                        how='left', suffixes=('', '_rental'))
        hpi['rental_permits'] = hpi['rental_permits'].fillna(0)
        hpi['rental_ratio'] = (hpi['rental_permits'] / hpi['total_permits'].replace(0, np.nan)).fillna(0)
        hpi['rental_pressure_score'] = (hpi['rental_ratio'] * 100).clip(0, 100)
        
        # Composite Housing Pressure Index
        hpi['housing_pressure_index'] = (
            hpi['production_gap_score'] * 0.40 +
            hpi['affordability_gap_score'] * 0.30 +
            hpi['adu_dependence_score'] * 0.15 +
            hpi['rental_pressure_score'] * 0.15
        ).round(1)
        
        # Select clean output
        result = hpi[[
            'jurisdiction', 'total_target', 'total_actual', 'progress_pct',
            'affordable_target', 'affordable_actual', 'affordable_progress_pct',
            'production_gap_score', 'affordability_gap_score', 
            'adu_dependence_score', 'rental_pressure_score',
            'housing_pressure_index'
        ]].copy()
        
        result['pressure_level'] = pd.cut(
            result['housing_pressure_index'],
            bins=[0, 40, 60, 80, 100],
            labels=['Low', 'Moderate', 'High', 'Critical']
        )
        
        return result


def load_housing_data(filepath: Union[str, Path] = 'APR_Download_2024.xlsx') -> dict:
    """
    Convenience function to load all housing metrics.
    
    Returns dict with:
    - 'loader': HousingDataLoader instance
    - 'production': Jurisdiction production summary
    - 'trend': Annual production trend
    - 'rhna': RHNA progress
    - 'pressure': Housing Pressure Index
    """
    loader = HousingDataLoader(filepath)
    
    return {
        'loader': loader,
        'production': loader.get_production_by_jurisdiction(),
        'trend': loader.get_production_trend(),
        'rhna': loader.get_rhna_progress(),
        'pressure': loader.get_housing_pressure_index(),
    }


if __name__ == '__main__':
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else 'APR_Download_2024.xlsx'
    data = load_housing_data(path)
    
    print("=" * 60)
    print("HOUSING PRESSURE INDEX — Santa Barbara County")
    print("=" * 60)
    
    hpi = data['pressure']
    print(hpi[['jurisdiction', 'progress_pct', 'affordable_progress_pct', 
               'housing_pressure_index', 'pressure_level']].to_string(index=False))
    
    print(f"\n{'='*60}")
    print("ANNUAL PRODUCTION TREND")
    print("=" * 60)
    trend = data['trend']
    print(trend[['Year', 'total_permits', 'affordable_permits', 'n_projects']].to_string(index=False))
    
    print(f"\n{'='*60}")
    print("RHNA 6TH CYCLE PROGRESS")
    print("=" * 60)
    rhna = data['rhna']
    print(rhna[['jurisdiction', 'total_target', 'total_actual', 'progress_pct']].to_string(index=False))

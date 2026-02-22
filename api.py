"""
FastAPI backend for the Coastal Labor-Resilience Engine.
Serves all data to the React frontend.
"""
import sys
sys.path.insert(0, '.')

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
import json, os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.models.markov_chain import (
    MarkovTransitionMatrix, LaborState, ShockParameters,
    STATE_INDEX, create_regional_chain
)
from src.models.resilience_ode import (
    ResilienceODE, ODEParameters, ClimateShock,
    create_ej_tract, create_shock_scenario
)

# Data loaders
try:
    from src.data.real_data_fetcher import (
        CensusBureauClient, NOAAClient, FEMAFloodClient, BLSClient
    )
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

try:
    from src.data.live_data_loader import LiveDataLoader
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False

try:
    from src.data.housing_loader import HousingDataLoader, load_housing_data
    HOUSING_DATA_AVAILABLE = Path('APR_Download_2024.xlsx').exists()
except ImportError:
    HOUSING_DATA_AVAILABLE = False

LIVE_DATA_DIR = Path("drive-download-20260221T181111Z-3-001")

app = FastAPI(title="Coastal Resilience API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Caches ───────────────────────────────────────────────────────────────────
_tract_cache = None
_noaa_cache = None
_workforce_cache = None
_housing_cache = None


def _get_synthetic_tracts():
    return pd.DataFrame({
        'tract_id': ['06083001500','06083001600','06083001701','06083001702',
                      '06083001800','06083001900','06083002001','06083002002',
                      '06083002100','06083002200','06083002300','06083002400',
                      '06083002500','06083002600','06083002700','06083002800'],
        'name': [f'Tract {x}' for x in ['1.5','1.6','1.7.01','1.7.02','1.8','1.9',
                 '2.0.01','2.0.02','2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8']],
        'lat': [34.4208,34.4059,34.4122,34.4085,34.4289,34.4256,34.4355,34.4480,
                34.4412,34.4389,34.4298,34.4352,34.4521,34.4140,34.4133,34.3988],
        'lon': [-119.6982,-119.6846,-119.7156,-119.7298,-119.6845,-119.7145,-119.7012,
                -119.7089,-119.7234,-119.7389,-119.7598,-119.8245,-119.8156,-119.8412,
                -119.8567,-119.5234],
        'ej_percentile': [55,68,42,38,72,45,58,35,32,28,18,52,48,65,78,62],
        'coastal_jobs_pct': [45,72,35,28,55,38,42,22,18,15,12,48,35,58,68,65],
        'population': [4200,3800,5100,4800,6200,5500,4100,3900,4500,4200,2800,5800,6100,8500,12000,5200],
        'median_income': [52000,45000,78000,85000,38000,72000,58000,95000,102000,115000,185000,62000,68000,28000,22000,55000],
        'limited_english': [12,18,5,4,22,8,14,3,2,2,1,15,12,8,25,18],
        'flood_zone': [True,True,False,False,False,False,True,False,False,False,True,True,False,True,True,True],
        'high_poverty': [False,True,False,False,True,False,False,False,False,False,False,False,False,True,True,False],
        'poverty_pct': [12,18,8,6,22,10,14,5,4,3,2,13,11,20,28,16],
        'minority_pct': [40,55,25,20,60,30,45,18,15,12,8,42,35,50,65,48],
        'limited_english_pct': [12,18,5,4,22,8,14,3,2,2,1,15,12,8,25,18],
    })


def _assign_city(lat: float, lon: float) -> str:
    """
    Map a lat/lon to the nearest Santa Barbara County city/community
    using approximate bounding boxes. Most-specific boxes listed first.
    """
    CITY_BOXES = [
        # (lat_min, lat_max, lon_min, lon_max, name)
        (34.95, 35.02, -120.61, -120.55, "Guadalupe"),
        (34.85, 34.95, -120.51, -120.38, "Orcutt"),
        (34.89, 35.02, -120.52, -120.37, "Santa Maria"),
        (34.62, 34.69, -120.53, -120.42, "Lompoc"),
        (34.70, 34.76, -120.54, -120.44, "Vandenberg Village"),
        (34.60, 34.64, -120.24, -120.18, "Buellton"),
        (34.59, 34.63, -120.17, -120.06, "Solvang / Santa Ynez"),
        (34.64, 34.71, -120.14, -120.06, "Los Olivos"),
        (34.40, 34.44, -119.89, -119.84, "Isla Vista"),
        (34.40, 34.49, -120.04, -119.81, "Goleta"),
        (34.41, 34.46, -119.67, -119.59, "Montecito"),
        (34.38, 34.43, -119.54, -119.47, "Carpinteria"),
        (34.37, 34.48, -119.89, -119.60, "Santa Barbara"),
    ]
    for lat_min, lat_max, lon_min, lon_max, city in CITY_BOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return city
    return "Unincorporated SB County"


def get_tracts():
    global _tract_cache
    if _tract_cache is not None:
        return _tract_cache

    if not REAL_DATA_AVAILABLE:
        _tract_cache = _get_synthetic_tracts()
        return _tract_cache

    try:
        census = CensusBureauClient()
        df = census.get_tract_demographics('06083')
        if df.empty:
            _tract_cache = _get_synthetic_tracts()
            return _tract_cache

        geometries = census.get_tract_geometries('06083')
        centroid_map = {}
        for feature in geometries.get('features', []):
            props = feature.get('properties', {})
            geoid = props.get('GEOID', '')[:11]
            if geoid not in centroid_map:
                centroid_map[geoid] = {
                    'lat': float(props.get('CENTLAT', 34.42)),
                    'lon': float(props.get('CENTLON', -119.70))
                }

        df['tract_id'] = df['GEOID']
        df['name'] = df['NAME'].str.replace('; Santa Barbara County; California', '', regex=False)
        df['name'] = df['name'].str.replace('Census Tract ', 'Tract ', regex=False)
        df['lat'] = df['GEOID'].map(lambda g: centroid_map.get(g, {}).get('lat', 34.42))
        df['lon'] = df['GEOID'].map(lambda g: centroid_map.get(g, {}).get('lon', -119.70))
        df['population'] = df.get('population', df.get('B01001_001E', 5000)).fillna(5000).astype(int)
        df['median_income'] = df.get('median_income', df.get('B19013_001E', 50000)).fillna(50000).astype(int)
        df['poverty_pct'] = df.get('poverty_pct', 10).fillna(10)
        df['minority_pct'] = df.get('minority_pct', 30).fillna(30)
        df['limited_english_pct'] = df.get('limited_english_pct', 5).fillna(5)
        df['ej_percentile'] = (
            (df['poverty_pct'] * 0.4) + (df['minority_pct'] * 0.3) + (df['limited_english_pct'] * 0.3)
        ).clip(0, 100).round(0).astype(int)

        bls = BLSClient()
        county_coastal_pct = bls.get_coastal_employment_pct('06083')
        df['coast_factor'] = ((df['lon'] + 120.1) / -0.6).clip(0.5, 1.5)
        df['coastal_jobs_pct'] = (county_coastal_pct * df['coast_factor']).round(0).astype(int)
        df['limited_english'] = df['limited_english_pct'].round(0).astype(int)
        df['high_poverty'] = df['poverty_pct'] > 15

        fema = FEMAFloodClient()
        flood_zones = []
        for _, row in df.iterrows():
            zone_info = fema.get_flood_zones_for_point(row['lat'], row['lon'])
            flood_zones.append(zone_info.get('special_flood_hazard', False))
        df['flood_zone'] = flood_zones

        _tract_cache = df
        return df
    except Exception:
        _tract_cache = _get_synthetic_tracts()
        return _tract_cache


def calc_exodus_prob(ej_pct, coastal_pct, severity):
    """
    Exodus probability based on EJ burden, coastal job dependency, and shock severity.
    Calibrated to produce a realistic spread across vulnerability categories.
    
    Factors:
    - EJ percentile: higher burden → higher flight risk (dominant driver)
    - Coastal job share: more coastal-dependent → more exposure
    - Shock severity: stronger shock → more displacement
    """
    # Normalize EJ percentile to 0-1 range (these are county-relative percentiles)
    ej_norm = ej_pct / 100.0
    
    # Coastal dependency: normalize assuming max ~25% coastal jobs in any tract
    coastal_norm = min(coastal_pct / 25.0, 1.0)
    
    # Composite vulnerability score (EJ is primary driver, coastal is secondary)
    vuln_score = 0.6 * ej_norm + 0.4 * coastal_norm
    
    # Apply severity scaling — at severity=0.5, moderate effect; at 1.0, maximum
    shock_multiplier = 0.5 + severity * 1.5  # range: 0.5 to 2.0
    
    # Map to probability using logistic-like curve for realistic spread
    # Produces full range across Low/Moderate/High/Critical
    raw = vuln_score * shock_multiplier
    prob = 0.02 + 0.80 * (raw ** 1.0)  # linear scaling for natural spread
    
    return min(max(prob, 0.02), 0.95)


def run_simulation(severity, duration, start_day, r, K, ej_percentile, n_days):
    base_beta = 0.001 + 0.009 * (ej_percentile / 100)
    params = ODEParameters(r=r, K=K, beta=base_beta)
    ode = ResilienceODE(params)
    shock = ClimateShock(
        start_time=float(start_day), duration=float(duration), severity=severity,
        K_reduction=0.3 * severity, beta_increase=0.02 * severity
    )
    ode.add_shock(shock)
    solution = ode.solve(L0=0.92, t_span=(0, n_days), n_points=min(n_days, 200))
    return {
        't': solution.t.tolist(),
        'L': solution.L.tolist(),
        'min_labor_force': float(solution.min_labor_force) if solution.min_labor_force else 0.0,
        'equilibrium': float(solution.equilibrium) if solution.equilibrium else 0.0,
        'recovery_time': float(solution.recovery_time) if solution.recovery_time else None,
        'resilience_score': float(solution.resilience_score) if solution.resilience_score else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/tracts")
def api_tracts(severity: float = 0.5):
    df = get_tracts().copy()
    df['exodus_prob'] = df.apply(
        lambda r: calc_exodus_prob(r['ej_percentile'], r['coastal_jobs_pct'], severity), axis=1
    )
    # Use quantile-based bins so all 4 categories are always represented
    try:
        df['vulnerability'] = pd.qcut(
            df['exodus_prob'], q=4, labels=['Low', 'Moderate', 'High', 'Critical']
        ).astype(str)
    except ValueError:
        # Fallback if too many duplicate edges
        bins = [0, 0.15, 0.30, 0.50, 1.0]
        labels = ['Low', 'Moderate', 'High', 'Critical']
        df['vulnerability'] = pd.cut(df['exodus_prob'], bins=bins, labels=labels).astype(str)

    df['city'] = df.apply(lambda r: _assign_city(r['lat'], r['lon']), axis=1)

    cols = ['tract_id','name','city','lat','lon','population','median_income','poverty_pct',
            'minority_pct','limited_english_pct','limited_english','ej_percentile',
            'coastal_jobs_pct','flood_zone','high_poverty','exodus_prob','vulnerability']
    return df[cols].to_dict(orient='records')


@app.get("/api/simulation")
def api_simulation(
    severity: float = 0.5,
    duration: int = 21,
    start_day: int = 30,
    r: float = 0.10,
    K: float = 0.95,
    sim_days: int = 365,
):
    sol = run_simulation(severity, duration, start_day, r, K, 50, sim_days)
    min_emp = sol['min_labor_force']
    eq = sol['equilibrium']
    labor_flight = (1 - min_emp) * 100
    ej_gap = (eq - 0.92) * 100

    df = get_tracts().copy()
    df['exodus_prob'] = df.apply(
        lambda r_: calc_exodus_prob(r_['ej_percentile'], r_['coastal_jobs_pct'], severity), axis=1
    )
    try:
        df['vulnerability'] = pd.qcut(
            df['exodus_prob'], q=4, labels=['Low','Moderate','High','Critical']
        ).astype(str)
    except ValueError:
        df['vulnerability'] = pd.cut(df['exodus_prob'], bins=[0,0.15,0.30,0.50,1.0],
                                      labels=['Low','Moderate','High','Critical']).astype(str)
    critical = int((df['vulnerability'] == 'Critical').sum())
    emergency_fund = labor_flight * 50000

    return {
        'resilience_score': sol['resilience_score'],
        'labor_flight_pct': round(labor_flight, 1),
        'recovery_time': sol['recovery_time'],
        'ej_gap': round(ej_gap, 1),
        'min_employment': round(min_emp * 100, 1),
        'equilibrium': round(eq * 100, 1),
        'critical_tracts': critical,
        'emergency_fund': round(emergency_fund),
        'status': 'RESILIENT' if sol['resilience_score'] > 0.7 else 'VULNERABLE',
    }


@app.get("/api/simulation/compare")
def api_compare(
    severity: float = 0.5,
    duration: int = 21,
    start_day: int = 30,
    r: float = 0.10,
    K: float = 0.95,
    sim_days: int = 365,
):
    profiles = [
        ('Low Burden', 20), ('Average', 50),
        ('High Burden', 75), ('Extreme', 90),
    ]
    results = {}
    for name, ej in profiles:
        sol = run_simulation(severity, duration, start_day, r, K, ej, sim_days)
        # downsample to 100 points
        step = max(1, len(sol['t']) // 100)
        results[name] = {
            't': sol['t'][::step],
            'L': [round(v * 100, 2) for v in sol['L'][::step]],
            'recovery_time': sol['recovery_time'],
            'resilience_score': sol['resilience_score'],
        }
    return {
        'profiles': results,
        'shock': {'start': start_day, 'end': start_day + duration, 'severity': severity},
        'baseline': 92,
    }


@app.get("/api/noaa")
def api_noaa():
    if not REAL_DATA_AVAILABLE:
        return {'data': [], 'stats': {}}
    try:
        noaa = NOAAClient()
        df = noaa.get_water_levels(hours=168)
        if df.empty:
            return {'data': [], 'stats': {}}
        records = df.to_dict(orient='records')
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = None
                elif hasattr(v, 'isoformat'):
                    r[k] = v.isoformat()
        stats = {
            'max': round(df['water_level'].max(), 2) if 'water_level' in df else None,
            'min': round(df['water_level'].min(), 2) if 'water_level' in df else None,
            'mean': round(df['water_level'].mean(), 2) if 'water_level' in df else None,
            'high_events': int((df['water_level'] > 5.5).sum()) if 'water_level' in df else 0,
        }
        return {'data': records, 'stats': stats}
    except Exception:
        return {'data': [], 'stats': {}}


@app.get("/api/workforce")
def api_workforce():
    global _workforce_cache
    if _workforce_cache is not None:
        return _workforce_cache

    if not LIVE_DATA_AVAILABLE or not LIVE_DATA_DIR.exists():
        _workforce_cache = {'persons': [], 'transitions': [], 'industries': [], 'stats': {}}
        return _workforce_cache

    try:
        loader = LiveDataLoader(LIVE_DATA_DIR)
        persons = loader.load_to_dataframe(filter_coastal=True, max_records=5000)
        transitions = loader.get_job_transitions(filter_coastal=True, max_records=5000)
        industries = loader.get_industry_distribution(filter_coastal=True, max_records=5000)

        top_transitions = []
        if transitions is not None and not transitions.empty:
            trans_summary = transitions.groupby(['from_industry', 'to_industry']).size().reset_index(name='count')
            trans_summary = trans_summary.nlargest(10, 'count')
            top_transitions = trans_summary.to_dict(orient='records')

        industry_list = []
        if industries is not None and not industries.empty:
            industry_list = industries.head(15).to_dict(orient='records')

        county_counts = {}
        if persons is not None and not persons.empty and 'current_county' in persons.columns:
            county_counts = persons['current_county'].value_counts().head(8).to_dict()

        _workforce_cache = {
            'stats': {
                'total_workers': len(persons) if persons is not None else 0,
                'total_transitions': len(transitions) if transitions is not None else 0,
            },
            'transitions': top_transitions,
            'industries': industry_list,
            'county_distribution': county_counts,
        }
        return _workforce_cache
    except Exception:
        _workforce_cache = {'persons': [], 'transitions': [], 'industries': [], 'stats': {}}
        return _workforce_cache


@app.get("/api/housing")
def api_housing():
    global _housing_cache
    if _housing_cache is not None:
        return _housing_cache

    if not HOUSING_DATA_AVAILABLE:
        _housing_cache = {'pressure': [], 'trend': [], 'stats': {}}
        return _housing_cache

    try:
        loader = HousingDataLoader('APR_Download_2024.xlsx')
        pressure = loader.get_housing_pressure_index()
        trend = loader.get_production_trend()

        pressure_list = pressure.to_dict(orient='records')
        trend_list = trend.to_dict(orient='records')

        avg_pressure = pressure['housing_pressure_index'].mean()
        critical_count = (pressure['pressure_level'] == 'Critical').sum()
        total_progress = pressure['progress_pct'].mean() if 'progress_pct' in pressure else 0

        _housing_cache = {
            'pressure': pressure_list,
            'trend': trend_list,
            'stats': {
                'avg_pressure': round(avg_pressure, 1),
                'critical_jurisdictions': int(critical_count),
                'total_jurisdictions': len(pressure),
                'avg_rhna_progress': round(total_progress, 1),
            }
        }
        return _housing_cache
    except Exception as e:
        _housing_cache = {'pressure': [], 'trend': [], 'stats': {}, 'error': str(e)}
        return _housing_cache


@app.get("/api/markov")
def api_markov(severity: float = 0.5, duration: int = 21):
    try:
        chain = create_regional_chain()
        shock = ShockParameters(severity=severity, duration_days=duration)
        shocked_matrix = chain.apply_shock(shock)

        coastal_idx = STATE_INDEX[LaborState.COASTAL_EMPLOYED]
        row = shocked_matrix[coastal_idx]

        return {
            'stayed_coastal': round(float(row[STATE_INDEX[LaborState.COASTAL_EMPLOYED]]) * 100, 1),
            'to_inland': round(float(row[STATE_INDEX[LaborState.INLAND_EMPLOYED]]) * 100, 1),
            'to_unemployed': round(float(row[STATE_INDEX[LaborState.UNEMPLOYED]]) * 100, 1),
            'to_transitioning': round(float(row[STATE_INDEX[LaborState.TRANSITIONING]]) * 100, 1),
        }
    except Exception:
        return {
            'stayed_coastal': 70, 'to_inland': 15,
            'to_unemployed': 10, 'to_transitioning': 5,
        }


# ── GitHub Models (Copilot) Chat ─────────────────────────────────────────────────
from fastapi import Body
from openai import OpenAI

@app.post("/api/chat")
async def chat(payload: dict = Body(...)):
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        return {"reply": "GITHUB_TOKEN not set in environment. Add it to your .env file."}

    message    = payload.get("message", "")
    tract      = payload.get("tract", {})
    params     = payload.get("params", {})
    sim_data   = payload.get("simData", {})
    all_tracts = payload.get("allTracts", [])

    pop = tract.get('population')
    income = tract.get('median_income', 0) or 0
    emergency = sim_data.get('emergency_fund', 0) or 0

    tract_selected = bool(tract.get('name'))

    # Build compact all-tracts table (sorted by exodus_prob desc)
    sorted_tracts = sorted(all_tracts, key=lambda t: t.get('exodus_prob', 0), reverse=True)
    tracts_table = "\n".join([
        f"  [{t.get('city','?')}] {t.get('name','?')}: pop={t.get('population','?')}, income=${t.get('median_income',0):,}, "
        f"EJ={t.get('ej_percentile','?')}%, coastal={t.get('coastal_jobs_pct','?')}%, "
        f"exodus={round((t.get('exodus_prob',0) or 0)*100,1)}%, vuln={t.get('vulnerability','?')}, "
        f"flood={'yes' if t.get('flood_zone') else 'no'}, poverty={t.get('poverty_pct','?')}%"
        for t in sorted_tracts
    ]) if sorted_tracts else "  (not loaded yet)"

    # Build city-level averages table
    from collections import defaultdict
    city_groups: dict = defaultdict(list)
    for t in all_tracts:
        city_groups[t.get('city', 'Unincorporated SB County')].append(t)
    city_rows = []
    for city, tracts_in_city in sorted(city_groups.items()):
        n = len(tracts_in_city)
        avg_exodus = sum((t.get('exodus_prob', 0) or 0) for t in tracts_in_city) / n
        avg_ej = sum((t.get('ej_percentile', 0) or 0) for t in tracts_in_city) / n
        avg_income = sum((t.get('median_income', 0) or 0) for t in tracts_in_city) / n
        total_pop = sum((t.get('population', 0) or 0) for t in tracts_in_city)
        flood_count = sum(1 for t in tracts_in_city if t.get('flood_zone'))
        city_rows.append(
            f"  {city} ({n} tracts, pop={total_pop:,}): avg_exodus={round(avg_exodus*100,1)}%, "
            f"avg_EJ={round(avg_ej,1)}%, avg_income=${avg_income:,.0f}, flood_tracts={flood_count}"
        )
    city_table = "\n".join(city_rows) if city_rows else "  (not loaded yet)"

    tract_selected = bool(tract.get('name'))

    system_prompt = f"""You are a coastal resilience policy analyst for Santa Barbara County, California.
You have access to real-time data from the Coastal Labor-Resilience Engine dashboard.

CITY / COMMUNITY SUMMARY (aggregated from all tracts):
{city_table}

ALL 109 CENSUS TRACTS (sorted by exodus risk, highest first; city shown in brackets):
{tracts_table}

SELECTED TRACT: {tract.get('name', 'None (no tract clicked on map)')}
- Population: {f"{pop:,}" if isinstance(pop, (int, float)) else 'N/A'}
- EJ Burden Percentile: {tract.get('ej_percentile', 'N/A')}%
- Median Household Income: ${income:,.0f}
- Coastal Jobs: {tract.get('coastal_jobs_pct', 'N/A')}%
- Exodus Probability: {round((tract.get('exodus_prob', 0) or 0) * 100, 1)}%

CURRENT SCENARIO:
- Storm Severity: {round((params.get('severity', 0.5)) * 100)}%
- Duration: {params.get('duration', 21)} days
- Recovery Rate (r): {params.get('r', 0.1)}
- Carrying Capacity (K): {round((params.get('K', 0.95)) * 100)}%

SIMULATION RESULTS:
- Resilience Score: {sim_data.get('resilience_score', 'N/A')}
- Recovery Time: {round(sim_data.get('recovery_time', 0) or 0)} days
- Min Labor Force: {round((sim_data.get('min_labor_force', 0) or 0) * 100, 1)}%
- Labor Flight: {sim_data.get('labor_flight_pct', 'N/A')}%
- Emergency Fund: ${emergency:,.0f}

INSTRUCTIONS:
- You have ALL tract data in the tables above — use it to answer questions about any specific tract by name or any city/community without requiring the user to click anything.
- For city-level questions (e.g. "average exodus risk in Santa Maria"), use the CITY/COMMUNITY SUMMARY table directly.
- If a SELECTED TRACT is shown, prioritize its data for context.
- Answer concisely and specifically. Focus on actionable policy insights.
- When comparing tracts or ranking by risk, use the full tables above.
- Cities/communities in Santa Barbara County include: Santa Barbara, Goleta, Montecito, Carpinteria, Isla Vista, Lompoc, Buellton, Solvang / Santa Ynez, Los Olivos, Santa Maria, Orcutt, Guadalupe, Vandenberg Village, Unincorporated SB County."""

    try:
        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=github_token,
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=512,
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        err = str(e)
        if '429' in err:
            return {"reply": "Rate limit reached. Please try again in a moment."}
        return {"reply": f"Chat error: {err}"}


# Serve React build if it exists
build_path = Path("frontend/dist")
if build_path.exists():
    app.mount("/", StaticFiles(directory=str(build_path), html=True), name="static")

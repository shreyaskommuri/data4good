"""
Coastal Labor-Resilience Engine
Command Center Dashboard

A premium dark-mode visualization platform for policymakers.
Design: Apple/Stripe aesthetic with glassmorphism.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Import our models
import sys
sys.path.insert(0, '.')

from src.models.markov_chain import (
    MarkovTransitionMatrix,
    LaborState,
    ShockParameters,
    STATE_INDEX,
    create_regional_chain
)
from src.models.resilience_ode import (
    ResilienceODE,
    ODEParameters,
    ClimateShock,
    create_ej_tract,
    create_shock_scenario
)

# Import Live Data loader
from pathlib import Path
try:
    from src.data.live_data_loader import LiveDataLoader, load_live_data
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False

# Live Data directory
LIVE_DATA_DIR = Path("drive-download-20260221T181111Z-3-001")

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Coastal Labor-Resilience Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PREMIUM DARK MODE CSS
# ============================================================================

st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a24;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --text-muted: #606070;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #22c55e;
    }
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary);
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 100%;
    }
    
    /* Typography */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        flex: 1;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.1);
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    .metric-delta.negative {
        color: var(--danger);
    }
    
    .metric-delta.positive {
        color: var(--success);
    }
    
    /* Severity slider */
    .severity-control {
        background: var(--bg-secondary);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
    }
    
    .severity-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .severity-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .severity-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        background: var(--accent-gradient);
    }
    
    /* Custom slider styling */
    .stSlider > div > div > div {
        background: var(--bg-tertiary) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--accent-gradient) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 2px solid var(--accent-blue) !important;
    }
    
    /* Sidebar panel */
    .sidebar-panel {
        background: var(--glass-bg);
        backdrop-filter: blur(30px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        height: fit-content;
    }
    
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .sidebar-subheader {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }
    
    /* Vulnerability score */
    .vuln-score {
        display: flex;
        align-items: baseline;
        gap: 0.25rem;
        margin-bottom: 1.5rem;
    }
    
    .vuln-number {
        font-size: 4rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, #ef4444 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    .vuln-max {
        font-size: 1.5rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    /* EJ Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .ej-badge {
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.35rem 0.75rem;
        border-radius: 100px;
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .ej-badge.warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    .ej-badge.info {
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Timeline section */
    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .timeline-title {
        font-size: 1.125rem;
        font-weight: 600;
    }
    
    .recovery-estimate {
        font-size: 0.875rem;
        color: var(--text-secondary);
        padding: 0.5rem 1rem;
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 8px;
    }
    
    .recovery-estimate strong {
        color: var(--accent-blue);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Export button */
    .export-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.875rem 1.5rem;
        background: var(--accent-gradient);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .export-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--glass-border);
        display: flex;
        justify-content: center;
        gap: 2rem;
        color: var(--text-muted);
        font-size: 0.75rem;
    }
    
    .footer-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sankey diagram container */
    .sankey-container {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Plotly chart overrides */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
    
    .js-plotly-plot .plotly .modebar-btn path {
        fill: var(--text-muted) !important;
    }
    
    /* Tract list */
    .tract-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background: var(--bg-secondary);
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .tract-item:hover {
        background: var(--bg-tertiary);
    }
    
    .tract-item.critical {
        border-left-color: var(--danger);
    }
    
    .tract-item.high {
        border-left-color: var(--warning);
    }
    
    .tract-item.moderate {
        border-left-color: var(--accent-blue);
    }
    
    .tract-name {
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .tract-prob {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Streamlit overrides */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border-color: var(--glass-border) !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }
    
    div[data-baseweb="select"] > div {
        background: var(--bg-secondary) !important;
        border-color: var(--glass-border) !important;
    }
    
    /* Pulse animation for live data */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: var(--success);
        font-weight: 500;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SANTA BARBARA CENSUS TRACT DATA
# ============================================================================

@st.cache_data
def get_sb_tracts():
    """Santa Barbara census tract data with EJ indicators."""
    return pd.DataFrame({
        'tract_id': [
            '06083001500', '06083001600', '06083001701', '06083001702',
            '06083001800', '06083001900', '06083002001', '06083002002',
            '06083002100', '06083002200', '06083002300', '06083002400',
            '06083002500', '06083002600', '06083002700', '06083002800'
        ],
        'name': [
            'Downtown', 'Waterfront', 'Mesa East', 'Mesa West',
            'Eastside', 'Westside', 'Lower State', 'Upper State',
            'San Roque', 'Samarkand', 'Hope Ranch', 'Goleta South',
            'Goleta North', 'UCSB', 'Isla Vista', 'Carpinteria'
        ],
        'lat': [
            34.4208, 34.4059, 34.4122, 34.4085,
            34.4289, 34.4256, 34.4355, 34.4480,
            34.4412, 34.4389, 34.4298, 34.4352,
            34.4521, 34.4140, 34.4133, 34.3988
        ],
        'lon': [
            -119.6982, -119.6846, -119.7156, -119.7298,
            -119.6845, -119.7145, -119.7012, -119.7089,
            -119.7234, -119.7389, -119.7598, -119.8245,
            -119.8156, -119.8412, -119.8567, -119.5234
        ],
        'ej_percentile': [55, 68, 42, 38, 72, 45, 58, 35, 32, 28, 18, 52, 48, 65, 78, 62],
        'coastal_jobs_pct': [45, 72, 35, 28, 55, 38, 42, 22, 18, 15, 12, 48, 35, 58, 68, 65],
        'population': [4200, 3800, 5100, 4800, 6200, 5500, 4100, 3900, 4500, 4200, 2800, 5800, 6100, 8500, 12000, 5200],
        'median_income': [52000, 45000, 78000, 85000, 38000, 72000, 58000, 95000, 102000, 115000, 185000, 62000, 68000, 28000, 22000, 55000],
        'limited_english': [12, 18, 5, 4, 22, 8, 14, 3, 2, 2, 1, 15, 12, 8, 25, 18],
        'flood_zone': [True, True, False, False, False, False, True, False, False, False, True, True, False, True, True, True],
        'high_poverty': [False, True, False, False, True, False, False, False, False, False, False, False, False, True, True, False]
    })

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

@st.cache_data
def run_simulation(severity, duration, start_day, r, K, ej_percentile, n_days):
    """Run the ODE simulation."""
    # Compute beta from EJ percentile
    beta = 0.01 + 0.99 * (ej_percentile / 100) * 0.5
    
    params = ODEParameters(r=r, K=K, beta=beta)
    ode = ResilienceODE(params)
    
    shock = ClimateShock(
        start_time=float(start_day),
        duration=float(duration),
        severity=severity,
        K_reduction=0.35 * severity,
        beta_increase=0.25 * severity
    )
    ode.add_shock(shock)
    
    solution = ode.solve(L0=0.92, t_span=(0, n_days), n_points=n_days)
    return solution, beta

# ============================================================================
# LIVE DATA LOADING
# ============================================================================

@st.cache_data
def load_workforce_data():
    """Load real workforce data from Live Data Technologies files."""
    if not LIVE_DATA_AVAILABLE or not LIVE_DATA_DIR.exists():
        return None, None, None
    
    try:
        loader = LiveDataLoader(LIVE_DATA_DIR)
        
        # Load persons (California coastal, max 5000 for performance)
        persons_df = loader.load_to_dataframe(filter_coastal=True, max_records=5000)
        
        # Load job transitions
        transitions_df = loader.get_job_transitions(filter_coastal=True, max_records=5000)
        
        # Industry stats
        industry_df = loader.get_industry_distribution(filter_coastal=True, max_records=5000)
        
        return persons_df, transitions_df, industry_df
    except Exception as e:
        st.sidebar.error(f"Live Data load error: {e}")
        return None, None, None

@st.cache_data
def run_comparison(severity, duration, start_day, r, K, n_days):
    """Run simulation for all EJ profiles."""
    profiles = [
        ('Low Burden', 20),
        ('Average', 50),
        ('High Burden', 75),
        ('Extreme', 90)
    ]
    results = {}
    for name, ej in profiles:
        sol, beta = run_simulation(severity, duration, start_day, r, K, ej, n_days)
        results[name] = {'solution': sol, 'beta': beta}
    return results

def calc_exodus_prob(ej_pct, coastal_pct, severity):
    """Calculate exodus probability for a tract."""
    base_prob = 0.04
    ej_factor = 1 + (ej_pct / 100) * 2
    coastal_factor = coastal_pct / 50
    shock_factor = 1 + severity * 3
    return min(base_prob * ej_factor * coastal_factor * shock_factor, 0.95)

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Header
st.markdown('<h1 class="hero-title">Coastal Labor-Resilience Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Real-time labor market dynamics modeling for Santa Barbara County under climate scenarios</p>', unsafe_allow_html=True)

# ============================================================================
# LIVE DATA STATUS
# ============================================================================

# Load real workforce data
persons_df, transitions_df, industry_df = load_workforce_data()
has_live_data = persons_df is not None and len(persons_df) > 0

if has_live_data:
    n_workers = len(persons_df)
    n_transitions = len(transitions_df) if transitions_df is not None else 0
    top_industries = persons_df['current_industry'].value_counts().head(3).index.tolist() if 'current_industry' in persons_df.columns else []
    coastal_counties = persons_df['current_county'].value_counts().head(5) if 'current_county' in persons_df.columns else None
    
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 1.5rem; padding: 1rem 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="live-indicator">
                    <span class="live-dot"></span>
                    <span>LIVE DATA CONNECTED</span>
                </div>
                <span style="color: var(--text-muted); font-size: 0.875rem;">
                    Live Data Technologies API
                </span>
            </div>
            <div style="display: flex; gap: 2rem; font-size: 0.875rem;">
                <div>
                    <span style="color: var(--text-muted);">Workers: </span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{n_workers:,}</span>
                </div>
                <div>
                    <span style="color: var(--text-muted);">Transitions: </span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{n_transitions:,}</span>
                </div>
                <div>
                    <span style="color: var(--text-muted);">Industries: </span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{len(industry_df) if industry_df is not None else 0}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 1.5rem; padding: 1rem 1.5rem; border-color: var(--warning);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="color: var(--warning); font-size: 0.875rem; font-weight: 500;">
                SIMULATED DATA MODE
            </span>
            <span style="color: var(--text-muted); font-size: 0.875rem;">
                Live Data files not found - using synthetic Santa Barbara data
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SEVERITY CONTROL BAR
# ============================================================================

st.markdown('<p class="section-title">Scenario Configuration</p>', unsafe_allow_html=True)

col_sev1, col_sev2, col_sev3, col_sev4 = st.columns([2, 1, 1, 1])

with col_sev1:
    shock_severity = st.slider(
        "Storm Severity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format="%.0f%%",
        help="0% = Normal conditions, 100% = Catastrophic event",
        key="severity_main"
    )

with col_sev2:
    shock_duration = st.selectbox(
        "Duration",
        options=[7, 14, 21, 30, 45, 60],
        index=2,
        format_func=lambda x: f"{x} days"
    )

with col_sev3:
    shock_start = st.selectbox(
        "Shock Start",
        options=[15, 30, 45, 60],
        index=1,
        format_func=lambda x: f"Day {x}"
    )

with col_sev4:
    sim_days = st.selectbox(
        "Forecast",
        options=[180, 365, 540, 730],
        index=1,
        format_func=lambda x: f"{x} days"
    )

# Recovery parameters (hidden in expander)
with st.expander("Advanced Parameters", expanded=False):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        recovery_rate = st.slider("Recovery Rate (r)", 0.02, 0.25, 0.10, 0.01)
    with adv_col2:
        carrying_capacity = st.slider("Max Employment (K)", 0.70, 1.00, 0.95, 0.01)

# ============================================================================
# GET TRACT DATA & COMPUTE PROBABILITIES
# ============================================================================

sb_tracts = get_sb_tracts()
sb_tracts['exodus_prob'] = sb_tracts.apply(
    lambda r: calc_exodus_prob(r['ej_percentile'], r['coastal_jobs_pct'], shock_severity),
    axis=1
)
sb_tracts['vulnerability'] = pd.cut(
    sb_tracts['exodus_prob'],
    bins=[0, 0.15, 0.30, 0.50, 1.0],
    labels=['Low', 'Moderate', 'High', 'Critical']
)

# ============================================================================
# KEY METRICS
# ============================================================================

# Run simulation with average tract
solution, avg_beta = run_simulation(
    shock_severity, shock_duration, shock_start,
    recovery_rate, carrying_capacity, 50, sim_days
)

st.markdown("""
<div class="metric-container">
    <div class="metric-card">
        <div class="metric-label">Minimum Employment</div>
        <div class="metric-value">{:.1f}%</div>
        <div class="metric-delta negative">{:+.1f}% from baseline</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Recovery Time</div>
        <div class="metric-value">{}</div>
        <div class="metric-delta">to 95% baseline</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Equilibrium</div>
        <div class="metric-value">{:.1f}%</div>
        <div class="metric-delta {}">{:+.1f}% permanent</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Resilience Index</div>
        <div class="metric-value">{:.2f}</div>
        <div class="metric-delta">{}</div>
    </div>
</div>
""".format(
    solution.min_labor_force * 100,
    (solution.min_labor_force - 0.92) * 100,
    f"{solution.recovery_time:.0f}d" if solution.recovery_time else "N/A",
    solution.equilibrium * 100,
    "negative" if solution.equilibrium < 0.92 else "positive",
    (solution.equilibrium - 0.92) * 100,
    solution.resilience_score,
    "High" if solution.resilience_score > 0.7 else ("Moderate" if solution.resilience_score > 0.4 else "Low")
), unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT: MAP + SIDEBAR
# ============================================================================

col_map, col_sidebar = st.columns([2, 1])

with col_map:
    st.markdown('<p class="section-title">Live Hazard Map</p>', unsafe_allow_html=True)
    
    # Live indicator
    st.markdown("""
    <div class="live-indicator">
        <span class="live-dot"></span>
        Real-time vulnerability mapping
    </div>
    """, unsafe_allow_html=True)
    
    # Create dark mode map
    fig_map = px.scatter_mapbox(
        sb_tracts,
        lat='lat',
        lon='lon',
        color='exodus_prob',
        size='coastal_jobs_pct',
        hover_name='name',
        hover_data={
            'exodus_prob': ':.1%',
            'ej_percentile': True,
            'coastal_jobs_pct': ':.0f',
            'population': ':,',
            'vulnerability': True,
            'lat': False,
            'lon': False
        },
        color_continuous_scale=[
            [0, '#22c55e'],
            [0.25, '#84cc16'],
            [0.5, '#eab308'],
            [0.75, '#f97316'],
            [1, '#ef4444']
        ],
        range_color=[0, 0.6],
        size_max=35,
        zoom=10.5,
        center={'lat': 34.42, 'lon': -119.75},
        mapbox_style='carto-darkmatter',
        labels={
            'exodus_prob': 'Exodus Probability',
            'ej_percentile': 'EJ Burden',
            'coastal_jobs_pct': 'Coastal Jobs %',
            'population': 'Population'
        }
    )
    
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color='#a0a0b0'),
        coloraxis_colorbar=dict(
            title=dict(text="Exodus<br>Prob.", font=dict(size=11)),
            tickformat=".0%",
            bgcolor='rgba(0,0,0,0)',
            tickfont=dict(size=10)
        )
    )
    
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

with col_sidebar:
    st.markdown('<p class="section-title">Tract Analysis</p>', unsafe_allow_html=True)
    
    # Tract selector
    selected_tract = st.selectbox(
        "Select Neighborhood",
        options=sb_tracts['name'].tolist(),
        index=1,  # Default to Waterfront
        label_visibility="collapsed"
    )
    
    tract = sb_tracts[sb_tracts['name'] == selected_tract].iloc[0]
    
    # Build badges HTML
    badges_html = ""
    if tract['high_poverty']:
        badges_html += "<span class='ej-badge'>High Poverty</span>"
    if tract['flood_zone']:
        badges_html += "<span class='ej-badge warning'>Flood Zone</span>"
    if tract['limited_english'] > 10:
        badges_html += f"<span class='ej-badge info'>Limited English {tract['limited_english']}%</span>"
    if tract['ej_percentile'] > 65:
        badges_html += "<span class='ej-badge'>High EJ Burden</span>"
    
    # Sidebar panel
    st.markdown(f"""
    <div class="sidebar-panel">
        <div class="sidebar-header">{tract['name']}</div>
        <div class="sidebar-subheader">Census Tract {tract['tract_id'][-4:]}</div>
        
        <div class="vuln-score">
            <span class="vuln-number">{tract['exodus_prob'] * 10:.1f}</span>
            <span class="vuln-max">/10</span>
        </div>
        
        <div class="metric-label" style="margin-bottom: 1rem;">Vulnerability Score</div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
            <div>
                <div class="metric-label">Population</div>
                <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono';">{tract['population']:,}</div>
            </div>
            <div>
                <div class="metric-label">Coastal Jobs</div>
                <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono';">{tract['coastal_jobs_pct']}%</div>
            </div>
            <div>
                <div class="metric-label">EJ Burden</div>
                <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono';">{tract['ej_percentile']}th</div>
            </div>
            <div>
                <div class="metric-label">Median Income</div>
                <div style="font-size: 1.25rem; font-weight: 600; font-family: 'JetBrains Mono';">${tract['median_income']//1000}k</div>
            </div>
        </div>
        
        <div class="metric-label">Risk Factors</div>
        <div class="badge-container">
            {badges_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top vulnerable tracts
    st.markdown('<p class="section-title" style="margin-top: 1.5rem;">Most Vulnerable Areas</p>', unsafe_allow_html=True)
    
    top_tracts = sb_tracts.nlargest(5, 'exodus_prob')
    for _, row in top_tracts.iterrows():
        risk_class = 'critical' if row['vulnerability'] == 'Critical' else ('high' if row['vulnerability'] == 'High' else 'moderate')
        color = '#ef4444' if risk_class == 'critical' else '#f59e0b'
        st.markdown(f"""
        <div class="tract-item {risk_class}">
            <span class="tract-name">{row['name']}</span>
            <span class="tract-prob" style="color: {color}">{row['exodus_prob']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RECOVERY FORECAST TIMELINE
# ============================================================================

st.markdown('<p class="section-title" style="margin-top: 2rem;">Recovery Forecast</p>', unsafe_allow_html=True)

# Timeline header
recovery_str = f"{solution.recovery_time:.0f} days" if solution.recovery_time else "Extended"
st.markdown(f"""
<div class="timeline-header">
    <span class="timeline-title">Labor Force Trajectory</span>
    <span class="recovery-estimate">Estimated recovery: <strong>{recovery_str}</strong></span>
</div>
""", unsafe_allow_html=True)

# Get comparison data
comparison = run_comparison(
    shock_severity, shock_duration, shock_start,
    recovery_rate, carrying_capacity, sim_days
)

# Create recovery chart
fig_timeline = go.Figure()

# Add traces for each EJ profile
colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
for (name, data), color in zip(comparison.items(), colors):
    sol = data['solution']
    fig_timeline.add_trace(go.Scatter(
        x=sol.t,
        y=sol.L * 100,
        mode='lines',
        name=f'{name}',
        line=dict(color=color, width=2.5),
        hovertemplate='%{y:.1f}%<extra>' + name + '</extra>'
    ))

# Add business as usual line
fig_timeline.add_trace(go.Scatter(
    x=[0, sim_days],
    y=[92, 92],
    mode='lines',
    name='Baseline',
    line=dict(color='#525252', width=1.5, dash='dash'),
    hoverinfo='skip'
))

# Add shock region
fig_timeline.add_vrect(
    x0=shock_start,
    x1=shock_start + shock_duration,
    fillcolor="rgba(239, 68, 68, 0.1)",
    layer="below",
    line_width=0,
    annotation_text="Climate Event",
    annotation_position="top left",
    annotation_font_color="#ef4444",
    annotation_font_size=11
)

fig_timeline.update_layout(
    template=None,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,10,15,1)',
    height=350,
    margin=dict(l=60, r=40, t=20, b=60),
    font=dict(family="Inter, sans-serif", color='#a0a0b0'),
    xaxis=dict(
        title=dict(text='Days', font=dict(size=12)),
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)',
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        title=dict(text='Labor Force Participation (%)', font=dict(size=12)),
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)',
        range=[40, 100],
        tickfont=dict(size=11)
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
        bgcolor='rgba(0,0,0,0)'
    ),
    hovermode="x unified"
)

st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# LIVE WORKFORCE INSIGHTS (if available)
# ============================================================================

if has_live_data and industry_df is not None and len(industry_df) > 0:
    st.markdown('<p class="section-title" style="margin-top: 1.5rem;">Workforce Intelligence</p>', unsafe_allow_html=True)
    
    col_ind, col_trans, col_county = st.columns(3)
    
    with col_ind:
        # Top industries bar chart
        top_ind = industry_df.nlargest(8, 'count')
        
        fig_ind = go.Figure(go.Bar(
            y=top_ind['industry'],
            x=top_ind['count'],
            orientation='h',
            marker=dict(
                color=top_ind['is_climate_sensitive'].map({True: '#ef4444', False: '#3b82f6'}),
            ),
            hovertemplate='%{y}: %{x} workers<extra></extra>'
        ))
        
        fig_ind.update_layout(
            template=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,10,15,1)',
            height=280,
            margin=dict(l=10, r=20, t=30, b=20),
            font=dict(family="Inter", size=10, color='#a0a0b0'),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                title=dict(text='Workers', font=dict(size=10)),
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                autorange='reversed',
            ),
            title=dict(
                text='Top Industries',
                font=dict(size=12, color='#ffffff'),
                x=0.5,
            )
        )
        
        st.plotly_chart(fig_ind, use_container_width=True, config={'displayModeBar': False})
        
        # Climate sensitivity legend
        climate_sens = industry_df[industry_df['is_climate_sensitive'] == True]['count'].sum()
        climate_res = industry_df[industry_df['is_climate_sensitive'] == False]['count'].sum()
        total = climate_sens + climate_res
        
        st.markdown(f"""
        <div style="font-size: 0.75rem; margin-top: 0.5rem;">
            <span style="color: #ef4444;">Climate-Sensitive: {climate_sens:,} ({climate_sens/total*100:.1f}%)</span>
            &nbsp;|&nbsp;
            <span style="color: #3b82f6;">Resilient: {climate_res:,}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_trans:
        # Job transition patterns
        if transitions_df is not None and len(transitions_df) > 0:
            # Create industry transition summary
            trans_summary = transitions_df.groupby(['from_industry', 'to_industry']).size().reset_index(name='count')
            trans_summary = trans_summary.nlargest(10, 'count')
            
            st.markdown("""
            <div class="glass-card" style="height: 280px; overflow-y: auto;">
                <div class="metric-label">Top Career Transitions</div>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            for _, row in trans_summary.head(6).iterrows():
                from_ind = row['from_industry'] if row['from_industry'] else 'Unknown'
                to_ind = row['to_industry'] if row['to_industry'] else 'Unknown'
                count = row['count']
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; font-size: 0.8rem;">
                    <span style="color: #a0a0b0;">{from_ind[:20]} → {to_ind[:20]}</span>
                    <span style="font-family: 'JetBrains Mono'; color: #8b5cf6;">{count}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="height: 280px;">
                <div class="metric-label">Career Transitions</div>
                <p style="color: var(--text-muted); font-size: 0.875rem; margin-top: 1rem;">
                    Transition data processing...
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_county:
        # County distribution
        if 'current_county' in persons_df.columns:
            county_counts = persons_df['current_county'].value_counts().head(8)
            
            fig_county = go.Figure(go.Pie(
                labels=county_counts.index,
                values=county_counts.values,
                hole=0.6,
                marker=dict(colors=['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16']),
                textinfo='percent',
                textfont=dict(size=9, color='white'),
                hovertemplate='%{label}: %{value:,}<extra></extra>'
            ))
            
            fig_county.update_layout(
                template=None,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(10,10,15,1)',
                height=280,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(family="Inter", size=10, color='#a0a0b0'),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=1.3,
                    font=dict(size=9),
                    bgcolor='rgba(0,0,0,0)'
                ),
                title=dict(
                    text='By County',
                    font=dict(size=12, color='#ffffff'),
                    x=0.5,
                )
            )
            
            st.plotly_chart(fig_county, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# SANKEY DIAGRAM - Worker Flow
# ============================================================================

st.markdown('<p class="section-title" style="margin-top: 1rem;">Worker Flow Analysis</p>', unsafe_allow_html=True)

col_sankey, col_sankey_info = st.columns([2, 1])

with col_sankey:
    # Create Sankey diagram showing worker transitions
    # Source: Coastal Hospitality (100 workers example)
    coastal_workers = int(tract['population'] * tract['coastal_jobs_pct'] / 100)
    
    # Calculate flows based on Markov probabilities and shock
    markov = MarkovTransitionMatrix()
    markov.apply_shock(ShockParameters(severity=shock_severity, duration_days=shock_duration))
    
    # Get transition probabilities
    to_inland = markov.get_transition_prob(LaborState.COASTAL, LaborState.INLAND)
    to_unemployed = markov.get_transition_prob(LaborState.COASTAL, LaborState.UNEMPLOYED)
    to_transitioning = markov.get_transition_prob(LaborState.COASTAL, LaborState.TRANSITIONING)
    stayed = 1 - to_inland - to_unemployed - to_transitioning
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color='rgba(255,255,255,0.1)', width=1),
            label=[
                f'Coastal Jobs<br>({coastal_workers:,})',
                f'Stayed<br>({int(coastal_workers * stayed):,})',
                f'Inland Jobs<br>({int(coastal_workers * to_inland):,})',
                f'Unemployed<br>({int(coastal_workers * to_unemployed):,})',
                f'Transitioning<br>({int(coastal_workers * to_transitioning):,})'
            ],
            color=['#3b82f6', '#22c55e', '#8b5cf6', '#ef4444', '#f59e0b'],
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=[0, 0, 0, 0],
            target=[1, 2, 3, 4],
            value=[
                int(coastal_workers * stayed),
                int(coastal_workers * to_inland),
                int(coastal_workers * to_unemployed),
                int(coastal_workers * to_transitioning)
            ],
            color=['rgba(34, 197, 94, 0.4)', 'rgba(139, 92, 246, 0.4)', 
                   'rgba(239, 68, 68, 0.4)', 'rgba(245, 158, 11, 0.4)'],
            hovertemplate='%{value:,} workers<extra></extra>'
        )
    )])
    
    fig_sankey.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=11, color='#a0a0b0'),
        height=280,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig_sankey, use_container_width=True, config={'displayModeBar': False})

with col_sankey_info:
    policy_text = "Allocate emergency employment grants to this tract. High displacement risk detected." if tract['exodus_prob'] > 0.3 else "Standard monitoring protocols sufficient for this area."
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Worker Displacement in {tract['name']}</div>
        <div style="margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="color: #22c55e;">Retained</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{stayed:.0%}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="color: #8b5cf6;">To Inland</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{to_inland:.0%}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="color: #ef4444;">Unemployed</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{to_unemployed:.0%}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #f59e0b;">Job Seeking</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">{to_transitioning:.0%}</span>
            </div>
        </div>
    </div>
    
    <div class="glass-card" style="margin-top: 1rem;">
        <div class="metric-label">Policy Recommendation</div>
        <p style="font-size: 0.875rem; color: #a0a0b0; margin-top: 0.75rem; line-height: 1.6;">
            {policy_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXPORT & FOOTER
# ============================================================================

st.markdown("---", unsafe_allow_html=True)

col_export, col_footer = st.columns([1, 2])

with col_export:
    st.markdown("""
    <a href="#" class="export-btn" onclick="window.print(); return false;">
        Generate PDF Report for City Council
    </a>
    """, unsafe_allow_html=True)

with col_footer:
    st.markdown("""
    <div class="footer">
        <span class="footer-item">Powered by Live Data Technologies</span>
        <span class="footer-item">Intelligence by RapidFire AI</span>
        <span class="footer-item">Data4Good Hackathon 2026</span>
    </div>
    """, unsafe_allow_html=True)

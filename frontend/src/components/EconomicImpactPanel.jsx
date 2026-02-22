import React from 'react';
import {
  Activity,
  Waves,
  DollarSign,
  Users,
  MapPin,
  Sparkles,
} from 'lucide-react';

const RISK_STYLES = {
  Low: { color: '#34d399', bg: 'rgba(52,211,153,0.12)' },
  Moderate: { color: '#fbbf24', bg: 'rgba(251,191,36,0.12)' },
  High: { color: '#fb923c', bg: 'rgba(251,146,60,0.12)' },
  Critical: { color: '#f43f5e', bg: 'rgba(244,63,94,0.12)' },
};

export default function EconomicImpactPanel({ impact, loading }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 520 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const noData = !impact || impact.error || !impact.tracts?.length;

  if (noData) {
    return (
      <div className="glass-card impact-panel" style={{ padding: 24 }}>
        <div className="impact-header">
          <div className="impact-icon">
            <Activity size={18} />
          </div>
          <div>
            <div className="impact-title">Economic Impact Score</div>
            <div className="impact-subtitle">XGBoost Regression · tract-level risk</div>
          </div>
        </div>
        <div className="impact-empty">
          <Activity size={28} />
          <div>Model not available</div>
          <div className="impact-empty-hint">Run: python train_economic_model.py</div>
        </div>
      </div>
    );
  }

  const stats = impact.stats || {};
  const total = stats.total_tracts || impact.tracts.length || 1;
  const distribution = impact.distribution || {};
  const meanScore = stats.mean_vulnerability ?? 0;
  const maxScore = stats.max_vulnerability ?? 0;
  const minScore = stats.min_vulnerability ?? 0;
  const meanPct = Math.min(100, Math.max(0, meanScore));

  const segments = [
    { key: 'Low', count: distribution.Low || 0, color: RISK_STYLES.Low.color },
    { key: 'Moderate', count: distribution.Moderate || 0, color: RISK_STYLES.Moderate.color },
    { key: 'High', count: distribution.High || 0, color: RISK_STYLES.High.color },
    { key: 'Critical', count: distribution.Critical || 0, color: RISK_STYLES.Critical.color },
  ];

  const topTracts = impact.tracts.slice(0, 6);
  const leadTract = topTracts[0];

  return (
    <div className="glass-card impact-panel" style={{ padding: 24 }}>
      <div className="impact-header">
        <div className="impact-icon">
          <Sparkles size={18} />
        </div>
        <div style={{ flex: 1 }}>
          <div className="impact-title">Economic Impact Score</div>
          <div className="impact-subtitle">
            Composite vulnerability across sea level, workforce, and equity signals
          </div>
        </div>
        <div className="impact-badge">
          {total} tracts analyzed
        </div>
      </div>

      <div className="impact-explainer">
        This panel ranks census tracts by a <strong>0-100 vulnerability score</strong>. Higher scores mean greater
        expected economic impact from coastal climate stress. The score blends sea-level exposure, housing pressure,
        workforce risk, and social vulnerability into one index.
      </div>

      <div className="impact-spotlight">
        <div className="impact-spotlight-card">
          <div className="impact-spotlight-label">Highest exposure tract</div>
          <div className="impact-spotlight-title">
            {leadTract?.tract_name || 'No tract data'}
          </div>
          <div className="impact-spotlight-meta">
            <span className="impact-chip impact-chip-strong">{leadTract?.risk_level || 'Low'}</span>
            <span className="impact-chip">{leadTract?.vulnerability_score ?? 0} score</span>
            <span className="impact-chip">${Math.round((leadTract?.median_income || 0) / 1000)}k income</span>
          </div>
        </div>
        <div className="impact-spotlight-card impact-spotlight-stats">
          <div className="impact-spotlight-label">Score range</div>
          <div className="impact-spotlight-range">
            <span>{minScore}</span>
            <span>{maxScore}</span>
          </div>
          <div className="impact-meter">
            <div className="impact-meter-bar" />
            <div className="impact-meter-marker" style={{ left: `${meanPct}%` }} />
            <div className="impact-meter-label">Mean {meanScore}</div>
          </div>
        </div>
      </div>

      <div className="impact-hero">
        <div className="impact-hero-main">
          <div className="impact-hero-label">Mean Vulnerability</div>
          <div className="impact-hero-score">{meanScore}</div>
          <div className="impact-hero-scale">0 - 100 risk scale</div>
        </div>
        <div className="impact-hero-grid">
          <div className="impact-kpi">
            <Waves size={16} />
            <div>
              <div className="impact-kpi-label">High Risk Tracts</div>
              <div className="impact-kpi-value">{stats.high_risk_count ?? 0}</div>
            </div>
          </div>
          <div className="impact-kpi">
            <DollarSign size={16} />
            <div>
              <div className="impact-kpi-label">Max Score</div>
              <div className="impact-kpi-value">{stats.max_vulnerability ?? 0}</div>
            </div>
          </div>
          <div className="impact-kpi">
            <Users size={16} />
            <div>
              <div className="impact-kpi-label">Total Tracts</div>
              <div className="impact-kpi-value">{total}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="impact-distribution">
        <div className="impact-distribution-title">Risk Distribution (count of tracts by score band)</div>
        <div className="impact-bar">
          {segments.map((segment) => {
            const pct = (segment.count / total) * 100;
            return (
              <div
                key={segment.key}
                className="impact-bar-seg"
                style={{ width: `${pct}%`, background: segment.color }}
                title={`${segment.key}: ${segment.count} (${pct.toFixed(1)}%)`}
              >
                {pct > 12 ? segment.count : ''}
              </div>
            );
          })}
        </div>
        <div className="impact-distribution-legend">
          {segments.map((segment) => (
            <div key={segment.key} className="impact-legend-item">
              <span className="impact-legend-dot" style={{ background: segment.color }} />
              <span>{segment.key}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="impact-grid">
        <div className="impact-list">
          <div className="impact-section-title">Highest-Risk Tracts (top scores)</div>
          {topTracts.map((tract, index) => {
            const style = RISK_STYLES[tract.risk_level] || RISK_STYLES.Moderate;
            return (
              <div key={tract.tract_id} className="impact-item" style={{ borderColor: style.color }}>
                <div className="impact-rank" style={{ background: style.bg, color: style.color }}>
                  {index + 1}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="impact-item-title">{tract.tract_name}</div>
                  <div className="impact-item-meta">
                    <span className="impact-chip" style={{ background: style.bg, color: style.color }}>
                      {tract.risk_level}
                    </span>
                    <span className="impact-chip">
                      Score {tract.vulnerability_score}
                    </span>
                    <span className="impact-chip">
                      Income ${Math.round(tract.median_income / 1000)}k
                    </span>
                  </div>
                </div>
                <div className="impact-item-geo">
                  <MapPin size={12} />
                  {tract.lat.toFixed(2)}, {tract.lon.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>

        <div className="impact-signals">
          <div className="impact-section-title">Key Signals (what drives the score)</div>
          <div className="impact-signal-card">
            <Waves size={16} />
            <div>
              <div className="impact-signal-title">Sea Level Exposure</div>
              <div className="impact-signal-text">Flood zones + annual high water events</div>
            </div>
          </div>
          <div className="impact-signal-card">
            <DollarSign size={16} />
            <div>
              <div className="impact-signal-title">Economic Vulnerability</div>
              <div className="impact-signal-text">Income, poverty, housing pressure</div>
            </div>
          </div>
          <div className="impact-signal-card">
            <Users size={16} />
            <div>
              <div className="impact-signal-title">Workforce Risk</div>
              <div className="impact-signal-text">Coastal jobs + climate-sensitive exposure</div>
            </div>
          </div>
          <div className="impact-signal-card">
            <Activity size={16} />
            <div>
              <div className="impact-signal-title">Social Vulnerability</div>
              <div className="impact-signal-text">EJ percentile + language access</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

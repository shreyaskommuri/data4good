import React from 'react';
import {
  Activity,
  Waves,
  DollarSign,
  Users,
  MapPin,
  Sparkles,
} from 'lucide-react';
import Tooltip from './Tooltip';

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
            <div className="impact-subtitle">Scenario Ensemble · ODE + Markov tract risk</div>
          </div>
        </div>
        <div className="impact-empty">
          <Activity size={28} />
          <div>Risk model unavailable</div>
          <div className="impact-empty-hint">Check backend data sources and try refresh</div>
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

  const topTracts = impact.tracts.slice(0, 4);

  let averageMeaning = 'Most tracts are likely to recover relatively quickly.';
  if (meanScore >= 75) {
    averageMeaning = 'Most tracts are likely to face severe, long recovery delays.';
  } else if (meanScore >= 50) {
    averageMeaning = 'Many tracts are likely to face slower-than-normal recovery.';
  } else if (meanScore >= 25) {
    averageMeaning = 'Recovery is mixed: some tracts may face meaningful delays.';
  }

  return (
    <div className="glass-card impact-panel" style={{ padding: 24 }}>
      {/* Header */}
      <div className="impact-header">
        <div className="impact-icon">
          <Activity size={18} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div className="impact-title">Recovery Risk</div>
            <Tooltip content="We simulate multiple shock scenarios and combine recovery dynamics with workforce transition pressure to produce a <strong>0–100 recovery risk score</strong> for each tract. Higher scores indicate communities that are likely to recover more slowly after a climate shock.<br/><br/><strong>Calculation:</strong> Recovery Risk = Sea Level Risk (0-25) + Economic Vulnerability (0-35) + Workforce Risk (0-25) + Social Vulnerability (0-15)<br/><br/>• <strong>Sea Level Risk (0-25):</strong> Flood zone (10pts) + coastal proximity (0-15pts)<br/>• <strong>Economic Vulnerability (0-35):</strong> Low income (0-15pts) + poverty rate (0-10pts) + housing pressure (0-10pts)<br/>• <strong>Workforce Risk (0-25):</strong> Coastal jobs % (0-10pts) + climate-sensitive jobs % (0-10pts) + job instability (0-5pts)<br/>• <strong>Social Vulnerability (0-15):</strong> EJ percentile (0-7pts) + limited English % (0-4pts) + minority % (0-4pts)" />
          </div>
          <div className="impact-subtitle">
            Which communities are most likely to recover slowly after a climate shock?
          </div>
        </div>
      </div>

      {/* Main Score with Context */}
      <div className="impact-hero-simple">
        <div className="impact-score-context">
          <div>
              <div className="impact-hero-label">Average Recovery Risk</div>
            <div className="impact-hero-score-large">{meanScore}</div>
            <div style={{ marginTop: 4, color: 'var(--text-muted)', fontSize: 13 }}>
              {averageMeaning}
            </div>
          </div>
          <div className="impact-score-details">
            <div className="impact-score-stat">
              <span className="stat-label">Lowest</span>
              <span className="stat-value">{minScore}</span>
            </div>
            <div className="impact-score-stat">
              <span className="stat-label">Highest</span>
              <span className="stat-value">{maxScore}</span>
            </div>
            <div className="impact-score-stat">
              <span className="stat-label">Areas</span>
              <span className="stat-value">{total}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Visual Distribution */}
      <div className="impact-distribution">
        <div className="impact-distribution-title">How many neighborhoods are in each risk tier</div>
        <div className="impact-bar">
          {segments.map((segment) => {
            const pct = (segment.count / total) * 100;
            return (
              <div
                key={segment.key}
                className="impact-bar-seg"
                style={{ width: `${pct}%`, background: segment.color }}
                title={`${segment.key}: ${segment.count} areas`}
              >
                {pct > 10 ? segment.count : ''}
              </div>
            );
          })}
        </div>
        <div className="impact-distribution-legend">
          {segments.map((segment) => {
            const labelMap = {
              Low: 'Low (faster recovery likely)',
              Moderate: 'Moderate (some delay risk)',
              High: 'High (slow recovery likely)',
              Critical: 'Critical (major delay risk)',
            };
            return (
            <div key={segment.key} className="impact-legend-item">
              <span className="impact-legend-dot" style={{ background: segment.color }} />
              <span>{labelMap[segment.key] || segment.key}</span>
              <span className="legend-count">({segment.count})</span>
            </div>
            );
          })}
        </div>
      </div>

      {/* Top Areas - Clear and Scannable */}
      <div className="impact-list-simple">
        <div className="impact-section-title">Highest-priority tracts</div>
        {topTracts.map((tract, index) => {
          const style = RISK_STYLES[tract.risk_level] || RISK_STYLES.Moderate;
          return (
            <div key={tract.tract_id} className="impact-item-simple">
              <div className="impact-rank-simple" style={{ background: style.color }}>
                {index + 1}
              </div>
              <div style={{ flex: 1 }}>
                <div className="impact-item-name">{tract.tract_name}</div>
                <div className="impact-item-detail">
                  <span style={{ color: style.color, fontWeight: 600 }}>{tract.risk_level} Risk</span>
                  <span className="detail-sep">•</span>
                  Risk score {tract.vulnerability_score}/100
                  <span className="detail-sep">•</span>
                  ${Math.round(tract.median_income / 1000)}k income
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

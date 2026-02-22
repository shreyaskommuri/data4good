import React from 'react';
import {
  Shield, TrendingDown, Clock, AlertTriangle, DollarSign, Activity
} from 'lucide-react';

function scoreColor(score) {
  if (score >= 0.7) return '#34d399';
  if (score >= 0.4) return '#fbbf24';
  return '#f43f5e';
}

function statusLabel(score) {
  if (score >= 0.7) return 'RESILIENT';
  if (score >= 0.4) return 'AT RISK';
  return 'VULNERABLE';
}

export default function KPIHeader({ sim, loading }) {
  if (loading || !sim) {
    return (
      <div style={{ padding: '32px 0' }}>
        <div className="skeleton" style={{ height: 140, width: '100%' }} />
      </div>
    );
  }

  const score = sim.resilience_score ?? 0;
  const color = scoreColor(score);

  const kpis = [
    {
      icon: <TrendingDown size={18} />,
      label: 'Labor Flight',
      value: `${sim.labor_flight_pct}%`,
      desc: 'Peak workforce displacement from baseline',
      color: sim.labor_flight_pct > 15 ? '#f43f5e' : '#fbbf24',
      badge: sim.labor_flight_pct > 15 ? 'critical' : 'warning',
    },
    {
      icon: <Clock size={18} />,
      label: 'Recovery Time',
      value: sim.recovery_time ? `${Math.round(sim.recovery_time)}d` : 'N/A',
      desc: 'Days to return to 95% pre-shock employment',
      color: '#3b82f6',
      badge: 'info',
    },
    {
      icon: <AlertTriangle size={18} />,
      label: 'Critical Tracts',
      value: sim.critical_tracts,
      desc: 'Census tracts with >50% exodus probability',
      color: '#f43f5e',
      badge: 'critical',
    },
    {
      icon: <DollarSign size={18} />,
      label: 'Relief Budget',
      value: `$${(sim.emergency_fund / 1e6).toFixed(1)}M`,
      desc: 'Estimated emergency workforce stabilization fund',
      color: '#a78bfa',
      badge: 'info',
    },
  ];

  return (
    <div className="fade-in" style={{ marginBottom: 32 }}>
      {/* North Star score */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 32,
        marginBottom: 28,
        flexWrap: 'wrap',
      }}>
        <div style={{ position: 'relative' }}>
          <div style={{
            fontSize: '4.5rem',
            fontWeight: 900,
            letterSpacing: '-0.04em',
            lineHeight: 1,
            background: `linear-gradient(135deg, ${color}, ${color}88)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            {(score * 100).toFixed(0)}
          </div>
          <div style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            color: color,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginTop: 4,
          }}>
            Resilience Score
          </div>
        </div>
        <div>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 16px',
            borderRadius: 20,
            background: `${color}15`,
            border: `1px solid ${color}40`,
            color: color,
            fontSize: '0.8rem',
            fontWeight: 600,
            fontFamily: 'var(--font-mono)',
            marginBottom: 8,
          }}>
            <Activity size={14} className="pulse" />
            {statusLabel(score)}
          </div>
          <p style={{
            color: 'var(--text-secondary)',
            fontSize: '0.9rem',
            maxWidth: 420,
            lineHeight: 1.6,
          }}>
            Santa Barbara County's ability to maintain coastal employment
            through climate disruption. Based on ODE labor-flow modeling
            across <strong style={{ color: 'var(--text-primary)' }}>109 census tracts</strong>.
          </p>
        </div>
      </div>

      {/* KPI strip */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 16,
      }}>
        {kpis.map((kpi, i) => (
          <div key={i} className={`glass-card fade-in-d${i + 1}`} style={{
            padding: '20px',
            position: 'relative',
            overflow: 'hidden',
          }}>
            {/* accent bar */}
            <div style={{
              position: 'absolute', top: 0, left: 0, right: 0, height: 3,
              background: kpi.color,
              borderRadius: '16px 16px 0 0',
            }} />
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              marginBottom: 8,
            }}>
              <span style={{ color: kpi.color }}>{kpi.icon}</span>
              <span style={{
                fontSize: '0.75rem', fontWeight: 600,
                color: 'var(--text-secondary)', textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}>
                {kpi.label}
              </span>
            </div>
            <div style={{
              fontSize: '1.8rem', fontWeight: 800,
              fontFamily: 'var(--font-mono)', color: kpi.color,
              lineHeight: 1.1, marginBottom: 6,
            }}>
              {kpi.value}
            </div>
            <div style={{
              fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4,
            }}>
              {kpi.desc}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

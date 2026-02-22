import React, { memo } from 'react';
import {
  Shield, TrendingDown, Clock, AlertTriangle, DollarSign, Activity
} from 'lucide-react';
import { useAnimatedNumber } from '../hooks';
import Tooltip from './Tooltip';

function scoreColor(score) {
  if (score >= 0.6) return '#34d399';
  if (score >= 0.35) return '#fbbf24';
  return '#f43f5e';
}

function statusLabel(score) {
  if (score >= 0.6) return 'RESILIENT';
  if (score >= 0.35) return 'AT RISK';
  return 'VULNERABLE';
}

function AnimatedNumber({ value, format, style }) {
  const animatedValue = useAnimatedNumber(value, 400);
  return (
    <span style={style}>
      {format ? format(animatedValue) : animatedValue}
    </span>
  );
}

const KPIHeader = memo(function KPIHeader({ sim, loading }) {
  // Only show loading on initial load, not during updates
  if (loading && !sim) {
    return (
      <div style={{ padding: '32px 0' }}>
        <div className="skeleton" style={{ height: 140, width: '100%' }} />
      </div>
    );
  }
  
  // If we don't have data yet, show loading
  if (!sim) {
    return (
      <div style={{ padding: '32px 0' }}>
        <div className="skeleton" style={{ height: 140, width: '100%' }} />
      </div>
    );
  }

  const score = sim.resilience_score ?? 0;
  const color = scoreColor(score);

  const laborFlightPct = sim.labor_flight_pct ?? 0;
  const recoveryTime = sim.recovery_time ?? 0;
  const criticalTracts = sim.critical_tracts ?? 0;
  const emergencyFund = sim.emergency_fund ?? 0;

  const kpis = [
    {
      icon: <TrendingDown size={18} />,
      label: 'Labor Flight',
      value: laborFlightPct,
      format: (v) => `${v.toFixed(1)}%`,
      desc: 'Peak workforce displacement from baseline',
      color: laborFlightPct > 15 ? '#f43f5e' : '#fbbf24',
      badge: laborFlightPct > 15 ? 'critical' : 'warning',
    },
    {
      icon: <Clock size={18} />,
      label: 'Recovery Time',
      value: recoveryTime,
      format: (v) => v > 0 ? `${Math.round(v)}d` : 'N/A',
      desc: 'Days to return to 95% pre-shock employment',
      color: '#3b82f6',
      badge: 'info',
    },
    {
      icon: <AlertTriangle size={18} />,
      label: 'Critical Tracts',
      value: criticalTracts,
      format: (v) => Math.round(v).toString(),
      desc: 'Census tracts with >50% exodus probability',
      color: '#f43f5e',
      badge: 'critical',
    },
    {
      icon: <DollarSign size={18} />,
      label: 'Relief Budget',
      value: emergencyFund / 1e6,
      format: (v) => `$${v.toFixed(1)}M`,
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
          <div 
            style={{
              fontSize: '4.5rem',
              fontWeight: 900,
              letterSpacing: '-0.04em',
              lineHeight: 1,
              color: color,
              transition: 'color 0.4s ease-out',
            }}
          >
            <AnimatedNumber
              value={score * 100}
              format={(v) => v.toFixed(0)}
            />
          </div>
          <div style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            color: color,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginTop: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}>
            Resilience Score
            <Tooltip content="Santa Barbara County's ability to maintain coastal employment through climate disruption. Based on ODE labor-flow modeling across <strong>109 census tracts</strong>." />
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
              transition: 'background 0.4s ease-out',
            }} />
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              marginBottom: 8,
            }}>
              <span style={{ color: kpi.color, transition: 'color 0.4s ease-out' }}>{kpi.icon}</span>
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
              transition: 'color 0.4s ease-out',
            }}>
              <AnimatedNumber
                value={kpi.value}
                format={kpi.format}
              />
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
}, (prevProps, nextProps) => {
  // Only re-render if loading state changes or actual data values change
  if (prevProps.loading !== nextProps.loading) return false;
  if (prevProps.loading || nextProps.loading) return true;
  if (!prevProps.sim || !nextProps.sim) return prevProps.sim !== nextProps.sim;
  
  // Compare actual values, not object references
  const prev = prevProps.sim;
  const next = nextProps.sim;
  return (
    prev.resilience_score === next.resilience_score &&
    prev.labor_flight_pct === next.labor_flight_pct &&
    prev.recovery_time === next.recovery_time &&
    prev.critical_tracts === next.critical_tracts &&
    prev.emergency_fund === next.emergency_fund
  );
});

export default KPIHeader;

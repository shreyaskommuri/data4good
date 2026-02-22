import React from 'react';
import { ShieldAlert } from 'lucide-react';

const SIZE = 170;
const STROKE = 14;
const RADIUS = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function WorkforceAtRiskGauge({ sensitivityPct = 0, sensitive = 0, resilient = 0, avgJobs = 0 }) {
  const pct = Math.min(Math.max(sensitivityPct, 0), 100);
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
  const total = sensitive + resilient;

  const color = pct > 30 ? '#f43f5e' : pct > 15 ? '#fbbf24' : '#22d3ee';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <div style={{ position: 'relative', width: SIZE, height: SIZE }}>
        <svg width={SIZE} height={SIZE} style={{ transform: 'rotate(-90deg)' }}>
          <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="50%" stopColor="#fbbf24" />
              <stop offset="100%" stopColor="#f43f5e" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={STROKE}
          />
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
            filter="url(#glow)"
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1)' }}
          />
        </svg>
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            fontSize: '1.8rem', fontWeight: 800,
            fontFamily: "'JetBrains Mono', monospace",
            color, lineHeight: 1,
          }}>
            {sensitive.toLocaleString()}
          </div>
          <div style={{
            fontSize: '0.6rem', fontWeight: 500, color: 'var(--text-muted)',
            marginTop: 4, textAlign: 'center', lineHeight: 1.3,
          }}>
            workers in<br />vulnerable industries
          </div>
        </div>
      </div>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '4px 12px', borderRadius: 20,
        background: `${color}12`,
        border: `1px solid ${color}30`,
        fontSize: '0.7rem', fontWeight: 600, color,
      }}>
        <ShieldAlert size={11} />
        {pct.toFixed(1)}% of {total.toLocaleString()} total
      </div>

      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6,
        width: '100%',
      }}>
        <MiniStat
          value={resilient.toLocaleString()}
          label="Resilient"
          color="#22d3ee"
        />
        <MiniStat
          value={avgJobs}
          label="Avg Jobs / Worker"
          color="#a78bfa"
        />
      </div>
    </div>
  );
}

function MiniStat({ value, label, color }) {
  return (
    <div style={{
      textAlign: 'center',
      padding: '6px 4px',
      borderRadius: 8,
      background: `${color}08`,
      border: `1px solid ${color}15`,
    }}>
      <div style={{
        fontSize: '0.9rem', fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        color,
      }}>
        {value}
      </div>
      <div style={{
        fontSize: '0.55rem', color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.03em',
        marginTop: 1,
      }}>
        {label}
      </div>
    </div>
  );
}

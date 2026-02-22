import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Legend,
} from 'recharts';
import { TrendingUp } from 'lucide-react';

const PROFILE_COLORS = {
  'Low Burden': '#34d399',
  'Average': '#3b82f6',
  'High Burden': '#fbbf24',
  'Extreme': '#f43f5e',
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(18,19,26,0.95)',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 12,
      padding: '12px 16px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    }}>
      <div style={{
        fontSize: '0.8rem', fontWeight: 600, color: '#e8e8f0',
        marginBottom: 6,
      }}>
        Day {Math.round(label)}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: '0.75rem', marginBottom: 3,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%', background: p.color,
          }} />
          <span style={{ color: '#8b8ca0' }}>{p.name}:</span>
          <span style={{
            color: p.color, fontFamily: 'var(--font-mono)', fontWeight: 500,
          }}>
            {p.value}%
          </span>
        </div>
      ))}
    </div>
  );
}

export default function RecoveryChart({ comparison, loading }) {
  if (loading || !comparison) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 460 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const { profiles, shock, baseline } = comparison;

  // Merge all profiles into a single data array
  const profileNames = Object.keys(profiles);
  const refProfile = profiles[profileNames[0]];
  const chartData = refProfile.t.map((t, i) => {
    const point = { day: t };
    profileNames.forEach(name => {
      point[name] = profiles[name].L[i] ?? null;
    });
    return point;
  });

  // Find where all lines have stabilized (within 0.3% of their final value for 3+ consecutive points)
  let cutoffDay = chartData[chartData.length - 1]?.day ?? 365;
  for (let i = 5; i < chartData.length - 3; i++) {
    const allStable = profileNames.every(name => {
      const final = profiles[name].L[profiles[name].L.length - 1];
      return [0, 1, 2].every(d =>
        Math.abs((chartData[i + d][name] || 0) - final) < 0.3
      );
    });
    if (allStable) {
      // Add 20% breathing room past the stabilization point
      cutoffDay = Math.ceil(chartData[i].day * 1.2);
      break;
    }
  }
  // Round to nearest 30 days, minimum shock end + 60
  cutoffDay = Math.max(shock.end + 60, Math.ceil(cutoffDay / 30) * 30);

  const filteredData = chartData.filter(d => d.day <= cutoffDay);

  // Generate clean ticks based on the dynamic range
  const tickStep = cutoffDay <= 120 ? 15 : cutoffDay <= 210 ? 30 : 60;
  const ticks = [];
  for (let t = 0; t <= cutoffDay; t += tickStep) ticks.push(t);
  if (ticks[ticks.length - 1] !== cutoffDay) ticks.push(cutoffDay);

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(59,130,246,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#3b82f6',
        }}>
          <TrendingUp size={18} />
        </span>
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            How Fast Can We Recover?
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Employment recovery by Environmental Justice burden level
          </div>
        </div>
      </div>

      <div className="explanation">
        This chart shows <strong>employment levels over time</strong> after a climate shock.
        Each line represents communities with different Environmental Justice (EJ) burden levels.
        <strong> Low-burden communities recover fastest</strong>, while high-burden (high-poverty,
        high-minority, limited-English) communities take much longer.
        The red zone marks the shock period.
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={filteredData} margin={{ top: 10, right: 30, bottom: 10, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="day"
            type="number"
            domain={[0, cutoffDay]}
            stroke="rgba(255,255,255,0.1)"
            tick={{ fill: '#5a5b70', fontSize: 11 }}
            tickFormatter={v => {
              if (v === 0) return '0';
              if (v < 30) return `${Math.round(v)}d`;
              return `${Math.round(v / 30)}mo`;
            }}
            ticks={ticks}
            interval={0}
            allowDataOverflow={false}
          />
          <YAxis
            domain={[
              (dataMin) => Math.floor(dataMin / 5) * 5 - 2,
              (dataMax) => Math.min(100, Math.ceil(dataMax / 5) * 5 + 2),
            ]}
            stroke="rgba(255,255,255,0.1)"
            tick={{ fill: '#5a5b70', fontSize: 11 }}
            tickFormatter={v => `${v}%`}
            width={48}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Shock zone */}
          <ReferenceArea
            x1={shock.start} x2={shock.end}
            fill="rgba(244,63,94,0.10)"
            stroke="rgba(244,63,94,0.25)"
            strokeDasharray="4 3"
            label={{
              value: 'SHOCK',
              position: 'insideTop',
              fill: 'rgba(244,63,94,0.4)',
              fontSize: 10,
              fontWeight: 600,
            }}
          />

          {/* Baseline */}
          <ReferenceLine
            y={baseline}
            stroke="rgba(52,211,153,0.3)"
            strokeDasharray="8 4"
            strokeWidth={1.5}
            label={{
              value: `${baseline}% pre-shock`,
              position: 'right',
              fill: '#34d399',
              fontSize: 10,
              fontWeight: 500,
            }}
          />

          {/* Profile lines */}
          {profileNames.map(name => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={PROFILE_COLORS[name] || '#8b8ca0'}
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, stroke: PROFILE_COLORS[name], strokeWidth: 2, fill: '#181924' }}
            />
          ))}

          <Legend
            verticalAlign="bottom"
            height={36}
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: '0.75rem', color: '#8b8ca0' }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Recovery time comparison */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 12,
        marginTop: 16,
      }}>
        {profileNames.map(name => {
          const p = profiles[name];
          const color = PROFILE_COLORS[name];
          return (
            <div key={name} style={{
              padding: '12px',
              borderRadius: 10,
              background: `${color}0a`,
              border: `1px solid ${color}20`,
              textAlign: 'center',
            }}>
              <div style={{
                fontSize: '0.7rem', color: '#8b8ca0',
                textTransform: 'uppercase', letterSpacing: '0.05em',
                marginBottom: 4,
              }}>
                {name}
              </div>
              <div style={{
                fontSize: '1.2rem', fontWeight: 700,
                fontFamily: 'var(--font-mono)', color,
              }}>
                {p.recovery_time ? `${Math.round(p.recovery_time)}d` : '—'}
              </div>
              <div style={{ fontSize: '0.65rem', color: '#5a5b70', marginTop: 2 }}>
                recovery
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

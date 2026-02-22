import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { Home } from 'lucide-react';
import Tooltip from './Tooltip';

const PRESSURE_COLORS = {
  Critical: '#f43f5e',
  High: '#fb923c',
  Moderate: '#fbbf24',
  Low: '#34d399',
};

const PRESSURE_LABELS = ['Critical', 'High', 'Moderate', 'Low'];

function titleCase(str) {
  if (!str) return '';
  return str
    .toLowerCase()
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function abbreviate(name) {
  if (!name) return '';
  const titled = titleCase(name);
  if (titled.length <= 12) return titled;
  return titled.replace('Santa Barbara County', 'SB County')
    .replace('Santa Barbara', 'Santa Barb.')
    .replace('Santa Maria', 'Sta. Maria');
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: 'rgba(18,19,26,0.95)',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 12,
      padding: '12px 16px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      maxWidth: 260,
    }}>
      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e8e8f0', marginBottom: 8 }}>
        {titleCase(d.jurisdiction)}
      </div>
      <div style={{ fontSize: '0.78rem', color: '#8b8ca0', lineHeight: 1.9 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span>Pressure Index</span>
          <span style={{ color: PRESSURE_COLORS[d.pressure_level] || '#8b8ca0', fontWeight: 600 }}>
            {d.housing_pressure_index?.toFixed(1)} / 100
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span>Level</span>
          <span style={{ color: PRESSURE_COLORS[d.pressure_level] || '#8b8ca0', fontWeight: 600 }}>
            {d.pressure_level}
          </span>
        </div>
        {d.rhna_target != null && (
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>RHNA Target</span>
            <span style={{ color: '#e8e8f0', fontWeight: 500 }}>{d.rhna_target?.toLocaleString()}</span>
          </div>
        )}
        {d.total_permitted != null && (
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>Total Permitted</span>
            <span style={{ color: '#e8e8f0', fontWeight: 500 }}>{d.total_permitted?.toLocaleString()}</span>
          </div>
        )}
        {d.progress_pct != null && (
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>RHNA Progress</span>
            <span style={{ color: '#e8e8f0', fontWeight: 500 }}>{d.progress_pct?.toFixed(1)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function HousingPanel({ housing, loading }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 400 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const pressure = housing?.pressure || [];
  const stats = housing?.stats || {};
  const noData = pressure.length === 0;

  const sortedPressure = [...pressure].sort(
    (a, b) => (b.housing_pressure_index || 0) - (a.housing_pressure_index || 0)
  );

  const worstJurisdiction = sortedPressure[0];
  const bestJurisdiction = sortedPressure[sortedPressure.length - 1];
  const criticalCount = stats.critical_jurisdictions ?? 0;
  const avgPressure = stats.avg_pressure ?? 0;
  const avgRhna = stats.avg_rhna_progress ?? 0;

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(251,191,36,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fbbf24',
        }}>
          <Home size={18} />
        </span>
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            Housing Pressure Index
            <Tooltip content="This index scores each jurisdiction from <strong>0 (no pressure) to 100 (extreme pressure)</strong> based on how much housing supply constrains climate resilience. It combines four factors: how far behind a jurisdiction is on state-mandated housing targets (40%), the shortage of affordable units (30%), over-reliance on ADUs instead of multi-family housing (15%), and rental market tightness (15%). A higher score means <strong>fewer housing options for displaced workers</strong> after a climate event." />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            SBCAG 2024 APR data · {stats.total_jurisdictions || 0} jurisdictions
          </div>
        </div>
      </div>

      {noData ? (
        <div style={{
          textAlign: 'center', padding: 48, color: 'var(--text-muted)',
        }}>
          <Home size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <div>Housing data not available</div>
          <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
            Requires APR_Download_2024.xlsx in project root
          </div>
        </div>
      ) : (
        <>
          {/* Stats strip */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12,
            marginBottom: 20,
          }}>
            <div style={{
              padding: 12, borderRadius: 10,
              background: 'rgba(251,191,36,0.08)',
              border: '1px solid rgba(251,191,36,0.15)',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#fbbf24' }}>
                {avgPressure.toFixed(1)}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#8b8ca0' }}>Avg Pressure</div>
            </div>
            <div style={{
              padding: 12, borderRadius: 10,
              background: 'rgba(244,63,94,0.08)',
              border: '1px solid rgba(244,63,94,0.15)',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#f43f5e' }}>
                {criticalCount}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#8b8ca0' }}>Critical</div>
            </div>
            <div style={{
              padding: 12, borderRadius: 10,
              background: 'rgba(59,130,246,0.08)',
              border: '1px solid rgba(59,130,246,0.15)',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#3b82f6' }}>
                {avgRhna.toFixed(0)}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#8b8ca0' }}>RHNA Progress</div>
            </div>
          </div>

          {/* Color legend */}
          <div style={{
            display: 'flex', gap: 16, marginBottom: 12,
            justifyContent: 'center', flexWrap: 'wrap',
          }}>
            {PRESSURE_LABELS.map(level => (
              <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <div style={{
                  width: 10, height: 10, borderRadius: 3,
                  background: PRESSURE_COLORS[level],
                }} />
                <span style={{ fontSize: '0.72rem', color: '#8b8ca0' }}>{level}</span>
              </div>
            ))}
          </div>

          {/* Chart — horizontal so jurisdiction labels sit flat on the left */}
          <ResponsiveContainer width="100%" height={Math.max(sortedPressure.length * 36, 200)}>
            <BarChart
              data={sortedPressure}
              layout="vertical"
              margin={{ top: 4, right: 32, bottom: 4, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
              <YAxis
                dataKey="jurisdiction"
                type="category"
                stroke="rgba(255,255,255,0.1)"
                tickFormatter={abbreviate}
                tick={{ fill: '#8b8ca0', fontSize: 11, fontWeight: 500 }}
                width={100}
                tickLine={false}
                axisLine={false}
                interval={0}
              />
              <XAxis
                type="number"
                stroke="rgba(255,255,255,0.1)"
                tick={{ fill: '#5a5b70', fontSize: 10 }}
                domain={[0, 100]}
                tickLine={false}
                axisLine={false}
              />
              <RechartsTooltip content={<CustomTooltip />} />
              <Bar dataKey="housing_pressure_index" radius={[0, 6, 6, 0]} name="Housing Pressure" barSize={20}>
                {sortedPressure.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={PRESSURE_COLORS[entry.pressure_level] || '#3b82f6'}
                    fillOpacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Interpretation callout */}
          {worstJurisdiction && (
            <div style={{
              padding: 12, borderRadius: 10, marginTop: 16,
              background: criticalCount > 0 ? 'rgba(244,63,94,0.06)' : 'rgba(52,211,153,0.06)',
              border: `1px solid ${criticalCount > 0 ? 'rgba(244,63,94,0.15)' : 'rgba(52,211,153,0.15)'}`,
              fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.7,
            }}>
              {criticalCount > 0 ? (
                <>
                  <strong style={{ color: '#f43f5e' }}>
                    {criticalCount} jurisdiction{criticalCount > 1 ? 's' : ''} at critical pressure.
                  </strong>{' '}
                  <strong>{titleCase(worstJurisdiction.jurisdiction)}</strong> has the highest score
                  at {worstJurisdiction.housing_pressure_index?.toFixed(1)}/100, meaning very limited housing
                  capacity to absorb workers displaced by a climate event.
                  {bestJurisdiction && bestJurisdiction.jurisdiction !== worstJurisdiction.jurisdiction && (
                    <> In contrast, <strong>{titleCase(bestJurisdiction.jurisdiction)}</strong> scores just{' '}
                    {bestJurisdiction.housing_pressure_index?.toFixed(1)}/100.</>
                  )}
                </>
              ) : avgPressure > 50 ? (
                <>
                  <strong style={{ color: '#fbbf24' }}>Elevated housing pressure across the region.</strong>{' '}
                  The average score of {avgPressure.toFixed(1)}/100 suggests most jurisdictions have
                  limited spare capacity. <strong>{titleCase(worstJurisdiction.jurisdiction)}</strong> leads
                  at {worstJurisdiction.housing_pressure_index?.toFixed(1)}/100.
                </>
              ) : (
                <>
                  <strong style={{ color: '#34d399' }}>Housing pressure is generally manageable.</strong>{' '}
                  With an average of {avgPressure.toFixed(1)}/100 and no critical jurisdictions,
                  the region has reasonable capacity to absorb displaced workers.
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

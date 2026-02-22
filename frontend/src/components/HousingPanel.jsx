import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from 'recharts';
import { Home, AlertTriangle, TrendingUp } from 'lucide-react';

const PRESSURE_COLORS = {
  Critical: '#f43f5e',
  High: '#fb923c',
  Moderate: '#fbbf24',
  Low: '#34d399',
};

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
      maxWidth: 250,
    }}>
      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>
        {d.jurisdiction}
      </div>
      <div style={{ fontSize: '0.75rem', color: '#8b8ca0', lineHeight: 1.8 }}>
        Housing Pressure: <span style={{ color: PRESSURE_COLORS[d.pressure_level], fontWeight: 600 }}>
          {d.housing_pressure_index?.toFixed(1)} ({d.pressure_level})</span><br/>
        {d.rhna_target != null && <>RHNA Target: {d.rhna_target}<br/></>}
        {d.total_permitted != null && <>Total Permitted: {d.total_permitted}<br/></>}
        {d.progress_pct != null && <>Progress: {d.progress_pct?.toFixed(1)}%</>}
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

  const noData = !housing || !housing.pressure || housing.pressure.length === 0;

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
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
          <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Housing Pressure Index
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            SBCAG 2024 APR data · {housing?.stats?.total_jurisdictions || 0} jurisdictions
          </div>
        </div>
      </div>

      <div className="explanation">
        The <strong>Housing Pressure Index</strong> measures how much housing supply
        constrains climate resilience. It combines RHNA progress gaps (40%),
        affordability gaps (30%), ADU dependence (15%), and rental market pressure (15%).
        Communities with high housing pressure have <strong>fewer options for displaced workers</strong>.
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
                {housing.stats.avg_pressure?.toFixed(1)}
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
                {housing.stats.critical_jurisdictions}
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
                {housing.stats.avg_rhna_progress?.toFixed(0)}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#8b8ca0' }}>RHNA Progress</div>
            </div>
          </div>

          {/* Chart */}
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={[...housing.pressure].sort((a, b) =>
                (b.housing_pressure_index || 0) - (a.housing_pressure_index || 0)
              )}
              margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="jurisdiction"
                stroke="rgba(255,255,255,0.1)"
                tick={{ fill: '#5a5b70', fontSize: 10 }}
                angle={-30}
                textAnchor="end"
                height={60}
              />
              <YAxis
                stroke="rgba(255,255,255,0.1)"
                tick={{ fill: '#5a5b70', fontSize: 11 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="housing_pressure_index" radius={[6, 6, 0, 0]}>
                {housing.pressure.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={PRESSURE_COLORS[entry.pressure_level] || '#3b82f6'}
                    fillOpacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Waves } from 'lucide-react';

export default function NoaaPanel({ noaa, loading }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 320 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const noData = !noaa || !noaa.data || noaa.data.length === 0;

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(34,211,238,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#22d3ee',
        }}>
          <Waves size={18} />
        </span>
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Real-Time Sea Level Intelligence
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            NOAA Station 9411340 · Santa Barbara Harbor · Last 7 days
          </div>
        </div>
      </div>

      <div className="explanation">
        Live water level data from <strong>NOAA tidal gauges</strong>. High water events
        (above 5.5ft MLLW) indicate flood risk that can disrupt coastal businesses and infrastructure.
        This is the real-time environmental signal our models use.
      </div>

      {noData ? (
        <div style={{
          textAlign: 'center', padding: 40, color: 'var(--text-muted)',
        }}>
          <Waves size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <div>NOAA data unavailable — API may be down</div>
        </div>
      ) : (
        <>
          {/* Stats */}
          {noaa.stats && noaa.data?.length > 0 && (
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10,
              marginBottom: 16,
            }}>
              {[
                { label: 'Current', value: `${noaa.data[noaa.data.length - 1].water_level.toFixed(1)} ft`, color: '#22d3ee' },
                { label: 'Peak', value: `${noaa.stats.max?.toFixed(1)} ft`, color: '#f43f5e' },
                { label: 'Low', value: `${noaa.stats.min?.toFixed(1)} ft`, color: '#34d399' },
                { label: 'High Events', value: noaa.stats.high_events, color: '#fbbf24' },
              ].map((s, i) => (
                <div key={i} style={{
                  textAlign: 'center', padding: 8, borderRadius: 8,
                  background: `${s.color}08`, border: `1px solid ${s.color}15`,
                }}>
                  <div style={{
                    fontSize: '1.1rem', fontWeight: 700,
                    fontFamily: 'var(--font-mono)', color: s.color,
                  }}>
                    {s.value}
                  </div>
                  <div style={{ fontSize: '0.65rem', color: '#5a5b70' }}>{s.label}</div>
                </div>
              ))}
            </div>
          )}

          {/* Chart */}
          <ResponsiveContainer width="100%" height={200}>
            <LineChart
              data={noaa.data}
              margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="timestamp"
                stroke="rgba(255,255,255,0.1)"
                tick={{ fill: '#5a5b70', fontSize: 10 }}
                tickFormatter={v => {
                  try { return new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }); }
                  catch { return ''; }
                }}
              />
              <YAxis
                stroke="rgba(255,255,255,0.1)"
                tick={{ fill: '#5a5b70', fontSize: 10 }}
                tickFormatter={v => `${v}ft`}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(18,19,26,0.95)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 12,
                  fontSize: '0.8rem',
                }}
                labelStyle={{ color: '#e8e8f0' }}
                labelFormatter={v => new Date(v).toLocaleString('en-US', { 
                  month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
                })}
                formatter={(value) => [`${value.toFixed(2)} ft`, 'Water Level']}
              />
              <Line
                type="monotone"
                dataKey="water_level"
                stroke="#22d3ee"
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from 'recharts';
import { Waves } from 'lucide-react';
import Tooltip from './Tooltip';

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
          <div style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            Real-Time Sea Level Intelligence
            <Tooltip content="Live water level data from <strong>NOAA tidal gauges</strong>. High water events (above 5.5ft MLLW) indicate flood risk that can disrupt coastal businesses and infrastructure. This is the real-time environmental signal our models use." />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            NOAA Station 9411340 · Santa Barbara Harbor · Last 7 days
          </div>
        </div>
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
          {/* Stats — inline strip */}
          {noaa.stats && noaa.data?.length > 0 && (() => {
            const stats = [
              { label: 'Current', value: noaa.data[noaa.data.length - 1].water_level.toFixed(1), unit: 'ft', color: '#22d3ee' },
              { label: 'Peak', value: noaa.stats.max?.toFixed(1), unit: 'ft', color: '#f43f5e' },
              { label: 'Low', value: noaa.stats.min?.toFixed(1), unit: 'ft', color: '#34d399' },
              { label: 'High Events', value: noaa.stats.high_events, unit: '', color: '#fbbf24' },
            ];
            return (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                marginBottom: 16,
                padding: '10px 4px',
                borderRadius: 12,
                background: 'rgba(255,255,255,0.02)',
              }}>
                {stats.map((s, i) => (
                  <React.Fragment key={i}>
                    <div style={{
                      flex: 1,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      padding: '0 16px',
                    }}>
                      <div style={{
                        width: 3, height: 28, borderRadius: 3,
                        background: s.color,
                        opacity: 0.8,
                        flexShrink: 0,
                      }} />
                      <div style={{ minWidth: 0 }}>
                        <div style={{
                          fontSize: '0.65rem',
                          color: '#6b6c80',
                          textTransform: 'uppercase',
                          letterSpacing: '0.06em',
                          lineHeight: 1,
                          marginBottom: 3,
                          whiteSpace: 'nowrap',
                        }}>
                          {s.label}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                          <span style={{
                            fontSize: '1.25rem',
                            fontWeight: 700,
                            fontFamily: 'var(--font-mono)',
                            color: s.color,
                            lineHeight: 1,
                          }}>
                            {s.value}
                          </span>
                          {s.unit && (
                            <span style={{
                              fontSize: '0.7rem',
                              fontWeight: 500,
                              color: '#4a4b5e',
                            }}>
                              {s.unit}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    {i < stats.length - 1 && (
                      <div style={{
                        width: 1,
                        height: 24,
                        background: 'rgba(255,255,255,0.06)',
                        flexShrink: 0,
                      }} />
                    )}
                  </React.Fragment>
                ))}
              </div>
            );
          })()}

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
              <RechartsTooltip
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

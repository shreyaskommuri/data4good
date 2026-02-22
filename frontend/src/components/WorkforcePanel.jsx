import React from 'react';
import { Users, ArrowRightLeft, Briefcase } from 'lucide-react';

export default function WorkforcePanel({ workforce, loading }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 400 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const noData = !workforce || workforce.stats?.total_workers === 0;

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(167,139,250,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#a78bfa',
        }}>
          <Users size={18} />
        </span>
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Workforce Intelligence
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Live Data Technologies · {workforce?.stats?.total_workers?.toLocaleString() || 0} worker records
          </div>
        </div>
      </div>

      <div className="explanation">
        Real workforce mobility data showing <strong>where workers go</strong> when
        coastal industries contract. This tracks actual job transitions across
        15 California coastal counties — critical for predicting labor displacement patterns.
      </div>

      {noData ? (
        <div style={{
          textAlign: 'center', padding: 48, color: 'var(--text-muted)',
        }}>
          <Users size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <div>Workforce data not available</div>
          <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
            Requires Live Data Technologies JSONL files
          </div>
        </div>
      ) : (
        <>
          {/* Top transitions */}
          {workforce.transitions?.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12,
                fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)',
                textTransform: 'uppercase', letterSpacing: '0.05em',
              }}>
                <ArrowRightLeft size={14} />
                Top Industry Transitions
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {workforce.transitions.slice(0, 6).map((t, i) => {
                  const maxCount = workforce.transitions[0]?.count || 1;
                  const pct = (t.count / maxCount) * 100;
                  return (
                    <div key={i} style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      fontSize: '0.8rem',
                    }}>
                      <div style={{
                        flex: 1, display: 'flex', alignItems: 'center', gap: 6,
                        minWidth: 0,
                      }}>
                        <span style={{
                          color: '#a78bfa', whiteSpace: 'nowrap',
                          overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 120,
                        }}>
                          {t.from_industry}
                        </span>
                        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>→</span>
                        <span style={{
                          color: '#22d3ee', whiteSpace: 'nowrap',
                          overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 120,
                        }}>
                          {t.to_industry}
                        </span>
                      </div>
                      <div style={{
                        width: 100, height: 6, borderRadius: 3,
                        background: 'rgba(255,255,255,0.06)', flexShrink: 0,
                      }}>
                        <div style={{
                          width: `${pct}%`, height: '100%', borderRadius: 3,
                          background: 'linear-gradient(90deg, #a78bfa, #22d3ee)',
                        }} />
                      </div>
                      <span style={{
                        fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                        color: 'var(--text-muted)', width: 32, textAlign: 'right',
                      }}>
                        {t.count}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Industry distribution */}
          {workforce.industries?.length > 0 && (
            <div>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12,
                fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)',
                textTransform: 'uppercase', letterSpacing: '0.05em',
              }}>
                <Briefcase size={14} />
                Industry Distribution
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {workforce.industries.slice(0, 10).map((ind, i) => (
                  <div key={i} style={{
                    padding: '5px 12px', borderRadius: 20,
                    background: 'rgba(167,139,250,0.08)',
                    border: '1px solid rgba(167,139,250,0.15)',
                    fontSize: '0.75rem', color: '#a78bfa',
                  }}>
                    {ind.industry || ind.name || 'Unknown'}
                    <span style={{
                      marginLeft: 6, fontFamily: 'var(--font-mono)',
                      color: 'var(--text-muted)',
                    }}>
                      {ind.count || ind.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* County distribution */}
          {workforce.county_distribution && Object.keys(workforce.county_distribution).length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div style={{
                fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)',
                textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10,
              }}>
                County Distribution
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {Object.entries(workforce.county_distribution)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 6)
                  .map(([county, count], i) => {
                    const maxVal = Math.max(...Object.values(workforce.county_distribution));
                    return (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        fontSize: '0.8rem',
                      }}>
                        <span style={{
                          width: 120, color: 'var(--text-secondary)',
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>
                          {county}
                        </span>
                        <div style={{
                          flex: 1, height: 6, borderRadius: 3,
                          background: 'rgba(255,255,255,0.06)',
                        }}>
                          <div style={{
                            width: `${(count / maxVal) * 100}%`,
                            height: '100%', borderRadius: 3,
                            background: 'linear-gradient(90deg, #3b82f6, #a78bfa)',
                          }} />
                        </div>
                        <span style={{
                          fontFamily: 'var(--font-mono)', fontSize: '0.75rem',
                          color: 'var(--text-muted)', width: 40, textAlign: 'right',
                        }}>
                          {count}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

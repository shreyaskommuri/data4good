import React from 'react';
import { AlertTriangle, TrendingUp, MapPin, Briefcase } from 'lucide-react';

export default function DisplacementRiskPanel({ risk, loading }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 500 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const noData = !risk || risk.error || !risk.model_loaded;

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(244,63,94,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#f43f5e',
        }}>
          <AlertTriangle size={18} />
        </span>
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Displacement Risk Model
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            XGBoost ML · {risk?.total_workers || 0} workers analyzed
          </div>
        </div>
      </div>

      <div className="explanation">
        Machine learning model predicts which workers are most likely to be <strong>displaced</strong> from
        coastal counties or climate-sensitive industries. Based on job history, industry exposure, and tenure patterns.
      </div>

      {noData ? (
        <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
          <AlertTriangle size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <div>Model not available</div>
          <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
            Run: python train_model.py --test
          </div>
        </div>
      ) : (
        <>
          {/* Risk Distribution Bar */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ 
              display: 'flex', 
              height: 40, 
              borderRadius: 10, 
              overflow: 'hidden',
              border: '1px solid rgba(255,255,255,0.08)',
            }}>
              {[
                { level: 'low', count: risk.risk_distribution.low, color: '#34d399' },
                { level: 'moderate', count: risk.risk_distribution.moderate, color: '#fbbf24' },
                { level: 'high', count: risk.risk_distribution.high, color: '#fb923c' },
                { level: 'critical', count: risk.risk_distribution.critical, color: '#f43f5e' },
              ].map(({ level, count, color }) => {
                const pct = (count / risk.total_workers) * 100;
                return (
                  <div
                    key={level}
                    style={{
                      width: `${pct}%`,
                      background: color,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      color: '#000',
                    }}
                    title={`${level}: ${count} (${pct.toFixed(1)}%)`}
                  >
                    {pct > 10 && `${count}`}
                  </div>
                );
              })}
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              marginTop: 6,
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
            }}>
              <span>Low Risk</span>
              <span>Critical Risk</span>
            </div>
          </div>

          {/* Top At-Risk Workers */}
          <div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12,
              fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              <TrendingUp size={14} />
              Top 10 Highest Risk Profiles
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {risk.top_at_risk.slice(0, 10).map((worker, i) => {
                const riskColor = 
                  worker.risk_level === 'critical' ? '#f43f5e' :
                  worker.risk_level === 'high' ? '#fb923c' :
                  worker.risk_level === 'moderate' ? '#fbbf24' : '#34d399';
                
                return (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      padding: '10px 12px',
                      background: 'rgba(255,255,255,0.02)',
                      border: `1px solid ${riskColor}22`,
                      borderRadius: 8,
                    }}
                  >
                    {/* Rank */}
                    <div style={{
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      background: `${riskColor}22`,
                      color: riskColor,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      flexShrink: 0,
                    }}>
                      {i + 1}
                    </div>
                    
                    {/* Risk Score */}
                    <div style={{ flexShrink: 0 }}>
                      <div style={{
                        fontSize: '1rem',
                        fontWeight: 700,
                        color: riskColor,
                      }}>
                        {(worker.risk_score * 100).toFixed(0)}%
                      </div>
                      <div style={{
                        fontSize: '0.6rem',
                        color: 'var(--text-muted)',
                        textTransform: 'uppercase',
                      }}>
                        risk
                      </div>
                    </div>
                    
                    {/* Explanation */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: '0.75rem',
                        color: 'var(--text-primary)',
                        marginBottom: 2,
                      }}>
                        {worker.explanation}
                      </div>
                      <div style={{
                        display: 'flex',
                        gap: 8,
                        fontSize: '0.65rem',
                        color: 'var(--text-muted)',
                      }}>
                        {worker.features.climate_sensitive && (
                          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                            <Briefcase size={10} /> Climate Sensitive
                          </span>
                        )}
                        {worker.features.coastal && (
                          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                            <MapPin size={10} /> Coastal
                          </span>
                        )}
                        <span>
                          {worker.features.tenure_months}mo tenure
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

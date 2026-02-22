import React, { useState } from 'react';
import { Users, ArrowRightLeft, Briefcase, MapPin, X, Zap } from 'lucide-react';
import TransitionSankey from './TransitionSankey';
import IndustryBubbles from './IndustryBubbles';
import ClimateExposureGauge from './ClimateExposureGauge';

export default function WorkforcePanel({ workforce, loading, projected, severity = 0.5 }) {
  const [selectedIndustry, setSelectedIndustry] = useState(null);

  if (loading) {
    return (
      <div className="glass-card" style={{ padding: 24, minHeight: 600 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const noData = !workforce || workforce.stats?.total_workers === 0;
  const stats = workforce?.stats || {};
  const impact = projected?.impact || {};
  const hasImpact = impact.displacement_rate > 0;

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(167,139,250,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#a78bfa',
        }}>
          <Users size={18} />
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Workforce Intelligence
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Live Data Technologies · {stats.total_workers?.toLocaleString() || 0} workers · {stats.total_transitions?.toLocaleString() || 0} transitions
          </div>
        </div>
        {selectedIndustry && (
          <button
            onClick={() => setSelectedIndustry(null)}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '4px 10px', borderRadius: 8, cursor: 'pointer',
              border: '1px solid rgba(167,139,250,0.3)',
              background: 'rgba(167,139,250,0.1)',
              color: '#a78bfa', fontSize: '0.7rem', fontWeight: 500,
            }}
          >
            <X size={12} /> Clear filter
          </button>
        )}
      </div>

      <div className="explanation">
        Click any <strong>industry bubble</strong> to filter the chord diagram and see
        exactly where those workers transition.
        {hasImpact ? (
          <> The <strong style={{ color: '#f43f5e' }}>red numbers</strong> show projected job losses under the current shock scenario.</>
        ) : (
          <> Increase storm severity to see projected workforce displacement.</>
        )}
      </div>

      {noData ? (
        <div style={{
          textAlign: 'center', padding: 48, color: 'var(--text-muted)',
        }}>
          <Users size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <div>Workforce data not available</div>
          <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
            Requires Live Data Technologies JSONL files in drive-download directory
          </div>
        </div>
      ) : (
        <>
          {/* Severity impact banner */}
          {hasImpact && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14,
              padding: '8px 14px', borderRadius: 10,
              background: 'rgba(244,63,94,0.06)',
              border: '1px solid rgba(244,63,94,0.15)',
            }}>
              <Zap size={14} color="#f43f5e" />
              <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                At <strong style={{ color: '#fbbf24' }}>{(severity * 100).toFixed(0)}%</strong> severity:&nbsp;
                <strong style={{ color: '#f43f5e' }}>{impact.total_displaced?.toLocaleString()}</strong> displaced&nbsp;
                from sensitive sectors · <strong style={{ color: '#34d399' }}>{impact.total_absorbed?.toLocaleString()}</strong> absorbed&nbsp;
                by resilient sectors · Net loss:&nbsp;
                <strong style={{ color: '#f43f5e' }}>{impact.net_job_loss?.toLocaleString()}</strong>
              </span>
            </div>
          )}

          {/* Row 1: Gauge + Circle Packing */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '200px 1fr',
            gap: 16,
            marginBottom: 16,
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              padding: '12px 0',
            }}>
              <ClimateExposureGauge
                sensitivityPct={stats.sensitivity_pct || 0}
                sensitive={stats.climate_sensitive_workers || 0}
                resilient={stats.climate_resilient_workers || 0}
                avgJobs={stats.avg_jobs_per_worker || 0}
              />
            </div>

            <div>
              <SectionLabel icon={<Briefcase size={13} />} text={
                selectedIndustry
                  ? `Industry Landscape — filtered: ${selectedIndustry}`
                  : 'Industry Landscape — click a bubble to filter'
              } />
              <IndustryBubbles
                industries={workforce.industries}
                projected={projected}
                selectedIndustry={selectedIndustry}
                onSelect={setSelectedIndustry}
              />
            </div>
          </div>

          {/* Row 2: Sankey flow diagram (full width) */}
          {workforce.transitions?.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <SectionLabel
                icon={<ArrowRightLeft size={13} />}
                text={selectedIndustry
                  ? `Worker flows involving ${selectedIndustry}`
                  : 'Worker Flows — left = previous industry, right = next industry'
                }
              />
              <TransitionSankey
                transitions={workforce.transitions}
                selectedIndustry={selectedIndustry}
              />
            </div>
          )}

          {/* Row 3: Employment bar + County distribution */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 16,
          }}>
            {workforce.employment_breakdown && Object.keys(workforce.employment_breakdown).length > 0 && (
              <div>
                <SectionLabel icon={<Users size={13} />} text="Employment Status" />
                <EmploymentBar breakdown={workforce.employment_breakdown} />
              </div>
            )}

            {workforce.county_distribution && Object.keys(workforce.county_distribution).length > 0 && (
              <div>
                <SectionLabel icon={<MapPin size={13} />} text="Geographic Distribution" />
                <CountyBars distribution={workforce.county_distribution} />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SectionLabel({ icon, text }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8,
      fontSize: '0.73rem', fontWeight: 600, color: 'var(--text-secondary)',
      textTransform: 'uppercase', letterSpacing: '0.05em',
    }}>
      {icon}
      {text}
    </div>
  );
}

function EmploymentBar({ breakdown }) {
  const total = Object.values(breakdown).reduce((a, b) => a + b, 0);
  const colors = { employed: '#34d399', unemployed: '#f43f5e', retired: '#fbbf24' };

  return (
    <div>
      <div style={{
        display: 'flex', gap: 3, height: 24, borderRadius: 8, overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.06)',
      }}>
        {Object.entries(breakdown).map(([status, count]) => {
          const pct = (count / total) * 100;
          if (pct < 0.5) return null;
          return (
            <div
              key={status}
              style={{
                width: `${pct}%`,
                background: colors[status] || '#3b82f6',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '0.65rem', fontWeight: 600, color: '#fff',
                transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
                minWidth: pct > 5 ? 'auto' : 0,
              }}
              title={`${status}: ${count.toLocaleString()} (${pct.toFixed(1)}%)`}
            >
              {pct > 8 ? `${pct.toFixed(0)}%` : ''}
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: '0.7rem' }}>
        {Object.entries(breakdown).map(([status, count]) => (
          <div key={status} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{
              width: 7, height: 7, borderRadius: '50%',
              background: colors[status] || '#3b82f6',
            }} />
            <span style={{ color: 'var(--text-muted)', textTransform: 'capitalize' }}>
              {status}
            </span>
            <span style={{
              fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)',
              fontSize: '0.68rem',
            }}>
              {count.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CountyBars({ distribution }) {
  const entries = Object.entries(distribution)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6);
  const maxVal = entries[0]?.[1] || 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      {entries.map(([county, count], i) => (
        <div key={i} style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: '0.75rem',
        }}>
          <span style={{
            width: 120, color: 'var(--text-secondary)',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            fontSize: '0.72rem',
          }}>
            {county?.replace(' County', '')}
          </span>
          <div style={{
            flex: 1, height: 6, borderRadius: 3,
            background: 'rgba(255,255,255,0.06)',
          }}>
            <div style={{
              width: `${(count / maxVal) * 100}%`,
              height: '100%', borderRadius: 3,
              background: 'linear-gradient(90deg, #3b82f6, #a78bfa)',
              transition: 'width 0.6s ease',
            }} />
          </div>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.68rem',
            color: 'var(--text-muted)', width: 32, textAlign: 'right',
          }}>
            {count}
          </span>
        </div>
      ))}
    </div>
  );
}

import React from 'react';
import { Shield, FileText, ExternalLink, Zap } from 'lucide-react';
import Tooltip from './Tooltip';

export default function PolicySection({ sim }) {
  if (!sim) return null;

  const urgency = sim.labor_flight_pct > 20 ? 'IMMEDIATE'
    : sim.labor_flight_pct > 10 ? 'HIGH' : 'MODERATE';

  const urgencyColor = urgency === 'IMMEDIATE' ? '#f43f5e'
    : urgency === 'HIGH' ? '#fbbf24' : '#34d399';

  const recommendations = [
    {
      title: 'Emergency Workforce Stabilization Fund',
      desc: `Allocate $${(sim.emergency_fund / 1e6).toFixed(1)}M for rapid redeployment grants in the ${sim.critical_tracts} critical tracts.`,
      tags: ['Funding', 'Short-term'],
      urgency: 'immediate',
    },
    {
      title: 'EJ Community Resilience Hubs',
      desc: 'Establish multilingual workforce centers in high-EJ-burden tracts. These provide retraining, translation services, and emergency support.',
      tags: ['Equity', 'Infrastructure'],
      urgency: sim.ej_gap < -5 ? 'immediate' : 'planned',
    },
    {
      title: 'Coastal Industry Diversification',
      desc: 'Reduce single-sector dependence by incentivizing hybrid coastal-tech businesses that can operate during disruption.',
      tags: ['Economic', 'Long-term'],
      urgency: 'planned',
    },
    {
      title: 'Housing Supply Acceleration',
      desc: 'Fast-track ADU permits and affordable housing in inland tracts to provide relocation options for displaced coastal workers.',
      tags: ['Housing', 'Medium-term'],
      urgency: sim.critical_tracts > 10 ? 'immediate' : 'planned',
    },
  ];

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(52,211,153,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#34d399',
        }}>
          <Shield size={18} />
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            Policy Recommendations
            <Tooltip content="These recommendations are <strong>auto-generated from the simulation</strong>. As you adjust the shock parameters above, recommendations update in real-time. Focus on &quot;immediate&quot; items first — these address the most severe projected impacts." />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Data-driven interventions based on simulation results
          </div>
        </div>
        <span style={{
          padding: '4px 12px', borderRadius: 20,
          background: `${urgencyColor}15`,
          border: `1px solid ${urgencyColor}40`,
          color: urgencyColor,
          fontSize: '0.75rem', fontWeight: 600,
          fontFamily: 'var(--font-mono)',
        }}>
          <Zap size={12} style={{ display: 'inline', verticalAlign: -2, marginRight: 4 }} />
          {urgency} PRIORITY
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {recommendations.map((rec, i) => (
          <div key={i} style={{
            padding: 16, borderRadius: 12,
            background: rec.urgency === 'immediate'
              ? 'rgba(244,63,94,0.04)' : 'rgba(255,255,255,0.02)',
            border: `1px solid ${rec.urgency === 'immediate'
              ? 'rgba(244,63,94,0.12)' : 'rgba(255,255,255,0.06)'}`,
          }}>
            <div style={{
              display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
              marginBottom: 6,
            }}>
              <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                {rec.title}
              </div>
              {rec.urgency === 'immediate' && (
                <span style={{
                  fontSize: '0.65rem', fontWeight: 600, color: '#f43f5e',
                  background: 'rgba(244,63,94,0.1)',
                  padding: '2px 8px', borderRadius: 10,
                  textTransform: 'uppercase',
                }}>
                  Immediate
                </span>
              )}
            </div>
            <div style={{
              fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6,
              marginBottom: 8,
            }}>
              {rec.desc}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {rec.tags.map(tag => (
                <span key={tag} style={{
                  padding: '2px 8px', borderRadius: 6,
                  background: 'rgba(59,130,246,0.08)',
                  border: '1px solid rgba(59,130,246,0.15)',
                  fontSize: '0.7rem', color: '#3b82f6',
                }}>
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

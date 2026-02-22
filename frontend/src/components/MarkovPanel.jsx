import React from 'react';
import { GitBranch } from 'lucide-react';
import Tooltip from './Tooltip';

const COLORS = {
  stayed_coastal: '#3b82f6',
  to_inland: '#fbbf24',
  to_unemployed: '#f43f5e',
  to_transitioning: '#a78bfa',
};

const LABELS = {
  stayed_coastal: 'Stayed Coastal',
  to_inland: 'Moved Inland',
  to_unemployed: 'Unemployed',
  to_transitioning: 'In Transition',
};

export default function MarkovPanel({ markov, loading }) {
  if (loading || !markov) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 300 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const flows = ['stayed_coastal', 'to_inland', 'to_unemployed', 'to_transitioning'];
  const total = flows.reduce((s, k) => s + (markov[k] || 0), 0);

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(34,211,238,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#22d3ee',
        }}>
          <GitBranch size={18} />
        </span>
        <div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            Where Do Workers Go?
            <Tooltip content="A <strong>Markov chain model</strong> predicts worker movement after disruption. Starting from coastal employment, workers may stay, relocate inland, become unemployed, or enter a transition state. Higher shock severity pushes more workers out of coastal jobs." />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Markov chain state transitions after climate shock
          </div>
        </div>
      </div>

      {/* Visual flow */}
      <div style={{ marginBottom: 20 }}>
        {/* Starting state */}
        <div style={{
          textAlign: 'center', marginBottom: 16,
          padding: '10px 20px', borderRadius: 10,
          background: 'rgba(59,130,246,0.1)',
          border: '1px solid rgba(59,130,246,0.2)',
          display: 'inline-block', width: '100%',
        }}>
          <div style={{ fontSize: '0.7rem', color: '#8b8ca0', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Starting State
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#3b82f6' }}>
            100% Coastal Employed
          </div>
        </div>

        {/* Arrow */}
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '1.2rem', margin: '8px 0' }}>↓</div>

        {/* Destination bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {flows.map(key => {
            const val = markov[key] || 0;
            const color = COLORS[key];
            return (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{
                  width: 120, fontSize: '0.8rem', color: 'var(--text-secondary)',
                  textAlign: 'right',
                }}>
                  {LABELS[key]}
                </div>
                <div style={{
                  flex: 1, height: 28, borderRadius: 8,
                  background: 'rgba(255,255,255,0.04)',
                  overflow: 'hidden', position: 'relative',
                }}>
                  <div style={{
                    width: `${val}%`,
                    height: '100%',
                    borderRadius: 8,
                    background: `${color}cc`,
                    transition: 'width 0.5s ease',
                    display: 'flex', alignItems: 'center',
                    paddingLeft: val > 10 ? 10 : 0,
                  }}>
                    {val > 10 && (
                      <span style={{
                        fontSize: '0.75rem', fontWeight: 600,
                        fontFamily: 'var(--font-mono)', color: '#fff',
                      }}>
                        {val}%
                      </span>
                    )}
                  </div>
                  {val <= 10 && (
                    <span style={{
                      position: 'absolute', left: `${val + 2}%`, top: '50%',
                      transform: 'translateY(-50%)',
                      fontSize: '0.75rem', fontWeight: 600,
                      fontFamily: 'var(--font-mono)', color: color,
                    }}>
                      {val}%
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Interpretation */}
      <div style={{
        padding: 12, borderRadius: 10,
        background: markov.to_unemployed > 15 ? 'rgba(244,63,94,0.06)' : 'rgba(52,211,153,0.06)',
        border: `1px solid ${markov.to_unemployed > 15 ? 'rgba(244,63,94,0.15)' : 'rgba(52,211,153,0.15)'}`,
        fontSize: '0.8rem', color: 'var(--text-secondary)',
      }}>
        {markov.to_unemployed > 15 ? (
          <>⚠️ <strong style={{ color: '#f43f5e' }}>High displacement risk:</strong> {markov.to_unemployed}% of coastal workers face unemployment. Targeted retraining programs urgently needed.</>
        ) : markov.to_unemployed > 8 ? (
          <>⚡ <strong style={{ color: '#fbbf24' }}>Moderate displacement:</strong> {markov.to_unemployed}% unemployment rate requires monitoring and proactive intervention.</>
        ) : (
          <>✓ <strong style={{ color: '#34d399' }}>Manageable displacement:</strong> Only {markov.to_unemployed}% unemployment — existing safety nets should suffice.</>
        )}
      </div>
    </div>
  );
}

import React, { useState, useCallback, useEffect, useRef, useMemo, lazy, Suspense } from 'react'
import { useApi } from './hooks';
import * as api from './api';

// Always-eager: small, above-the-fold components
import KPIHeader from './components/KPIHeader';
import ControlPanel from './components/ControlPanel';

// Lazy-loaded: heavy library consumers — downloaded only when rendered
const TractMap          = lazy(() => import('./components/TractMap'));
const RecoveryChart     = lazy(() => import('./components/RecoveryChart'));
const MarkovPanel       = lazy(() => import('./components/MarkovPanel'));
const WorkforcePanel    = lazy(() => import('./components/WorkforcePanel'));
const NoaaPanel         = lazy(() => import('./components/NoaaPanel'));
const EconomicImpactPanel = lazy(() => import('./components/EconomicImpactPanel'));
const PolicySection     = lazy(() => import('./components/PolicySection'));
const PDFExportButton   = lazy(() => import('./components/PDFExportButton'));
const ChatPanel         = lazy(() => import('./components/ChatPanel'));

import { Shield, Database } from 'lucide-react';

// Lightweight skeleton shown while a lazy chunk is downloading
function PanelSkeleton({ height = 240 }) {
  return (
    <div style={{
      height,
      borderRadius: 12,
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.06)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'var(--text-muted)',
      fontSize: '0.75rem',
      animation: 'pulse 1.5s ease-in-out infinite',
    }}>
      Loading…
    </div>
  );
}

export default function App() {
  // Local state for immediate slider feedback (no API calls)
  const [localParams, setLocalParams] = useState({
    severity: 0.5,
    duration: 21,
    start_day: 30,
    r: 0.10,
    K: 0.95,
    sim_days: 365,
  });
  
  const [selectedTract, setSelectedTract] = useState(null);

  // Debounced params that trigger API calls
  const [params, setParams] = useState(localParams);
  const debounceTimerRef = useRef(null);

  // Debounce param updates - only update params after user stops sliding
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    debounceTimerRef.current = setTimeout(() => {
      setParams(localParams);
    }, 300); // 300ms delay after user stops adjusting

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [localParams]);

  // ── Critical path: load tracts + simulation first ──────────────────────────
  const sim = useApi(
    () => api.getSimulation(params),
    [params.severity, params.duration, params.start_day, params.r, params.K]
  );
  const tracts = useApi(
    () => api.getTracts(params.severity),
    [params.severity]
  );

  // Secondary data fires only after both critical datasets have landed.
  // This keeps the browser focused on the critical path first.
  const primaryReady = !!(sim.data && tracts.data);

  const comparison = useApi(
    () => api.getComparison(params),
    [params.severity, params.duration, params.start_day, params.r, params.K],
    { enabled: primaryReady }
  );
  const markov = useApi(
    () => api.getMarkov(params.severity, params.duration),
    [params.severity, params.duration],
    { enabled: primaryReady }
  );
  const noaa = useApi(() => api.getNoaa(), [], { enabled: primaryReady });
  const workforce = useApi(() => api.getWorkforce(), [], { enabled: primaryReady });
  const economicImpact = useApi(() => api.getEconomicImpact(), [], { enabled: primaryReady });
  const workforceProjected = useApi(
    () => api.getWorkforceProjected(params.severity, params.duration),
    [params.severity, params.duration],
    { enabled: primaryReady }
  );

  // Check if params are being debounced (localParams !== params)
  const isUpdating = useMemo(() => {
    return JSON.stringify(localParams) !== JSON.stringify(params);
  }, [localParams, params]);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg-primary)',
    }}>
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <header style={{
        padding: '16px 32px',
        background: 'rgba(10,11,15,0.8)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'linear-gradient(135deg, #3b82f6, #22d3ee)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Shield size={18} color="#fff" />
          </div>
          <div>
            <div style={{
              fontSize: '1rem', fontWeight: 700, letterSpacing: '-0.02em',
            }}>
              Coastal Labor-Resilience Engine
            </div>
            <div style={{
              fontSize: '0.7rem', color: 'var(--text-muted)',
            }}>
              Santa Barbara County · Data4Good
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {isUpdating && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 12px', borderRadius: 20,
              background: 'rgba(251,191,36,0.1)',
              border: '1px solid rgba(251,191,36,0.2)',
              animation: 'fadeIn 0.2s ease-in',
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: '#fbbf24',
              }} className="pulse" />
              <span style={{
                fontSize: '0.7rem', color: '#fbbf24', fontWeight: 500,
              }}>
                Updating...
              </span>
            </div>
          )}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '4px 12px', borderRadius: 20,
            background: 'rgba(52,211,153,0.1)',
            border: '1px solid rgba(52,211,153,0.2)',
          }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: '#34d399',
            }} className="pulse" />
            <span style={{
              fontSize: '0.7rem', color: '#34d399', fontWeight: 500,
            }}>
              Live Data
            </span>
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: '0.7rem', color: 'var(--text-muted)',
          }}>
            <Database size={12} />
            6 sources
          </div>
        </div>
      </header>

      {/* ── Main content ────────────────────────────────────────────── */}
      <main style={{
        maxWidth: 1440,
        margin: '0 auto',
        padding: '32px 32px 64px',
      }}>
        {/* KPI Header */}
        <KPIHeader sim={sim.data} loading={sim.loading} />

        {/* Controls + Map row */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '300px 1fr',
          gap: 24,
          marginBottom: 24,
        }}>
          <ControlPanel params={localParams} onChange={setLocalParams} />
          <Suspense fallback={<PanelSkeleton height={480} />}>
            <TractMap
              tracts={tracts.data}
              loading={tracts.loading}
              onSelectTract={setSelectedTract}
              economicImpact={economicImpact.data}
            />
          </Suspense>
        </div>

        {/* Recovery forecast */}
        <div style={{ marginBottom: 24 }}>
          <Suspense fallback={<PanelSkeleton height={260} />}>
            <RecoveryChart comparison={comparison.data} loading={comparison.loading} />
          </Suspense>
        </div>

        {/* Markov + NOAA row */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 24,
          marginBottom: 24,
        }}>
          <Suspense fallback={<PanelSkeleton height={300} />}>
            <MarkovPanel markov={markov.data} loading={markov.loading} />
          </Suspense>
          <Suspense fallback={<PanelSkeleton height={300} />}>
            <NoaaPanel noaa={noaa.data} loading={noaa.loading} />
          </Suspense>
        </div>

        {/* Workforce Intelligence (full width) */}
        <div style={{ marginBottom: 24 }}>
          <Suspense fallback={<PanelSkeleton height={340} />}>
            <WorkforcePanel
              workforce={workforce.data}
              loading={workforce.loading}
              projected={workforceProjected.data}
              severity={params.severity}
            />
          </Suspense>
        </div>

        {/* Economic Impact Scoring (full width) */}
        <div style={{ marginBottom: 24 }}>
          <Suspense fallback={<PanelSkeleton height={300} />}>
            <EconomicImpactPanel impact={economicImpact.data} loading={economicImpact.loading} />
          </Suspense>
        </div>

        {/* Policy */}
        <Suspense fallback={<PanelSkeleton height={200} />}>
          <PolicySection sim={sim.data} />
        </Suspense>

        {/* PDF Export Button */}
        <div style={{
          marginTop: 32,
          padding: '24px',
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '12px',
          display: 'flex',
          justifyContent: 'center',
        }}>
          <Suspense fallback={null}>
            <PDFExportButton params={params} simData={sim.data} tracts={tracts.data} selectedTract={selectedTract} onSelectTract={setSelectedTract} />
          </Suspense>
        </div>

        {/* AI Chatbot (floating) */}
        <Suspense fallback={null}>
          <ChatPanel params={params} simData={sim.data} selectedTract={selectedTract} allTracts={tracts.data} />
        </Suspense>

        {/* Footer */}
        <footer style={{
          marginTop: 48,
          padding: '24px 0',
          borderTop: '1px solid var(--border-subtle)',
          textAlign: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.75rem',
        }}>
          <div style={{ marginBottom: 4, fontWeight: 500, color: 'var(--text-secondary)' }}>
            Coastal Labor-Resilience Engine
          </div>
          <div>
            Built for Data4Good · Sources: Census ACS, NOAA CO-OPS, FEMA NFHL,
            BLS QCEW, SBCAG APR, Live Data Technologies
          </div>
          <div style={{ marginTop: 4 }}>
            ODE labor-flow model: dL/dt = rL(1−L/K) − β(EJ) · 109 census tracts
          </div>
        </footer>
      </main>
    </div>
  );
}

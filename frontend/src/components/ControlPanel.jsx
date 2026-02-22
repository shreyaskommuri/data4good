import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Sliders, History } from 'lucide-react';

const PRESETS = [
  {
    name: 'Montecito Debris Flow',
    year: 2018,
    desc: 'US-101 shut 13 days, 20k+ without power',
    severity: 0.80, duration: 13, start_day: 9, r: 0.05, K: 0.92,
  },
  {
    name: 'Thomas Fire',
    year: 2017,
    desc: '39-day regional wildfire emergency',
    severity: 0.60, duration: 39, start_day: 338, r: 0.06, K: 0.95,
  },
  {
    name: 'Alisal Fire',
    year: 2021,
    desc: 'US-101 closed ~3 days',
    severity: 0.45, duration: 3, start_day: 284, r: 0.12, K: 0.98,
  },
  {
    name: 'Winter Storms',
    year: 2023,
    desc: 'Evacuations & shelter-in-place',
    severity: 0.30, duration: 2, start_day: 9, r: 0.18, K: 0.99,
  },
  {
    name: 'Lake Fire',
    year: 2024,
    desc: '18-day evacuations in SB County',
    severity: 0.40, duration: 18, start_day: 187, r: 0.10, K: 0.97,
  },
];

function Slider({ label, value, onChange, min, max, step, format }) {
  const [localValue, setLocalValue] = useState(value);
  const [dragging, setDragging] = useState(false);
  const isDraggingRef = useRef(false);

  useEffect(() => {
    if (!isDraggingRef.current) {
      setLocalValue(value);
    }
  }, [value]);

  const handleChange = (e) => {
    const newValue = parseFloat(e.target.value);
    setLocalValue(newValue);
    isDraggingRef.current = true;
    onChange(newValue);
  };

  const handlePointerDown = () => { isDraggingRef.current = true; setDragging(true); };
  const handlePointerUp = () => {
    isDraggingRef.current = false;
    setDragging(false);
    setLocalValue(value);
  };

  const pct = ((localValue - min) / (max - min)) * 100;

  return (
    <div className={`slider-row${dragging ? ' active' : ''}`}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 6,
      }}>
        <span className="slider-label">{label}</span>
        <span className={`slider-value${dragging ? ' pop' : ''}`}>
          {format ? format(localValue) : localValue}
        </span>
      </div>
      <div className="slider-track-wrap">
        <div className="slider-fill" style={{ width: `${pct}%` }} />
        <input
          type="range"
          min={min} max={max} step={step}
          value={localValue}
          onChange={handleChange}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onLostPointerCapture={handlePointerUp}
        />
      </div>
    </div>
  );
}

export default function ControlPanel({ params, onChange, onUpdating }) {
  const [localParams, setLocalParams] = useState(params);
  const debounceRef = useRef(null);
  const committedRef = useRef(params);

  // Sync local state when parent params change (e.g. initial load)
  useEffect(() => {
    committedRef.current = params;
    setLocalParams(params);
  }, [params]);

  const flush = useCallback((next) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    onUpdating?.(true);
    debounceRef.current = setTimeout(() => {
      committedRef.current = next;
      onChange(next);
      onUpdating?.(false);
    }, 300);
  }, [onChange, onUpdating]);

  const set = useCallback((key) => (val) => {
    setLocalParams(prev => {
      const next = { ...prev, [key]: val };
      flush(next);
      return next;
    });
  }, [flush]);

  const applyPreset = useCallback((preset) => {
    const next = {
      severity: preset.severity,
      duration: preset.duration,
      start_day: preset.start_day,
      r: preset.r,
      K: preset.K,
      sim_days: 365,
    };
    setLocalParams(next);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    onChange(next);
    onUpdating?.(false);
  }, [onChange, onUpdating]);

  const activePresetIndex = useMemo(() => {
    return PRESETS.findIndex(p =>
      p.severity === localParams.severity &&
      p.duration === localParams.duration &&
      p.start_day === localParams.start_day &&
      p.r === localParams.r &&
      p.K === localParams.K
    );
  }, [localParams]);

  return (
    <div className="glass-card" style={{ padding: 24 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20,
      }}>
        <span style={{
          width: 32, height: 32, borderRadius: 8,
          background: 'rgba(59,130,246,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--accent-blue)',
        }}>
          <Sliders size={16} />
        </span>
        <div>
          <div style={{ fontSize: '0.95rem', fontWeight: 600 }}>Scenario Controls</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Adjust climate shock parameters
          </div>
        </div>
      </div>

      <Slider
        label="Shock Severity"
        value={localParams.severity}
        onChange={set('severity')}
        min={0} max={1} step={0.05}
        format={v => `${(v * 100).toFixed(0)}%`}
      />
      <Slider
        label="Duration (days)"
        value={localParams.duration}
        onChange={set('duration')}
        min={1} max={90} step={1}
        format={v => `${v}d`}
      />
      <Slider
        label="Shock Start Day"
        value={localParams.start_day}
        onChange={set('start_day')}
        min={1} max={365} step={1}
        format={v => `Day ${v}`}
      />
      <Slider
        label="Recovery Rate (r)"
        value={localParams.r}
        onChange={set('r')}
        min={0.01} max={0.30} step={0.01}
        format={v => v.toFixed(2)}
      />
      <Slider
        label="Carrying Capacity (K)"
        value={localParams.K}
        onChange={set('K')}
        min={0.8} max={1.0} step={0.01}
        format={v => `${(v * 100).toFixed(0)}%`}
      />

      {/* Divider */}
      <div style={{
        height: 1, background: 'rgba(255,255,255,0.06)', margin: '6px 0 14px',
      }} />

      {/* Disaster presets */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10,
      }}>
        <History size={13} style={{ color: 'var(--text-muted)' }} />
        <span style={{
          fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)',
          textTransform: 'uppercase', letterSpacing: '0.05em',
        }}>
          Previous SB County Disasters
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {PRESETS.map((preset, i) => {
          const isActive = i === activePresetIndex;
          return (
            <button
              key={i}
              onClick={() => applyPreset(preset)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 12px',
                borderRadius: 8,
                border: `1px solid ${isActive ? 'rgba(34,211,238,0.45)' : 'rgba(255,255,255,0.06)'}`,
                background: isActive ? 'rgba(34,211,238,0.1)' : 'rgba(255,255,255,0.02)',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                textAlign: 'left',
                width: '100%',
              }}
            >
              <span style={{
                minWidth: 32, height: 32, borderRadius: 7,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem', fontWeight: 700,
                background: isActive ? 'rgba(34,211,238,0.15)' : 'rgba(255,255,255,0.04)',
                color: isActive ? '#22d3ee' : 'var(--text-muted)',
                transition: 'all 0.2s ease',
              }}>
                {preset.year}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: '0.75rem',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? '#22d3ee' : 'var(--text-primary)',
                  transition: 'color 0.2s ease',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {preset.name}
                </div>
                <div style={{
                  fontSize: '0.63rem', color: 'var(--text-muted)',
                  marginTop: 1,
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {preset.desc}
                </div>
              </div>
              <span style={{
                fontSize: '0.6rem', fontFamily: 'var(--font-mono)',
                color: isActive ? 'rgba(34,211,238,0.7)' : 'var(--text-muted)',
                whiteSpace: 'nowrap',
                transition: 'color 0.2s ease',
              }}>
                {(preset.severity * 100).toFixed(0)}% · {preset.duration}d
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

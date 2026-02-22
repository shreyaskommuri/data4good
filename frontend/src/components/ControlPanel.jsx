import React, { useState, useEffect, useRef, useMemo } from 'react';
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
  // Local state for immediate visual feedback (slider thumb position)
  const [localValue, setLocalValue] = useState(value);
  const isDraggingRef = useRef(false);

  // Sync local value when prop changes (but not while dragging)
  useEffect(() => {
    if (!isDraggingRef.current) {
      setLocalValue(value);
    }
  }, [value]);

  // Handle slider change - update local state immediately for visual feedback
  // and update parent state (which is debounced)
  const handleChange = (e) => {
    const newValue = parseFloat(e.target.value);
    setLocalValue(newValue); // Immediate visual update
    isDraggingRef.current = true;
    onChange(newValue); // Update parent (debounced in App.jsx)
  };

  const handleMouseDown = () => {
    isDraggingRef.current = true;
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
    // Ensure final value is synced
    setLocalValue(value);
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 6,
      }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.8rem',
          color: 'var(--accent-cyan)', fontWeight: 500,
        }}>
          {format ? format(localValue) : localValue}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={localValue}
        onChange={handleChange}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onTouchStart={handleMouseDown}
        onTouchEnd={handleMouseUp}
        style={{
          width: '100%',
          cursor: 'pointer',
        }}
      />
    </div>
  );
}

export default function ControlPanel({ params, onChange }) {
  const set = (key) => (val) => onChange({ ...params, [key]: val });

  const activePresetIndex = useMemo(() => {
    return PRESETS.findIndex(p =>
      p.severity === params.severity &&
      p.duration === params.duration &&
      p.start_day === params.start_day &&
      p.r === params.r &&
      p.K === params.K
    );
  }, [params]);

  const applyPreset = (preset) => {
    onChange({
      severity: preset.severity,
      duration: preset.duration,
      start_day: preset.start_day,
      r: preset.r,
      K: preset.K,
    });
  };

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

      {/* Historical event presets */}
      <div style={{ marginBottom: 20 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8,
        }}>
          <History size={13} style={{ color: 'var(--text-muted)' }} />
          <span style={{
            fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)',
            textTransform: 'uppercase', letterSpacing: '0.05em',
          }}>
            Historical Events
          </span>
          <span style={{
            fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 400,
            textTransform: 'none', letterSpacing: 0,
          }}>
            — real SB County disasters
          </span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {PRESETS.map((preset, i) => {
            const isActive = i === activePresetIndex;
            return (
              <button
                key={i}
                onClick={() => applyPreset(preset)}
                title={preset.desc}
                style={{
                  padding: '5px 10px',
                  borderRadius: 8,
                  border: `1px solid ${isActive ? 'rgba(34,211,238,0.5)' : 'rgba(255,255,255,0.08)'}`,
                  background: isActive ? 'rgba(34,211,238,0.12)' : 'rgba(255,255,255,0.03)',
                  color: isActive ? '#22d3ee' : 'var(--text-secondary)',
                  fontSize: '0.7rem',
                  fontWeight: isActive ? 600 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  lineHeight: 1.3,
                  textAlign: 'left',
                }}
              >
                <span>{preset.name}</span>
                <span style={{
                  marginLeft: 4,
                  opacity: 0.55,
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.65rem',
                }}>
                  '{String(preset.year).slice(-2)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div style={{
        height: 1, background: 'rgba(255,255,255,0.06)', marginBottom: 18,
      }} />

      <Slider
        label="Shock Severity"
        value={params.severity}
        onChange={set('severity')}
        min={0} max={1} step={0.05}
        format={v => `${(v * 100).toFixed(0)}%`}
      />
      <Slider
        label="Duration (days)"
        value={params.duration}
        onChange={set('duration')}
        min={1} max={90} step={1}
        format={v => `${v}d`}
      />
      <Slider
        label="Shock Start Day"
        value={params.start_day}
        onChange={set('start_day')}
        min={1} max={365} step={1}
        format={v => `Day ${v}`}
      />
      <Slider
        label="Recovery Rate (r)"
        value={params.r}
        onChange={set('r')}
        min={0.01} max={0.30} step={0.01}
        format={v => v.toFixed(2)}
      />
      <Slider
        label="Carrying Capacity (K)"
        value={params.K}
        onChange={set('K')}
        min={0.8} max={1.0} step={0.01}
        format={v => `${(v * 100).toFixed(0)}%`}
      />
    </div>
  );
}

import React, { useState, useEffect, useRef } from 'react';
import { Sliders } from 'lucide-react';

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
        min={1} max={180} step={1}
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

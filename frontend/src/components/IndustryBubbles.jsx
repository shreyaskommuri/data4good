import React, { useMemo, useCallback } from 'react';
import { ResponsiveCirclePacking } from '@nivo/circle-packing';

const DARK_THEME = {
  background: 'transparent',
  labels: { text: { fill: '#ffffff', fontSize: 10, fontWeight: 600 } },
  tooltip: {
    container: {
      background: '#181924',
      color: '#e8e8f0',
      fontSize: '0.78rem',
      borderRadius: 8,
      border: '1px solid rgba(255,255,255,0.08)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      padding: '8px 12px',
    },
  },
};

export default function IndustryBubbles({ industries, projected, selectedIndustry, onSelect }) {
  const data = useMemo(() => {
    if (!industries?.length) return null;

    const projMap = {};
    if (projected?.industries) {
      projected.industries.forEach(p => { projMap[p.industry] = p; });
    }

    const sorted = [...industries].sort((a, b) => (b.count || 0) - (a.count || 0));
    const top = sorted.slice(0, 18);

    return {
      id: 'root',
      value: 0,
      children: top.map(ind => {
        const proj = projMap[ind.industry];
        return {
          id: ind.industry || 'Unknown',
          value: ind.count || 1,
          sensitive: !!ind.is_climate_sensitive,
          delta: proj?.delta || 0,
        };
      }),
    };
  }, [industries, projected]);

  const handleClick = useCallback((node) => {
    if (!node || node.depth === 0) return;
    onSelect?.(node.id === selectedIndustry ? null : node.id);
  }, [selectedIndustry, onSelect]);

  if (!data) return null;

  return (
    <div style={{ height: 280, position: 'relative' }}>
      <ResponsiveCirclePacking
        data={data}
        id="id"
        value="value"
        margin={{ top: 2, right: 2, bottom: 2, left: 2 }}
        padding={3}
        leavesOnly={true}
        colors={(node) => {
          if (node.depth === 0) return 'transparent';
          if (selectedIndustry && node.data.id !== selectedIndustry) return 'rgba(60,60,80,0.2)';
          return node.data.sensitive ? 'rgba(244,63,94,0.3)' : 'rgba(34,211,238,0.2)';
        }}
        borderWidth={(node) => {
          if (node.depth === 0) return 0;
          if (selectedIndustry === node.data.id) return 2;
          return 1;
        }}
        borderColor={(node) => {
          if (node.depth === 0) return 'transparent';
          if (selectedIndustry === node.data.id) return '#ffffff';
          if (selectedIndustry) return 'rgba(60,60,80,0.15)';
          return node.data.sensitive ? 'rgba(244,63,94,0.5)' : 'rgba(34,211,238,0.3)';
        }}
        enableLabels={true}
        labelsSkipRadius={18}
        labelsFilter={(node) => node.node.height === 0}
        labelTextColor="#ffffff"
        label={(node) => {
          const name = node.id.length > 12 ? node.id.slice(0, 10) + '..' : node.id;
          return name;
        }}
        onClick={handleClick}
        theme={DARK_THEME}
        tooltip={({ id, value, data: d }) => (
          <div>
            <div style={{
              fontWeight: 600, marginBottom: 2,
              color: d.sensitive ? '#f43f5e' : '#22d3ee',
            }}>
              {id}
            </div>
            <div style={{ color: '#a0a0b8' }}>
              {value} workers
            </div>
            <div style={{ color: d.sensitive ? '#f43f5e' : '#34d399', fontSize: '0.72rem' }}>
              {d.sensitive ? 'Climate-Sensitive' : 'Climate-Resilient'}
            </div>
            {d.delta !== 0 && (
              <div style={{
                color: d.delta < 0 ? '#f43f5e' : '#34d399', marginTop: 2,
                fontFamily: "'JetBrains Mono', monospace", fontSize: '0.75rem',
              }}>
                Shock projection: {d.delta > 0 ? '+' : ''}{d.delta}
              </div>
            )}
          </div>
        )}
        animate={true}
        motionConfig="gentle"
      />
    </div>
  );
}

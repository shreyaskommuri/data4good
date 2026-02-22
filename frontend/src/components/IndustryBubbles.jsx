import React, { useMemo, useCallback } from 'react';
import { animated, to } from '@react-spring/web';
import { ResponsiveCirclePacking } from '@nivo/circle-packing';

export default function IndustryBubbles({ industries, projected, selectedIndustry, onSelect }) {
  const data = useMemo(() => {
    if (!industries?.length) return null;

    const projMap = {};
    if (projected?.industries) {
      projected.industries.forEach(p => { projMap[p.industry] = p; });
    }

    const sorted = [...industries].sort((a, b) => (b.count || 0) - (a.count || 0));
    const top = sorted.slice(0, 16);

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
    <div style={{ height: 420, position: 'relative' }}>
      <ResponsiveCirclePacking
        data={data}
        id="id"
        value="value"
        margin={{ top: 4, right: 4, bottom: 4, left: 4 }}
        padding={6}
        leavesOnly={true}
        colors={(node) => {
          if (node.depth === 0) return 'transparent';
          if (selectedIndustry && node.data.id !== selectedIndustry) return 'rgba(60,60,80,0.15)';
          return node.data.sensitive ? 'rgba(244,63,94,0.3)' : 'rgba(34,211,238,0.18)';
        }}
        borderWidth={(node) => {
          if (node.depth === 0) return 0;
          return selectedIndustry === node.data.id ? 2.5 : 1;
        }}
        borderColor={(node) => {
          if (node.depth === 0) return 'transparent';
          if (selectedIndustry === node.data.id) return '#ffffff';
          if (selectedIndustry) return 'rgba(60,60,80,0.1)';
          return node.data.sensitive ? 'rgba(244,63,94,0.45)' : 'rgba(34,211,238,0.25)';
        }}
        enableLabels={true}
        labelsSkipRadius={8}
        labelsFilter={(node) => node.node.height === 0}
        labelTextColor={() => '#ffffff'}
        labelComponent={({ node, label, style }) => {
          if (node.height !== 0) return null;
          const r = node.radius;
          if (r < 12) return null;

          const dimmed = selectedIndustry && node.data.id !== selectedIndustry;
          const maxChars = Math.max(6, Math.floor(r / 3.5));
          const name = node.id;
          const line1 = name.length > maxChars ? name.slice(0, maxChars - 1) + '..' : name;

          const hasTwoWords = name.includes(' ') && r > 25;
          let top1, top2;
          if (hasTwoWords) {
            const parts = name.split(/\s+/);
            const mid = Math.ceil(parts.length / 2);
            top1 = parts.slice(0, mid).join(' ');
            top2 = parts.slice(mid).join(' ');
            if (top1.length > maxChars + 2) top1 = top1.slice(0, maxChars) + '..';
            if (top2.length > maxChars + 2) top2 = top2.slice(0, maxChars) + '..';
          }

          const fontSize = Math.max(8, Math.min(13, r / 3.8));
          const countSize = Math.max(7, Math.min(10, r / 5));

          return (
            <animated.g
              transform={to([style.x, style.y], (x, y) => `translate(${x},${y})`)}
              style={{ pointerEvents: 'none', opacity: style.opacity }}
            >
              {hasTwoWords ? (
                <>
                  <text
                    x={0} y={-countSize * 0.6}
                    textAnchor="middle" dominantBaseline="central"
                    style={{
                      fill: dimmed ? 'rgba(255,255,255,0.15)' : '#ffffff',
                      fontSize, fontWeight: 600,
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    {top1}
                  </text>
                  <text
                    x={0} y={fontSize * 0.6}
                    textAnchor="middle" dominantBaseline="central"
                    style={{
                      fill: dimmed ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.75)',
                      fontSize: fontSize * 0.9, fontWeight: 500,
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    {top2}
                  </text>
                </>
              ) : (
                <text
                  x={0} y={-countSize * 0.3}
                  textAnchor="middle" dominantBaseline="central"
                  style={{
                    fill: dimmed ? 'rgba(255,255,255,0.15)' : '#ffffff',
                    fontSize, fontWeight: 600,
                    fontFamily: 'Inter, sans-serif',
                  }}
                >
                  {line1}
                </text>
              )}
              {r > 20 && (
                <text
                  x={0} y={fontSize * (hasTwoWords ? 1.5 : 0.9)}
                  textAnchor="middle" dominantBaseline="central"
                  style={{
                    fill: dimmed ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.5)',
                    fontSize: countSize,
                    fontFamily: "'JetBrains Mono', monospace",
                    fontWeight: 500,
                  }}
                >
                  {node.value.toLocaleString()}
                </text>
              )}
            </animated.g>
          );
        }}
        onClick={handleClick}
        theme={{
          background: 'transparent',
          labels: { text: { fontSize: 10, fontWeight: 600, fontFamily: 'Inter, sans-serif' } },
        }}
        tooltip={({ id, value, data: d }) => (
          <BubbleTooltip id={id} value={value} sensitive={d.sensitive} delta={d.delta} />
        )}
        animate={true}
        motionConfig="gentle"
      />
    </div>
  );
}

function BubbleTooltip({ id, value, sensitive, delta }) {
  const accentColor = sensitive ? '#f43f5e' : '#22d3ee';
  const tagBg = sensitive ? 'rgba(244,63,94,0.15)' : 'rgba(34,211,238,0.15)';
  const tagBorder = sensitive ? 'rgba(244,63,94,0.3)' : 'rgba(34,211,238,0.3)';
  const tagText = sensitive ? 'Vulnerable to Shocks' : 'Shock-Resilient';

  return (
    <div style={{
      padding: '10px 14px',
      background: '#14151f',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 10,
      boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
      minWidth: 160,
      maxWidth: 240,
    }}>
      <div style={{
        fontSize: '0.88rem', fontWeight: 700,
        color: '#f0f0f5', marginBottom: 6, lineHeight: 1.3,
      }}>
        {id}
      </div>
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 8,
      }}>
        <span style={{
          fontSize: '1.3rem', fontWeight: 800,
          fontFamily: "'JetBrains Mono', monospace",
          color: accentColor,
        }}>
          {value.toLocaleString()}
        </span>
        <span style={{ fontSize: '0.72rem', color: '#8b8ca0' }}>workers</span>
      </div>
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        padding: '3px 8px', borderRadius: 6,
        background: tagBg,
        border: `1px solid ${tagBorder}`,
        fontSize: '0.65rem', fontWeight: 600,
        color: accentColor,
        marginBottom: delta !== 0 ? 8 : 0,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: accentColor }} />
        {tagText}
      </div>
      {delta !== 0 && (
        <div style={{
          padding: '5px 8px', borderRadius: 6,
          background: delta < 0 ? 'rgba(244,63,94,0.08)' : 'rgba(52,211,153,0.08)',
          border: `1px solid ${delta < 0 ? 'rgba(244,63,94,0.15)' : 'rgba(52,211,153,0.15)'}`,
          fontSize: '0.72rem',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span style={{ color: '#8b8ca0' }}>Shock impact</span>
          <span style={{
            fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
            color: delta < 0 ? '#f43f5e' : '#34d399',
          }}>
            {delta > 0 ? '+' : ''}{delta}
          </span>
        </div>
      )}
    </div>
  );
}

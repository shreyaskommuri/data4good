import React, { useMemo, useState, useCallback } from 'react';
import { ResponsiveSankey } from '@nivo/sankey';

const COLORS = [
  '#a78bfa', '#22d3ee', '#f43f5e', '#fbbf24',
  '#34d399', '#3b82f6', '#ec4899', '#f97316',
  '#8b5cf6', '#06b6d4', '#14b8a6', '#e879f9',
];

const DARK_THEME = {
  background: 'transparent',
  text: { fill: '#a0a0b8', fontSize: 11 },
  tooltip: {
    container: {
      background: '#181924',
      color: '#e8e8f0',
      fontSize: '0.78rem',
      borderRadius: 8,
      border: '1px solid rgba(255,255,255,0.08)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    },
  },
};

export default function TransitionSankey({ transitions, selectedIndustry }) {
  const [hoveredNode, setHoveredNode] = useState(null);

  const sankeyData = useMemo(() => {
    if (!transitions?.length) return null;

    const top = transitions.slice(0, 12);

    const sourceSet = new Set();
    const targetSet = new Set();
    top.forEach(t => {
      sourceSet.add(t.from_industry);
      targetSet.add(t.to_industry);
    });

    const nodeIds = new Set();
    const nodes = [];

    sourceSet.forEach(name => {
      const id = `from_${name}`;
      if (!nodeIds.has(id)) {
        nodeIds.add(id);
        nodes.push({ id, label: name });
      }
    });

    targetSet.forEach(name => {
      const id = `to_${name}`;
      if (!nodeIds.has(id)) {
        nodeIds.add(id);
        nodes.push({ id, label: name });
      }
    });

    const links = top.map(t => ({
      source: `from_${t.from_industry}`,
      target: `to_${t.to_industry}`,
      value: t.count,
      fromLabel: t.from_industry,
      toLabel: t.to_industry,
    }));

    return { nodes, links };
  }, [transitions]);

  const handleMouseEnter = useCallback((node) => {
    setHoveredNode(node.label || node.id);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoveredNode(null);
  }, []);

  if (!sankeyData) return null;

  const activeIndustry = selectedIndustry || hoveredNode;

  return (
    <div style={{ height: 340, position: 'relative' }}>
      <ResponsiveSankey
        data={sankeyData}
        margin={{ top: 12, right: 160, bottom: 12, left: 160 }}
        align="justify"
        sort="descending"
        colors={(node) => {
          const label = node.label || node.id?.replace(/^(from_|to_)/, '');
          if (activeIndustry) {
            if (label === activeIndustry || node.id?.includes(activeIndustry)) {
              const idx = sankeyData.nodes.findIndex(n => n.id === node.id);
              return COLORS[idx % COLORS.length];
            }
            return 'rgba(80,80,100,0.25)';
          }
          const idx = sankeyData.nodes.findIndex(n => n.id === node.id);
          return COLORS[idx % COLORS.length];
        }}
        nodeOpacity={1}
        nodeHoverOpacity={1}
        nodeHoverOthersOpacity={0.2}
        nodeThickness={14}
        nodeSpacing={16}
        nodeBorderWidth={0}
        nodeBorderRadius={3}
        linkOpacity={(link) => {
          if (!activeIndustry) return 0.35;
          const from = link.source?.label || link.source?.id?.replace('from_', '');
          const to = link.target?.label || link.target?.id?.replace('to_', '');
          if (from === activeIndustry || to === activeIndustry) return 0.7;
          return 0.06;
        }}
        linkHoverOpacity={0.75}
        linkHoverOthersOpacity={0.08}
        linkContract={1}
        linkBlendMode="normal"
        enableLinkGradient={true}
        enableLabels={true}
        labelPosition="outside"
        labelOrientation="horizontal"
        labelPadding={12}
        labelTextColor="#a0a0b8"
        label={(node) => node.label || node.id?.replace(/^(from_|to_)/, '')}
        theme={DARK_THEME}
        animate={true}
        motionConfig="gentle"
        nodeTooltip={({ node }) => (
          <div style={{
            padding: '8px 12px', background: '#181924',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 8, fontSize: '0.8rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          }}>
            <strong style={{ color: node.color }}>
              {node.label || node.id?.replace(/^(from_|to_)/, '')}
            </strong>
            <div style={{ color: '#a0a0b8', marginTop: 2 }}>
              {node.id?.startsWith('from_') ? 'Source' : 'Destination'}: {node.value} transitions
            </div>
          </div>
        )}
        linkTooltip={({ link }) => (
          <div style={{
            padding: '8px 12px', background: '#181924',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 8, fontSize: '0.8rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: link.source.color, fontWeight: 600 }}>
                {link.source.label || link.source.id?.replace('from_', '')}
              </span>
              <span style={{ color: '#a0a0b8' }}>→</span>
              <span style={{ color: link.target.color, fontWeight: 600 }}>
                {link.target.label || link.target.id?.replace('to_', '')}
              </span>
            </div>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              color: '#e8e8f0', marginTop: 4, fontSize: '0.9rem', fontWeight: 600,
            }}>
              {link.value} workers
            </div>
          </div>
        )}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}

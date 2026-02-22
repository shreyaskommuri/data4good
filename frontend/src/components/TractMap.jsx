import React, { useState, useMemo, useCallback } from 'react';
import MapGL, { Source, Layer, Popup, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapPin, AlertCircle, Maximize2 } from 'lucide-react';
import Tooltip from './Tooltip';

const VULN_HEX = {
  Critical: '#f43f5e',
  Moderate: '#fbbf24',
  Low: '#34d399',
};

// Free dark basemap — CARTO Dark Matter (no API key)
const DARK_STYLE = {
  version: 8,
  name: 'Dark',
  sources: {
    'carto-dark': {
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
                setViewState(v => ({ ...v, longitude: item.lon, latitude: item.lat, zoom: 12 }));
      tileSize: 256,
      attribution: '&copy; CARTO &copy; OSM',
    },
  },
  layers: [{
    id: 'carto-dark-layer',
    type: 'raster',
    source: 'carto-dark',
    minzoom: 0,
    maxzoom: 20,
  }],
};

function tractsToGeoJSON(tracts, impactMap, mode) {
  return {
    type: 'FeatureCollection',
    features: tracts.map(t => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [t.lon, t.lat] },
      properties: {
        ...t,
        impact_score: impactMap?.get(t.tract_id)?.vulnerability_score,
        impact_level: impactMap?.get(t.tract_id)?.risk_level,
        mode_value: mode === 'impact'
          ? impactMap?.get(t.tract_id)?.vulnerability_score
          : t.exodus_prob,
        radius: mode === 'impact'
          ? Math.max(6, Math.min(20, (impactMap?.get(t.tract_id)?.vulnerability_score || 0) / 5))
          : Math.max(5, Math.min(18, Math.sqrt(t.population / 400))),
        color: mode === 'impact'
          ? (VULN_HEX[impactMap?.get(t.tract_id)?.risk_level] || '#34d399')
          : (VULN_HEX[t.vulnerability] || '#34d399'),
      },
    })),
  };
}

export default function TractMap({ tracts, loading, economicImpact, onSelectTract }) {
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [mode, setMode] = useState('exodus');

  const [viewState, setViewState] = useState({
    longitude: -119.85,
    latitude: 34.55,
    zoom: 9.2,
    pitch: 0,
    bearing: 0,
  });

  const impactMap = useMemo(() => {
    if (!economicImpact?.tracts?.length) return null;
    return new globalThis.Map(economicImpact.tracts.map(t => [t.tract_id, t]));
  }, [economicImpact]);

  const geojson = useMemo(() => {
    if (!tracts?.length) return null;
    return tractsToGeoJSON(tracts, impactMap, mode);
  }, [tracts, impactMap, mode]);

  const summary = useMemo(() => {
    const s = { Critical: 0, High: 0, Moderate: 0, Low: 0 };
    if (mode === 'impact' && economicImpact?.distribution) {
      return {
        Critical: economicImpact.distribution.Critical || 0,
        High: economicImpact.distribution.High || 0,
        Moderate: economicImpact.distribution.Moderate || 0,
        Low: economicImpact.distribution.Low || 0,
      };
    }
    if (tracts) tracts.forEach(t => { s[t.vulnerability] = (s[t.vulnerability] || 0) + 1; });
    return s;
  }, [tracts, economicImpact, mode]);

  const topRisk = useMemo(() => {
    if (mode === 'impact' && economicImpact?.tracts?.length) {
      return economicImpact.tracts.slice(0, 5);
    }
    if (!tracts) return [];
    return [...tracts].sort((a, b) => b.exodus_prob - a.exodus_prob).slice(0, 5);
  }, [tracts, economicImpact, mode]);

  const handleClick = useCallback((e) => {
    const f = e.features?.[0];
    if (f) {
      const tract = { ...f.properties, lon: e.lngLat.lng, lat: e.lngLat.lat };
      setSelected(tract);
      onSelectTract?.(tract);
    } else {
      setSelected(null);
      onSelectTract?.(null);
    }
  }, [onSelectTract]);

  const handleHover = useCallback((e) => {
    if (e.features?.length) {
      setHovered(e.features[0].properties);
      e.target.getCanvas().style.cursor = 'pointer';
    } else {
      setHovered(null);
      e.target.getCanvas().style.cursor = '';
    }
  }, []);

  const resetView = useCallback(() => {
    setViewState({ longitude: -119.85, latitude: 34.55, zoom: 9.2, pitch: 0, bearing: 0 });
  }, []);

  const selectedColor = useMemo(() => {
    if (!selected) return null;
    if (mode === 'impact') {
      return VULN_HEX[selected.impact_level] || selected.color || '#34d399';
    }
    return VULN_HEX[selected.vulnerability] || selected.color || '#34d399';
  }, [selected, mode]);

  if (loading || !geojson) {
    return (
      <div className="glass-card" style={{ padding: 24, height: 580 }}>
        <div className="skeleton" style={{ height: '100%', width: '100%', borderRadius: 12 }} />
      </div>
    );
  }

  return (
    <div className="glass-card fade-in" style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'rgba(244,63,94,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#f43f5e',
        }}>
          <MapPin size={18} />
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center' }}>
            {mode === 'impact' ? 'Economic Impact Vulnerability' : 'Where Workers Are Most Vulnerable'}
            <Tooltip content={mode === 'impact'
              ? 'Each circle is a <strong>census tract</strong>. Color shows the <strong>economic impact score</strong> (0-100). Click a tract for its risk profile.'
              : 'Each circle is a <strong>census tract</strong>. Color shows <strong>exodus probability</strong>, how likely workers are to leave after a climate shock. Click any tract for a detailed breakdown.'
            } />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            {tracts.length} census tracts · Santa Barbara County · click a tract for details
          </div>
        </div>
      </div>

      {/* Controls row */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap',
      }}>
        {Object.entries(summary).map(([level, count]) => (
          <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{
              width: 10, height: 10, borderRadius: '50%',
              background: VULN_HEX[level],
              boxShadow: `0 0 6px ${VULN_HEX[level]}60`,
            }} />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
              {level} ({count})
            </span>
          </div>
        ))}
        {economicImpact?.tracts?.length && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 8 }}>
            <button
              type="button"
              onClick={() => setMode('exodus')}
              className={`map-toggle ${mode === 'exodus' ? 'active' : ''}`}
            >
              Exodus Risk
            </button>
            <button
              type="button"
              onClick={() => setMode('impact')}
              className={`map-toggle ${mode === 'impact' ? 'active' : ''}`}
            >
              Economic Impact
            </button>
          </div>
        )}
        <div style={{ flex: 1 }} />
        <button onClick={resetView} style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '4px 10px', borderRadius: 8, cursor: 'pointer',
          border: '1px solid rgba(255,255,255,0.08)',
          background: 'transparent', color: 'var(--text-muted)',
          fontSize: '0.7rem', fontWeight: 500, transition: 'all 0.2s',
        }}>
          <Maximize2 size={12} /> Reset
        </button>
      </div>

      {mode === 'impact' && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
          marginBottom: 12, fontSize: '0.7rem', color: 'var(--text-secondary)',
        }}>
          <span style={{ textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
            Score Ranges
          </span>
          {[
            { label: '0-25 Low', color: VULN_HEX.Low },
            { label: '25-50 Moderate', color: VULN_HEX.Moderate },
            { label: '50-75 High', color: VULN_HEX.High },
            { label: '75-100 Critical', color: VULN_HEX.Critical },
          ].map(range => (
            <span key={range.label} style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '2px 8px', borderRadius: 999,
              background: `${range.color}12`,
              border: `1px solid ${range.color}40`,
              color: range.color,
              fontWeight: 600,
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: range.color }} />
              {range.label}
            </span>
          ))}
        </div>
      )}

      {/* Map */}
      <div style={{
        borderRadius: 12, overflow: 'hidden', height: 420,
        position: 'relative', border: '1px solid rgba(255,255,255,0.06)',
      }}>
        <MapGL
          {...viewState}
          onMove={evt => setViewState(evt.viewState)}
          mapStyle={DARK_STYLE}
          interactiveLayerIds={['tract-dots']}
          onClick={handleClick}
          onMouseMove={handleHover}
          attributionControl={false}
          style={{ width: '100%', height: '100%' }}
        >
          <NavigationControl position="top-right" showCompass={false} />

          <Source id="tracts" type="geojson" data={geojson}>
            {/* Glow halo */}
            <Layer id="tract-halo" type="circle" paint={{
              'circle-radius': ['get', 'radius'],
              'circle-color': ['get', 'color'],
              'circle-opacity': 0.15,
              'circle-blur': 0.8,
            }} />
            {/* Main dots */}
            <Layer id="tract-dots" type="circle" paint={{
              'circle-radius': [
                'interpolate', ['linear'], ['zoom'],
                8, ['*', ['get', 'radius'], 0.6],
                12, ['*', ['get', 'radius'], 1.5],
              ],
              'circle-color': ['get', 'color'],
              'circle-opacity': 0.88,
              'circle-stroke-color': ['get', 'color'],
              'circle-stroke-width': 1.5,
              'circle-stroke-opacity': 0.3,
            }} />
          </Source>

          {/* Click popup */}
          {selected && (
            <Popup
              longitude={selected.lon}
              latitude={selected.lat}
              anchor="bottom"
              onClose={() => { setSelected(null); onSelectTract?.(null); }}
              closeButton={true}
              closeOnClick={false}
              maxWidth="280px"
            >
              <div style={{
                background: '#181924', color: '#e8e8f0',
                padding: '12px 14px', borderRadius: 10,
                fontSize: '0.8rem', lineHeight: 1.6, minWidth: 220,
              }}>
                <div style={{
                  fontWeight: 700, fontSize: '0.9rem', marginBottom: 8,
                  display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: selectedColor,
                    boxShadow: `0 0 8px ${selectedColor}80`,
                  }} />
                  {selected.name}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
                  <Stat label="Population" value={Number(selected.population).toLocaleString()} />
                  {mode === 'impact' ? (
                    <>
                      <Stat label="Vulnerability" value={`${Number(selected.impact_score || 0).toFixed(1)}`} color={selectedColor} />
                      <Stat label="Risk Level" value={selected.impact_level || 'Low'} color={selectedColor} />
                      <Stat label="Income" value={`$${(Number(selected.median_income) / 1000).toFixed(0)}k`} />
                      <Stat label="Poverty" value={`${Number(selected.poverty_pct).toFixed(1)}%`} />
                      <Stat label="Flood Zone" value={selected.flood_zone ? 'Yes' : 'No'} />
                      <Stat label="Minority" value={`${Number(selected.minority_pct).toFixed(1)}%`} />
                    </>
                  ) : (
                    <>
                      <Stat label="EJ Score" value={selected.ej_percentile}
                        color={selected.ej_percentile > 60 ? '#f43f5e' : selected.ej_percentile > 30 ? '#fbbf24' : '#34d399'} />
                      <Stat label="Coastal Jobs" value={`${selected.coastal_jobs_pct}%`} />
                      <Stat label="Exodus Risk" value={`${(Number(selected.exodus_prob) * 100).toFixed(1)}%`}
                        color={selected.exodus_prob > 0.5 ? '#f43f5e' : selected.exodus_prob > 0.3 ? '#fbbf24' : '#34d399'} />
                      <Stat label="Income" value={`$${(Number(selected.median_income) / 1000).toFixed(0)}k`} />
                      <Stat label="Poverty" value={`${Number(selected.poverty_pct).toFixed(1)}%`} />
                      <Stat label="Minority" value={`${Number(selected.minority_pct).toFixed(1)}%`} />
                    </>
                  )}
                </div>
                <div style={{
                  marginTop: 8, padding: '6px 10px', borderRadius: 6,
                  background: `${selectedColor}12`,
                  border: `1px solid ${selectedColor}30`,
                  textAlign: 'center', fontSize: '0.75rem', fontWeight: 600,
                  color: selectedColor,
                  fontFamily: 'var(--font-mono)',
                }}>
                  {mode === 'impact'
                    ? `${selected.impact_level || 'Low'} Impact`
                    : `${selected.vulnerability} Vulnerability`
                  }
                </div>
              </div>
            </Popup>
          )}
        </MapGL>

        {/* Hover floating tooltip */}
        {hovered && !selected && (
          <div style={{
            position: 'absolute', bottom: 12, left: 12,
            background: 'rgba(18,19,26,0.92)',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 10, padding: '8px 14px',
            fontSize: '0.78rem', color: '#e8e8f0',
            pointerEvents: 'none', zIndex: 5,
            display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: hovered.color || VULN_HEX[hovered.vulnerability],
              boxShadow: `0 0 8px ${hovered.color || VULN_HEX[hovered.vulnerability]}80`,
            }} />
            <span style={{ fontWeight: 600 }}>{hovered.name}</span>
            <span style={{ color: '#8b8ca0' }}>Pop: {Number(hovered.population).toLocaleString()}</span>
            <span style={{
              fontFamily: 'var(--font-mono)', fontWeight: 600,
              color: hovered.color || VULN_HEX[hovered.vulnerability],
            }}>
              {mode === 'impact'
                ? `${Number(hovered.impact_score || 0).toFixed(1)} score`
                : `${(Number(hovered.exodus_prob) * 100).toFixed(1)}% risk`
              }
            </span>
          </div>
        )}
      </div>

      {/* Most at-risk list */}
      <div style={{ marginTop: 16 }}>
        <div style={{
          fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)',
          textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8,
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <AlertCircle size={14} color="#f43f5e" />
          {mode === 'impact' ? 'Highest Economic Impact — click to zoom' : 'Highest Exodus Risk — click to zoom'}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {topRisk.map(t => (
            <button
              key={t.tract_id}
              onClick={() => {
                const base = tracts.find(baseTract => baseTract.tract_id === t.tract_id);
                const item = mode === 'impact'
                  ? {
                      ...(base || t),
                      impact_score: t.vulnerability_score,
                      impact_level: t.risk_level,
                      color: VULN_HEX[t.risk_level] || '#34d399',
                    }
                  : t;
                setSelected(item);
                onSelectTract?.(item);
                setViewState(v => ({ ...v, longitude: item.lon, latitude: item.lat, zoom: 12 }));
              }}
              style={{
                padding: '6px 14px', borderRadius: 8, cursor: 'pointer',
                background: 'rgba(244,63,94,0.08)',
                border: '1px solid rgba(244,63,94,0.2)',
                fontSize: '0.75rem', transition: 'all 0.2s',
                display: 'flex', alignItems: 'center', gap: 6,
                color: 'inherit',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = 'rgba(244,63,94,0.18)';
                e.currentTarget.style.borderColor = 'rgba(244,63,94,0.4)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = 'rgba(244,63,94,0.08)';
                e.currentTarget.style.borderColor = 'rgba(244,63,94,0.2)';
              }}
            >
              <span style={{ color: '#f43f5e', fontWeight: 600 }}>{t.name || t.tract_name}</span>
              <span style={{
                fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontWeight: 500,
              }}>
                {mode === 'impact'
                  ? `${Number(t.vulnerability_score).toFixed(0)}`
                  : `${(t.exodus_prob * 100).toFixed(0)}%`
                }
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div>
      <span style={{ color: '#5a5b70', fontSize: '0.7rem' }}>{label}: </span>
      <span style={{
        fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.78rem',
        color: color || '#e8e8f0',
      }}>{value}</span>
    </div>
  );
}

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import MapGL, { Source, Layer, NavigationControl, Marker } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapPin, Maximize2, Box, ChevronDown, Map } from 'lucide-react';
import { DARK_STYLE, VULN_HEX, METRICS, DEFAULT_VIEW, interpolateColor } from './mapConstants';

const ANIMATED_KEYS = ['exodus_prob', 'vulnerability_score', 'coastal_jobs_pct', 'structural_exposure', 'ej_percentile', 'poverty_pct'];
const LERP_MS = 800;

function getMetricRange(features, metricKey) {
  let min = Infinity, max = -Infinity;
  for (const f of features) {
    const v = Number(f.properties?.[metricKey] || 0);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!isFinite(min)) { min = 0; max = 1; }
  if (max === min) max = min + 1;
  return [min, max];
}

function colorForValue(val, min, max, ramp) {
  const t = (val - min) / (max - min);
  return interpolateColor(ramp, t);
}

function buildColorExpr(features, metricKey, ramp, fixedRange) {
  const [min, max] = fixedRange || getMetricRange(features, metricKey);
  const steps = 10;
  const expr = ['interpolate', ['linear'], ['get', metricKey]];
  for (let i = 0; i <= steps; i++) {
    const linearT = i / steps;
    const v = min + (max - min) * linearT;
    const colorT = Math.pow(linearT, 0.82);
    expr.push(v, interpolateColor(ramp, colorT));
  }
  return expr;
}

function buildHeightExpr(metricKey, features, fixedRange) {
  if (!features?.length) return 0;
  const [min, max] = fixedRange || getMetricRange(features, metricKey);
  const range = max - min || 1;
  return ['*', ['^', ['/', ['-', ['get', metricKey], min], range], 1.6], 8000];
}

export default function TractMap({ tracts, loading, onSelectTract, tractBoundaries, cityBoundaries, countyOutline, selectedTract }) {
  const mapRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [metric, setMetric] = useState('exodus_prob');
  const [viewMode, setViewMode] = useState('dots');
  const [showMetricDropdown, setShowMetricDropdown] = useState(false);
  const [metricTooltip, setMetricTooltip] = useState(false);

  const [viewState, setViewState] = useState(DEFAULT_VIEW);

  // ── Animated boundaries (lerp numeric props over 800ms) ────────────
  const [lerpedBoundaries, setLerpedBoundaries] = useState(null);
  const animPrevRef = useRef(null);
  const animRafRef = useRef(null);

  useEffect(() => {
    if (animRafRef.current) { cancelAnimationFrame(animRafRef.current); animRafRef.current = null; }

    if (!tractBoundaries?.features?.length) {
      animPrevRef.current = null;
      return;
    }

    const prev = animPrevRef.current;
    const canLerp = prev && prev.length === tractBoundaries.features.length;

    if (!canLerp) {
      animPrevRef.current = tractBoundaries.features.map(f => ({ ...f.properties }));
      return;
    }

    const fromProps = prev;
    const toProps = tractBoundaries.features.map(f => ({ ...f.properties }));
    const start = performance.now();

    const tick = (now) => {
      const elapsed = now - start;
      const t = Math.min(elapsed / LERP_MS, 1);
      const ease = 1 - Math.pow(1 - t, 3);

      setLerpedBoundaries({
        ...tractBoundaries,
        features: tractBoundaries.features.map((f, i) => {
          const mixed = { ...toProps[i] };
          for (const key of ANIMATED_KEYS) {
            const a = Number(fromProps[i][key]);
            const b = Number(toProps[i][key]);
            if (isFinite(a) && isFinite(b)) mixed[key] = a + (b - a) * ease;
          }
          return { ...f, properties: mixed };
        }),
      });

      if (t < 1) animRafRef.current = requestAnimationFrame(tick);
      else animPrevRef.current = toProps;
    };

    animRafRef.current = requestAnimationFrame(tick);
    return () => { if (animRafRef.current) cancelAnimationFrame(animRafRef.current); };
  }, [tractBoundaries]);

  // Use lerped data when available, raw data as immediate fallback
  const displayBoundaries = lerpedBoundaries ?? tractBoundaries;
  const hasBoundaries = displayBoundaries?.features?.length > 0;
  const currentMetric = METRICS.find(m => m.key === metric) || METRICS[0];
  const showPolygons = hasBoundaries && viewMode !== 'dots';
  const is3D = viewMode === '3d';

  const centroidGeojson = useMemo(() => {
    if (!hasBoundaries) return null;
    return {
      type: 'FeatureCollection',
      features: displayBoundaries.features.map(f => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [Number(f.properties.lon), Number(f.properties.lat)] },
        properties: f.properties,
      })),
    };
  }, [displayBoundaries, hasBoundaries]);

  const colorExpr = useMemo(() => {
    if (!hasBoundaries) return '#34d399';
    return buildColorExpr(displayBoundaries.features, metric, currentMetric.ramp, currentMetric.fixedRange);
  }, [displayBoundaries, metric, currentMetric, hasBoundaries]);

  const metricRange = useMemo(() => {
    if (!hasBoundaries) return [0, 1];
    return currentMetric.fixedRange || getMetricRange(displayBoundaries.features, metric);
  }, [displayBoundaries, metric, currentMetric, hasBoundaries]);

  const topRisk = useMemo(() => {
    if (!hasBoundaries) {
      if (!tracts) return [];
      return [...tracts].sort((a, b) => b.exodus_prob - a.exodus_prob).slice(0, 5);
    }
    const sorted = [...displayBoundaries.features]
      .sort((a, b) => Number(b.properties[metric] || 0) - Number(a.properties[metric] || 0));
    return sorted.slice(0, 5).map(f => f.properties);
  }, [displayBoundaries, tracts, metric, hasBoundaries]);

  // Sync external selectedTract via derived state
  const effectiveSelected = selectedTract?.tract_id ? selectedTract : selected;

  const flyTo = useCallback((lon, lat, zoom = 12) => {
    setViewState(v => ({ ...v, longitude: lon, latitude: lat, zoom, pitch: is3D ? 45 : 0 }));
  }, [is3D]);

  const handleClick = useCallback((e) => {
    const f = e.features?.[0];
    if (f?.properties) {
      const p = typeof f.properties === 'string' ? JSON.parse(f.properties) : { ...f.properties };
      p.lon = p.lon || e.lngLat.lng;
      p.lat = p.lat || e.lngLat.lat;
      setSelected(p);
      onSelectTract?.(p);
    } else {
      setSelected(null);
      onSelectTract?.(null);
    }
  }, [onSelectTract]);

  const handleHover = useCallback((e) => {
    if (e.features?.length) {
      const p = e.features[0].properties;
      setHovered(typeof p === 'string' ? JSON.parse(p) : p);
      e.target.getCanvas().style.cursor = 'pointer';
    } else {
      setHovered(null);
      e.target.getCanvas().style.cursor = '';
    }
  }, []);

  const resetView = useCallback(() => {
    setViewState({ longitude: -119.85, latitude: 34.72, zoom: 8.8, pitch: is3D ? 45 : 0, bearing: 0 });
  }, [is3D]);

  const switchViewMode = useCallback((mode) => {
    setViewMode(prev => {
      const next = prev === mode ? 'dots' : mode;
      setViewState(v => ({
        ...v,
        pitch: next === '3d' ? 50 : 0,
        bearing: next === '3d' ? -15 : 0,
      }));
      return next;
    });
  }, []);

  const selectedColor = useMemo(() => {
    if (!effectiveSelected) return '#34d399';
    const [min, max] = metricRange;
    const v = Number(effectiveSelected[metric] || 0);
    return colorForValue(v, min, max, currentMetric.ramp);
  }, [effectiveSelected, metric, metricRange, currentMetric]);

  if (loading || (!hasBoundaries && !tracts?.length)) {
    return (
      <div style={{ height: 620, borderRadius: 12, overflow: 'hidden' }}>
        <div className="skeleton" style={{ height: '100%', width: '100%' }} />
      </div>
    );
  }

  const interactiveIds = showPolygons ? ['tract-fill'] : ['tract-dots'];

  return (
    <div className="fade-in" style={{ position: 'relative', borderRadius: 14, overflow: 'hidden', height: 620 }}>
      {/* Full-bleed map */}
      <MapGL
        ref={mapRef}
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        mapStyle={DARK_STYLE}
        interactiveLayerIds={interactiveIds}
        onClick={handleClick}
        onMouseMove={handleHover}
        attributionControl={false}
        style={{ width: '100%', height: '100%' }}
      >
        <NavigationControl position="top-right" showCompass={is3D} />

          {/* Choropleth polygon layers (flat mode) */}
          {showPolygons && !is3D && (
            <Source id="boundaries" type="geojson" data={displayBoundaries}>
              <Layer id="tract-fill" type="fill" paint={{
                'fill-color': colorExpr,
                'fill-opacity': 0.78,
              }} />
              <Layer id="tract-outline" type="line" paint={{
                'line-color': 'rgba(255,255,255,0.45)',
                'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.6, 11, 1.5, 13, 2.5],
              }} />
              <Layer id="tract-highlight" type="line" filter={
                effectiveSelected?.tract_id ? ['==', 'tract_id', effectiveSelected.tract_id] : ['==', 'tract_id', '']
              } paint={{
                'line-color': '#ffffff',
                'line-width': 4,
                'line-opacity': 1,
              }} />
            </Source>
          )}

          {/* 3D extrusion layer */}
          {showPolygons && is3D && (
            <Source id="boundaries-3d" type="geojson" data={displayBoundaries}>
              <Layer id="tract-fill" type="fill-extrusion" paint={{
                'fill-extrusion-color': colorExpr,
                'fill-extrusion-height': buildHeightExpr(metric, displayBoundaries?.features, currentMetric.fixedRange),
                'fill-extrusion-base': 0,
                'fill-extrusion-opacity': 0.88,
              }} />
              <Layer id="tract-outline-3d" type="line" paint={{
                'line-color': 'rgba(255,255,255,0.35)',
                'line-width': 1.2,
              }} />
            </Source>
          )}

          {/* City boundary outlines — keep source for markers */}
          {cityBoundaries?.features?.length > 0 && (
            <Source id="city-bounds" type="geojson" data={cityBoundaries} />
          )}

          {/* Centroid dot layer — shown in dots mode or as fallback */}
          {!showPolygons && (centroidGeojson || tracts?.length > 0) && (
            <Source id="tracts-pts" type="geojson" data={centroidGeojson || {
              type: 'FeatureCollection',
              features: tracts.map(t => ({
                type: 'Feature',
                geometry: { type: 'Point', coordinates: [t.lon, t.lat] },
                properties: t,
              })),
            }}>
              <Layer id="tract-dots-glow" type="circle" paint={{
                'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 6, 12, 16],
                'circle-color': hasBoundaries ? colorExpr : ['case',
                  ['==', ['get', 'vulnerability'], 'Critical'], VULN_HEX.Critical,
                  ['==', ['get', 'vulnerability'], 'High'], VULN_HEX.High,
                  ['==', ['get', 'vulnerability'], 'Moderate'], VULN_HEX.Moderate,
                  VULN_HEX.Low,
                ],
                'circle-opacity': 0.15,
                'circle-blur': 0.8,
              }} />
              <Layer id="tract-dots" type="circle" paint={{
                'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 4, 12, 12],
                'circle-color': hasBoundaries ? colorExpr : ['case',
                  ['==', ['get', 'vulnerability'], 'Critical'], VULN_HEX.Critical,
                  ['==', ['get', 'vulnerability'], 'High'], VULN_HEX.High,
                  ['==', ['get', 'vulnerability'], 'Moderate'], VULN_HEX.Moderate,
                  VULN_HEX.Low,
                ],
                'circle-opacity': 0.9,
                'circle-stroke-width': 1.5,
                'circle-stroke-color': 'rgba(255,255,255,0.25)',
              }} />
            </Source>
          )}

          {/* City name markers — bell-curve opacity: fade when zoomed too far in or out */}
          {cityBoundaries?.features?.map(f => {
            const z = viewState.zoom;
            const peak = 10.2;
            const spread = 1.8;
            const cityOpacity = Math.max(0, Math.min(1, 1 - Math.pow((z - peak) / spread, 2)));
            if (cityOpacity < 0.05) return null;
            return (
              <Marker key={f.properties.city}
                longitude={f.properties.centroid_lon}
                latitude={f.properties.centroid_lat}
                anchor="center"
              >
                <div style={{
                  color: '#fff', fontSize: 13, fontWeight: 800,
                  textTransform: 'uppercase', letterSpacing: '0.1em',
                  textShadow: '0 0 6px rgba(0,0,0,0.9), 0 0 12px rgba(0,0,0,0.7), 0 1px 3px rgba(0,0,0,0.8)',
                  pointerEvents: 'none', whiteSpace: 'nowrap',
                  userSelect: 'none',
                  opacity: cityOpacity,
                  transition: 'opacity 0.3s ease',
                }}>
                  {f.properties.city}
                </div>
              </Marker>
            );
          })}


      </MapGL>

      {/* ── Overlays on top of map ──────────────────────────────────── */}

      {/* Top-left: header + controls bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
        padding: '14px 16px 10px',
        background: 'linear-gradient(180deg, rgba(10,11,15,0.85) 0%, rgba(10,11,15,0.5) 70%, transparent 100%)',
        pointerEvents: 'none',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, pointerEvents: 'auto' }}>
          <span style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'rgba(244,63,94,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#f43f5e',
          }}>
            <MapPin size={16} />
          </span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
              Neighborhood Vulnerability Map
            </div>
            <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.5)' }}>
              {displayBoundaries?.features?.length || tracts?.length || 0} tracts · Santa Barbara County
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', pointerEvents: 'auto' }}>
          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowMetricDropdown(v => !v)}
              onMouseEnter={() => setMetricTooltip(true)}
              onMouseLeave={() => setMetricTooltip(false)}
              className="map-toggle active"
              style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 120 }}>
              {currentMetric.label} <ChevronDown size={11} />
            </button>
            {metricTooltip && !showMetricDropdown && (
              <div style={{
                position: 'absolute', top: '100%', left: 0, marginTop: 6, zIndex: 60,
                background: 'rgba(12,13,18,0.95)', backdropFilter: 'blur(14px)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
                padding: '8px 12px', maxWidth: 240, boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
                fontSize: '0.72rem', color: '#a0a0b8', lineHeight: 1.5,
              }}>
                {currentMetric.desc}
              </div>
            )}
            {showMetricDropdown && (
              <div style={{
                position: 'absolute', top: '100%', left: 0, marginTop: 4, zIndex: 60,
                background: 'rgba(12,13,18,0.97)', backdropFilter: 'blur(14px)',
                border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10,
                padding: 4, minWidth: 280, boxShadow: '0 12px 40px rgba(0,0,0,0.7)',
              }}>
                {METRICS.map(m => (
                  <button key={m.key} onClick={() => { setMetric(m.key); setShowMetricDropdown(false); }}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left', padding: '8px 12px',
                      borderRadius: 6, border: 'none', cursor: 'pointer',
                      background: m.key === metric ? 'rgba(34,211,238,0.12)' : 'transparent',
                      color: m.key === metric ? '#22d3ee' : 'var(--text-secondary)',
                      fontWeight: m.key === metric ? 600 : 400,
                    }}>
                    <div style={{ fontSize: '0.78rem' }}>{m.label}</div>
                    <div style={{ fontSize: '0.65rem', color: '#5a5b70', marginTop: 2, lineHeight: 1.4 }}>{m.desc}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button onClick={() => switchViewMode('2d')} className={`map-toggle ${viewMode === '2d' ? 'active' : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Map size={11} /> 2D
          </button>
          <button onClick={() => switchViewMode('3d')} className={`map-toggle ${viewMode === '3d' ? 'active' : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Box size={11} /> 3D
          </button>
          <div style={{ flex: 1 }} />
          <button onClick={resetView} className="map-toggle" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Maximize2 size={11} /> Reset
          </button>
        </div>
      </div>

      {/* Embedded tract info panel */}
      {effectiveSelected && effectiveSelected.lon && (
        <div style={{
          position: 'absolute', top: 80, left: 14, zIndex: 10,
          background: 'rgba(12,13,18,0.92)', backdropFilter: 'blur(14px)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 14, padding: '14px 16px',
          fontSize: '0.8rem', color: '#e8e8f0',
          width: 250, lineHeight: 1.6,
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: '0.88rem' }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: selectedColor, boxShadow: `0 0 10px ${selectedColor}80` }} />
              {effectiveSelected.name || effectiveSelected.tract_id}
            </div>
            <button onClick={() => { setSelected(null); onSelectTract?.(null); }} style={{
              background: 'none', border: 'none', color: '#5a5b70', cursor: 'pointer', fontSize: '1.1rem', lineHeight: 1, padding: '0 2px',
            }}>&times;</button>
          </div>
          {effectiveSelected.city && <div style={{ fontSize: '0.68rem', color: '#5a5b70', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{effectiveSelected.city}</div>}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px 14px' }}>
            <Stat label="Population" value={Number(effectiveSelected.population || 0).toLocaleString()} />
            <Stat label="Income" value={`$${(Number(effectiveSelected.median_income || 0) / 1000).toFixed(0)}k`} />
            <Stat label="EJ Score" value={effectiveSelected.ej_percentile} color={Number(effectiveSelected.ej_percentile) > 60 ? '#f43f5e' : '#fbbf24'} />
            <Stat label="Coastal Jobs" value={`${effectiveSelected.coastal_jobs_pct || 0}%`} />
            <Stat label="Exodus Risk" value={`${(Number(effectiveSelected.exodus_prob || 0) * 100).toFixed(1)}%`}
              color={Number(effectiveSelected.exodus_prob) > 0.5 ? '#f43f5e' : '#fbbf24'} />
            <Stat label="Poverty" value={`${Number(effectiveSelected.poverty_pct || 0).toFixed(1)}%`} />
          </div>
          <div style={{
            marginTop: 10, padding: '5px 8px', borderRadius: 8,
            background: `${selectedColor}12`, border: `1px solid ${selectedColor}25`,
            textAlign: 'center', fontSize: '0.76rem', fontWeight: 700,
            color: selectedColor, fontFamily: 'var(--font-mono)',
          }}>
            {currentMetric.label}: {currentMetric.prefix || ''}{(Number(effectiveSelected[metric] || 0) * (currentMetric.mult || 1)).toFixed(currentMetric.decimals)}{currentMetric.unit}
          </div>
        </div>
      )}

      {/* Hover tooltip (bottom-left) */}
      {hovered && !effectiveSelected && (
        <div style={{
          position: 'absolute', bottom: 50, left: 14, zIndex: 5,
          background: 'rgba(12,13,18,0.9)', backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 10, padding: '8px 14px',
          fontSize: '0.78rem', color: '#e8e8f0',
          pointerEvents: 'none',
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: selectedColor }} />
          <span style={{ fontWeight: 600 }}>{hovered.name || hovered.tract_id}</span>
          <span style={{ color: '#8b8ca0' }}>Pop: {Number(hovered.population || 0).toLocaleString()}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#22d3ee' }}>
            {currentMetric.prefix || ''}{(Number(hovered[metric] || 0) * (currentMetric.mult || 1)).toFixed(currentMetric.decimals)}{currentMetric.unit}
          </span>
        </div>
      )}

      {/* Gradient legend (bottom-right) */}
      <div style={{
        position: 'absolute', bottom: 12, right: 12, zIndex: 5,
        background: 'rgba(12,13,18,0.88)', backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 10, padding: '8px 12px', minWidth: 160,
      }}>
        <div style={{ fontSize: '0.65rem', color: '#8b8ca0', marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {currentMetric.label}
        </div>
        <div style={{
          height: 8, borderRadius: 4, width: '100%',
          background: `linear-gradient(90deg, ${currentMetric.ramp.join(', ')})`,
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: '0.62rem', color: '#5a5b70', fontFamily: 'var(--font-mono)' }}>
          <span>{currentMetric.prefix || ''}{(metricRange[0] * (currentMetric.mult || 1)).toFixed(currentMetric.decimals)}{currentMetric.unit}</span>
          <span>{currentMetric.prefix || ''}{(metricRange[1] * (currentMetric.mult || 1)).toFixed(currentMetric.decimals)}{currentMetric.unit}</span>
        </div>
      </div>

      {/* Top-risk pills (bottom-left, above hover) */}
      <div style={{
        position: 'absolute', bottom: 12, left: 14, zIndex: 5,
        display: 'flex', gap: 6, flexWrap: 'wrap', maxWidth: '60%',
      }}>
        {topRisk.map(t => {
          const val = Number(t[metric] || 0);
          const display = currentMetric.prefix
            ? `${currentMetric.prefix}${(val * (currentMetric.mult || 1)).toFixed(currentMetric.decimals)}`
            : `${(val * (currentMetric.mult || 1)).toFixed(currentMetric.decimals)}${currentMetric.unit}`;
          return (
            <button
              key={t.tract_id}
              onClick={() => {
                setSelected(t);
                onSelectTract?.(t);
                flyTo(Number(t.lon), Number(t.lat), 12);
              }}
              style={{
                padding: '4px 10px', borderRadius: 20, cursor: 'pointer',
                background: 'rgba(12,13,18,0.85)', backdropFilter: 'blur(8px)',
                border: '1px solid rgba(244,63,94,0.25)',
                fontSize: '0.68rem', transition: 'all 0.2s',
                display: 'flex', alignItems: 'center', gap: 5,
                color: '#e8e8f0',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(244,63,94,0.5)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(244,63,94,0.25)'; }}
            >
              <span style={{ color: '#f43f5e', fontWeight: 600 }}>{t.name || t.tract_name || t.tract_id}</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: '#8b8ca0', fontWeight: 500 }}>{display}</span>
            </button>
          );
        })}
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

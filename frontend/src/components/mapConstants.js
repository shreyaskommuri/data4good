export const DARK_STYLE = {
  version: 8,
  name: 'Dark',
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
      ],
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

export const VULN_HEX = {
  Critical: '#f43f5e',
  High: '#fb923c',
  Moderate: '#fbbf24',
  Low: '#34d399',
};

export const METRICS = [
  { key: 'exodus_prob', label: 'Exodus Risk', unit: '%', mult: 100, decimals: 1, ramp: ['#34d399','#fbbf24','#f43f5e'], desc: 'Likelihood workers leave the area after a climate shock, based on EJ burden and coastal job dependency', fixedRange: [0, 0.95] },
  { key: 'vulnerability_score', label: 'Economic Impact', unit: '', mult: 1, decimals: 0, ramp: ['#34d399','#fbbf24','#fb923c','#f43f5e'], desc: 'Composite recovery risk score (0-100) from simulated shock scenarios combining ODE labor dynamics and Markov displacement', fixedRange: [0, 100] },
  { key: 'coastal_jobs_pct', label: 'Coastal Jobs', unit: '%', mult: 1, decimals: 0, ramp: ['#22d3ee','#3b82f6','#a78bfa'], desc: 'Share of employment in climate-sensitive coastal industries like tourism, fishing, and hospitality', fixedRange: [0, 80] },
];

export const DEFAULT_VIEW = {
  longitude: -119.85,
  latitude: 34.55,
  zoom: 9.2,
  pitch: 0,
  bearing: 0,
};

export function interpolateColor(ramp, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const segments = ramp.length - 1;
  const seg = Math.min(Math.floor(clamped * segments), segments - 1);
  const local = (clamped * segments) - seg;
  const c1 = hexToRgb(ramp[seg]);
  const c2 = hexToRgb(ramp[seg + 1]);
  const r = Math.round(c1[0] + (c2[0] - c1[0]) * local);
  const g = Math.round(c1[1] + (c2[1] - c1[1]) * local);
  const b = Math.round(c1[2] + (c2[2] - c1[2]) * local);
  return `rgb(${r},${g},${b})`;
}

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}

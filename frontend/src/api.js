const API = import.meta.env.VITE_API_URL || '';

async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json();
}

export async function getTracts(severity = 0.5) {
  return fetchJSON(`/api/tracts?severity=${severity}`);
}

export async function getSimulation(params = {}) {
  const q = new URLSearchParams({
    severity: params.severity ?? 0.5,
    duration: params.duration ?? 21,
    start_day: params.start_day ?? 30,
    r: params.r ?? 0.10,
    K: params.K ?? 0.95,
    sim_days: params.sim_days ?? 365,
  });
  return fetchJSON(`/api/simulation?${q}`);
}

export async function getComparison(params = {}) {
  const q = new URLSearchParams({
    severity: params.severity ?? 0.5,
    duration: params.duration ?? 21,
    start_day: params.start_day ?? 30,
    r: params.r ?? 0.10,
    K: params.K ?? 0.95,
    sim_days: params.sim_days ?? 365,
  });
  return fetchJSON(`/api/simulation/compare?${q}`);
}

export async function getNoaa() {
  return fetchJSON('/api/noaa');
}

export async function getWorkforce() {
  return fetchJSON('/api/workforce');
}

export async function getHousing() {
  return fetchJSON('/api/housing');
}

export async function getMarkov(severity = 0.5, duration = 21) {
  return fetchJSON(`/api/markov?severity=${severity}&duration=${duration}`);
}

export async function getWorkforceProjected(severity = 0.5, duration = 21) {
  return fetchJSON(`/api/workforce/projected?severity=${severity}&duration=${duration}`);
}

export async function sendChat(message, tract, params, simData) {
  const res = await fetch(`${API}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, tract, params, simData }),
  });
  if (!res.ok) throw new Error(`Chat API failed: ${res.status}`);
  return res.json();
}

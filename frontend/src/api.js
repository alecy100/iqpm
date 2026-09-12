const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  machines: () => request("/machines"),
  streamStatus: () => request("/stream/status"),
  startStream: () => request("/stream/start", { method: "POST" }),
  stopStream: () => request("/stream/stop", { method: "POST" }),
  current: (machineId) => request(`/machines/${machineId}/current`),
  history: (machineId) => request(`/machines/${machineId}/history?limit=120`),
  fleetRisk: () => request("/fleet/risk"),
  retrain: (machineId) => request(`/machines/${machineId}/retrain`, { method: "POST" }),
  deployment: (machineId) => request(`/machines/${machineId}/deployment`),
  promote: (machineId) => request(`/machines/${machineId}/deployment/promote`, { method: "POST" }),
  agent: (message, machineId) =>
    request("/agent/message", {
      method: "POST",
      body: JSON.stringify({ message, machine_id: machineId || null }),
    }),
};

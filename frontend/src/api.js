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
  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  health: () => request("/health"),
  machines: () => request("/machines"),
  sensors: () => request("/sensors"),
  streamStatus: () => request("/stream/status"),
  startStream: () => request("/stream/start", { method: "POST" }),
  stopStream: () => request("/stream/stop", { method: "POST" }),

  current: (machineId) => request(`/machines/${machineId}/current`),
  readings: (machineId, limit = 300) => request(`/machines/${machineId}/readings?limit=${limit}`),
  deployment: (machineId) => request(`/machines/${machineId}/deployment`),
  approvals: (machineId) => request(`/machines/${machineId}/approvals`),
  leaderboard: (machineId) => request(`/models/leaderboard${machineId ? `?machine_id=${machineId}` : ""}`),

  action: (machineId, action, body = {}) =>
    request(`/machines/${machineId}/actions/${action}`, { method: "POST", body: JSON.stringify(body) }),

  thread: (threadId) => request(`/agent/threads/${threadId}`),
  sendMessage: (threadId, message, machineId) =>
    request(`/agent/threads/${threadId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message, machine_id: machineId || null }),
    }),
  resume: (threadId, approved) =>
    request(`/agent/threads/${threadId}/resume`, { method: "POST", body: JSON.stringify({ approved }) }),
};

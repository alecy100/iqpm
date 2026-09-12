import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  ChevronDown,
  CircleDot,
  MessageSquarePlus,
  MonitorUp,
  Play,
  Send,
  Square,
  Wrench,
  X,
} from "lucide-react";
import { api } from "./api";
import "./styles.css";

const examples = [
  "Which machines are at risk?",
  "Explain the latest prediction",
  "Retrain the model for this machine",
];

function App() {
  const [windows, setWindows] = useState([
    { id: "chat-1", type: "chat", title: "Agent", machineId: "", messages: [] },
    { id: "monitor-1", type: "monitor", title: "Live Metrics", machineId: "" },
  ]);
  const [machines, setMachines] = useState([]);
  const [stream, setStream] = useState(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [machineData, status] = await Promise.all([api.machines(), api.streamStatus()]);
        if (!alive) return;
        setMachines(machineData);
        setStream(status);
        if (machineData.length) {
          setWindows((items) =>
            items.map((item) => (item.machineId ? item : { ...item, machineId: machineData[0] }))
          );
        }
      } catch {
        if (alive) setStream(null);
      }
    };
    tick();
    const timer = setInterval(tick, 2500);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const addWindow = (type) => {
    const id = `${type}-${Date.now()}`;
    setWindows((items) => [
      ...items,
      {
        id,
        type,
        title: type === "chat" ? "Agent" : "Live Metrics",
        machineId: machines[0] || "",
        messages: [],
      },
    ]);
  };

  const updateWindow = (id, patch) => {
    setWindows((items) => items.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  };

  const removeWindow = (id) => {
    setWindows((items) => (items.length === 1 ? items : items.filter((item) => item.id !== id)));
  };

  const toggleStream = async () => {
    const next = stream?.running ? await api.stopStream() : await api.startStream();
    setStream(next);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <CircleDot size={18} />
          <span>iqPM</span>
        </div>
        <div className="topbar-actions">
          <button className="icon-button" title="New chat" onClick={() => addWindow("chat")}>
            <MessageSquarePlus size={18} />
          </button>
          <button className="icon-button" title="New monitor" onClick={() => addWindow("monitor")}>
            <MonitorUp size={18} />
          </button>
          <button className="stream-button" onClick={toggleStream}>
            {stream?.running ? <Square size={16} /> : <Play size={16} />}
            <span>{stream?.running ? "Stop" : "Start"}</span>
          </button>
        </div>
        <div className="stream-state">
          <Activity size={16} />
          <span>{stream?.rows_streamed ?? 0} rows</span>
          <span>{stream?.labels_revealed ?? 0} labels</span>
        </div>
      </header>

      <section className="workspace">
        {windows.map((item) => (
          <WindowFrame
            key={item.id}
            item={item}
            machines={machines}
            onPatch={(patch) => updateWindow(item.id, patch)}
            onClose={() => removeWindow(item.id)}
          />
        ))}
      </section>
    </main>
  );
}

function WindowFrame({ item, machines, onPatch, onClose }) {
  return (
    <article className={`workspace-window ${item.type}`}>
      <div className="window-titlebar">
        <div className="window-title">
          {item.type === "chat" ? <Bot size={16} /> : <Activity size={16} />}
          <span>{item.title}</span>
        </div>
        <div className="window-controls">
          <MachineSelect
            value={item.machineId}
            machines={machines}
            onChange={(machineId) => onPatch({ machineId })}
          />
          <button className="icon-button tight" title="Close" onClick={onClose}>
            <X size={15} />
          </button>
        </div>
      </div>
      {item.type === "chat" ? (
        <ChatWindow window={item} onPatch={onPatch} />
      ) : (
        <MonitorWindow machineId={item.machineId} />
      )}
    </article>
  );
}

function MachineSelect({ value, machines, onChange }) {
  return (
    <label className="select-wrap">
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {!machines.length && <option value="">No stream</option>}
        {machines.map((machine) => (
          <option key={machine} value={machine}>
            {machine}
          </option>
        ))}
      </select>
      <ChevronDown size={14} />
    </label>
  );
}

function ChatWindow({ window, onPatch }) {
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const send = async (text) => {
    const message = text.trim();
    if (!message) return;
    const userMessage = { role: "user", content: message };
    onPatch({ messages: [...window.messages, userMessage] });
    setDraft("");
    setBusy(true);
    try {
      const response = await api.agent(message, window.machineId);
      onPatch({
        messages: [
          ...window.messages,
          userMessage,
          { role: "assistant", content: response.reply, tool: response.tool, data: response.data },
        ],
      });
    } catch (error) {
      onPatch({
        messages: [...window.messages, userMessage, { role: "assistant", content: error.message }],
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat-body">
      <div className="messages">
        {!window.messages.length && (
          <div className="prompt-grid">
            {examples.map((example) => (
              <button key={example} onClick={() => send(example)}>
                {example}
              </button>
            ))}
          </div>
        )}
        {window.messages.map((message, index) => (
          <MessageBubble key={`${message.role}-${index}`} message={message} />
        ))}
        {busy && <div className="bubble assistant">Working...</div>}
      </div>
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          send(draft);
        }}
      >
        <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Message iqPM" />
        <button className="icon-button" title="Send" type="submit">
          <Send size={17} />
        </button>
      </form>
    </div>
  );
}

function MessageBubble({ message }) {
  return (
    <div className={`bubble ${message.role}`}>
      <p>{message.content}</p>
      {message.tool && <span className="tool-chip">{message.tool}</span>}
      {message.data && <DataPreview data={message.data} />}
    </div>
  );
}

function DataPreview({ data }) {
  const rows = Array.isArray(data) ? data.slice(0, 3) : [data];
  return (
    <div className="data-preview">
      {rows.map((row, index) => (
        <pre key={index}>{JSON.stringify(row, null, 2)}</pre>
      ))}
    </div>
  );
}

function MonitorWindow({ machineId }) {
  const [snapshot, setSnapshot] = useState(null);
  const [history, setHistory] = useState([]);
  const [deployment, setDeployment] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!machineId) return;
    let alive = true;
    const tick = async () => {
      try {
        const [current, points, deploys] = await Promise.all([
          api.current(machineId),
          api.history(machineId),
          api.deployment(machineId),
        ]);
        if (!alive) return;
        setSnapshot(current);
        setHistory(points);
        setDeployment(deploys);
        setError("");
      } catch (err) {
        if (alive) setError(err.message);
      }
    };
    tick();
    const timer = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [machineId]);

  const sensors = useMemo(() => Object.entries(snapshot?.features || {}), [snapshot]);
  const healthPercent = Math.round((snapshot?.health || 0) * 100);
  const production = deployment.find((item) => item.stage === "Production");

  return (
    <div className="monitor-body">
      {error && <div className="empty-state">{error}</div>}
      {!error && (
        <>
          <div className="metric-strip">
            <Metric label="Health" value={`${healthPercent}%`} />
            <Metric label="Risk" value={`${Math.round((snapshot?.risk_score || 0) * 100)}%`} />
            <Metric label="Anomaly" value={`${Math.round((snapshot?.anomaly_score || 0) * 100)}%`} />
            <Metric label="Status" value={snapshot ? "Online" : "Offline"} />
          </div>
          <div className="sensor-grid">
            {sensors.map(([key, value]) => (
              <Metric key={key} label={key.replaceAll("_", " ")} value={formatSensor(value)} />
            ))}
          </div>
          <LiveGraph history={history} />
          <div className="deployment-line">
            <Wrench size={15} />
            <span>{production ? production.model_version : "heuristic"}</span>
          </div>
        </>
      )}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LiveGraph({ history }) {
  const healthPath = linePath(history.map((point) => point.health * 100));
  const riskPath = linePath(history.map((point) => point.risk_score * 100));

  return (
    <div className="graph-wrap">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <path className="graph-health" d={healthPath} />
        <path className="graph-risk" d={riskPath} />
      </svg>
      <div className="graph-labels">
        <span>Health</span>
        <span>Risk</span>
      </div>
    </div>
  );
}

function linePath(values) {
  if (!values.length) return "";
  if (values.length === 1) return `M 0 ${100 - values[0]} L 100 ${100 - values[0]}`;
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 100;
      const y = 100 - Math.max(0, Math.min(100, value));
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function formatSensor(value) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

createRoot(document.getElementById("root")).render(<App />);

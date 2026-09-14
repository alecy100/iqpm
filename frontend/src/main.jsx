import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Bot, ChevronDown, CircleDot, MessageSquarePlus, MonitorUp, Play, Square, X } from "lucide-react";
import { api } from "./api";
import { usePolling, useSplitPanes } from "./hooks";
import ChatWindow from "./components/ChatWindow";
import MonitorWindow from "./components/MonitorWindow";
import "./styles.css";

let windowCounter = 2;

function App() {
  const [panes, setPanes] = useState({
    "chat-1": { id: "chat-1", type: "chat", title: "Agent", machineId: "" },
    "monitor-1": { id: "monitor-1", type: "monitor", title: "Live Monitor", machineId: "" },
  });
  const split = useSplitPanes(["chat-1", "monitor-1"]);

  const [machines, machinesError] = usePolling(() => api.machines(), [], 3000);
  const [sensors] = usePolling(() => api.sensors(), [], 60000);
  const [stream, , refreshStream] = usePolling(() => api.streamStatus(), [], 2000);

  useEffect(() => {
    if (!machines?.length) return;
    setPanes((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const id of Object.keys(next)) {
        if (!next[id].machineId) {
          next[id] = { ...next[id], machineId: machines[0] };
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [machines]);

  const addWindow = (type) => {
    const id = `${type}-${windowCounter++}`;
    setPanes((prev) => ({
      ...prev,
      [id]: { id, type, title: type === "chat" ? "Agent" : "Live Monitor", machineId: machines?.[0] || "" },
    }));
    split.addPane(id);
  };

  const patchPane = (id, patch) => setPanes((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));

  const closePane = (id) => {
    if (split.order.length === 1) return;
    split.removePane(id);
    setPanes((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  const toggleStream = async () => {
    await (stream?.running ? api.stopStream() : api.startStream());
    refreshStream();
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
            <span>{stream?.running ? "Stop stream" : "Start stream"}</span>
          </button>
        </div>
        <div className="stream-state">
          <Activity size={16} />
          <span>{stream?.rows_streamed ?? 0} rows</span>
          {stream?.simulated_time && <span>as of {new Date(stream.simulated_time).toLocaleTimeString()}</span>}
          {machinesError && <span className="stream-warning">backend unreachable</span>}
        </div>
      </header>

      <section className="workspace" ref={split.containerRef}>
        {split.order.map((id, index) => (
          <React.Fragment key={id}>
            {index > 0 && <div className="gutter" onPointerDown={split.startDrag(index - 1)} />}
            <article className="workspace-window" style={{ flexBasis: `${split.fractions[index] * 100}%` }}>
              <WindowFrame
                pane={panes[id]}
                machines={machines || []}
                sensors={sensors || []}
                onPatch={(patch) => patchPane(id, patch)}
                onClose={() => closePane(id)}
                closable={split.order.length > 1}
              />
            </article>
          </React.Fragment>
        ))}
      </section>
    </main>
  );
}

function WindowFrame({ pane, machines, sensors, onPatch, onClose, closable }) {
  if (!pane) return null;
  return (
    <>
      <div className="window-titlebar">
        <div className="window-title">
          {pane.type === "chat" ? <Bot size={16} /> : <Activity size={16} />}
          <span>{pane.title}</span>
        </div>
        <div className="window-controls">
          <MachineSelect value={pane.machineId} machines={machines} onChange={(machineId) => onPatch({ machineId })} />
          {closable && (
            <button className="icon-button tight" title="Close" onClick={onClose}>
              <X size={15} />
            </button>
          )}
        </div>
      </div>
      {pane.type === "chat" ? (
        <ChatWindow threadId={`chat-${pane.id}`} machineId={pane.machineId} machines={machines} />
      ) : (
        <MonitorWindow machineId={pane.machineId} sensors={sensors} />
      )}
    </>
  );
}

function MachineSelect({ value, machines, onChange }) {
  return (
    <label className="select-wrap">
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {!machines.length && <option value="">No machines</option>}
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

createRoot(document.getElementById("root")).render(<App />);

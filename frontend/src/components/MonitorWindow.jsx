import React, { useEffect, useState } from "react";
import { RefreshCw, RotateCcw, Rocket, History } from "lucide-react";
import { api } from "../api";
import { usePolling } from "../hooks";
import SensorChart from "./SensorChart";
import StatusBadge from "./StatusBadge";
import ApprovalCard from "./ApprovalCard";

export default function MonitorWindow({ machineId, sensors }) {
  const [activeSensor, setActiveSensor] = useState(sensors?.[0]?.key);
  const [busyAction, setBusyAction] = useState(null);

  useEffect(() => {
    if (sensors?.length && !sensors.some((sensor) => sensor.key === activeSensor)) {
      setActiveSensor(sensors[0].key);
    }
  }, [sensors, activeSensor]);

  const ready = Boolean(machineId);
  const [current, currentError] = usePolling(
    () => (ready ? api.current(machineId) : Promise.resolve(null)),
    [machineId],
    2500
  );
  const [readingsData] = usePolling(() => (ready ? api.readings(machineId, 240) : Promise.resolve(null)), [machineId], 3000);
  const [deployment] = usePolling(() => (ready ? api.deployment(machineId) : Promise.resolve(null)), [machineId], 5000);
  const [approvals, , refreshApprovals] = usePolling(
    () => (ready ? api.approvals(machineId) : Promise.resolve([])),
    [machineId],
    3000
  );

  if (!ready) {
    return <div className="monitor-body empty-state">No machine selected yet.</div>;
  }

  const runAction = async (action) => {
    setBusyAction(action);
    try {
      await api.action(machineId, action);
      await refreshApprovals();
    } catch (err) {
      window.alert(err.message);
    } finally {
      setBusyAction(null);
    }
  };

  const resume = async (threadId, approved) => {
    await api.resume(threadId, approved);
    await refreshApprovals();
  };

  const activeSensorMeta = sensors?.find((sensor) => sensor.key === activeSensor);
  const points = (readingsData?.readings || []).map((row) => ({ timestamp: row.timestamp, value: row[activeSensor] }));

  const production = deployment?.serving?.failure_model_version;
  const staged = deployment?.failure_model_versions?.find((version) => version.stage === "Staging");
  const previousProduction = deployment?.failure_model_versions?.find(
    (version) => version.stage === "Archived" && version.tags?.promoted_at
  );

  return (
    <div className="monitor-body">
      {currentError && <div className="empty-state">{currentError}</div>}
      {!currentError && (
        <>
          <StatusBadge condition={current?.condition} />

          <div className="metric-strip">
            <Metric
              label="Failure probability"
              value={current?.failure_probability != null ? `${Math.round(current.failure_probability * 100)}%` : "—"}
            />
            <Metric label="Anomaly score" value={current?.anomaly_score != null ? current.anomaly_score.toFixed(3) : "—"} />
            <Metric label="Failure model" value={current?.failure_model_version || "none"} />
            <Metric label="Anomaly model" value={current?.anomaly_model_version || "none"} />
          </div>

          <div className="chart-panel">
            <div className="chart-tabs">
              {sensors?.map((sensor) => (
                <button
                  key={sensor.key}
                  className={sensor.key === activeSensor ? "chart-tab active" : "chart-tab"}
                  onClick={() => setActiveSensor(sensor.key)}
                >
                  {sensor.label}
                </button>
              ))}
            </div>
            <div className="chart-panel-body">
              <SensorChart points={points} label={activeSensorMeta?.label || ""} unit={activeSensorMeta?.unit || ""} />
            </div>
          </div>

          <div className="ops-row">
            <button disabled={busyAction} onClick={() => runAction("retrain")}>
              <RefreshCw size={14} /> Retrain
            </button>
            <button disabled={busyAction || !staged} onClick={() => runAction("promote")}>
              <Rocket size={14} /> Promote staged
            </button>
            <button disabled={busyAction || !previousProduction} onClick={() => runAction("rollback")}>
              <RotateCcw size={14} /> Rollback
            </button>
            <span className="ops-model-line">
              <History size={13} /> Production: {production || "none"}
              {staged && ` · Staged: ${staged.version}`}
            </span>
          </div>

          {approvals?.map((approval) => (
            <ApprovalCard key={approval.thread_id} approval={approval} onResume={resume} />
          ))}
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

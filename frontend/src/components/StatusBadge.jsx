import React from "react";
import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";

const CONDITIONS = {
  healthy: { tone: "good", icon: CheckCircle2, text: "Healthy" },
  failure_risk: { tone: "critical", icon: AlertTriangle, text: "Potential failure — monitor system closely" },
  failure_risk_and_anomaly: { tone: "critical", icon: AlertTriangle, text: "Potential failure — monitor system closely" },
  anomaly: { tone: "critical", icon: AlertTriangle, text: "Anomaly detected — check system" },
  no_models: { tone: "unknown", icon: HelpCircle, text: "No models deployed yet" },
  no_data: { tone: "unknown", icon: HelpCircle, text: "No data yet" },
};

export default function StatusBadge({ condition }) {
  const info = CONDITIONS[condition] || CONDITIONS.no_data;
  const Icon = info.icon;
  return (
    <div className={`status-badge status-${info.tone}`}>
      <Icon size={20} />
      <span>{info.text}</span>
    </div>
  );
}

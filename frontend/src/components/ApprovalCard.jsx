import React, { useState } from "react";
import { Check, X } from "lucide-react";

const TITLES = {
  approve_promotion: "Promote staged model to Production?",
  confirm_rollback: "Roll back Production model?",
};

export default function ApprovalCard({ approval, onResume }) {
  const [busy, setBusy] = useState(false);

  const decide = async (approved) => {
    setBusy(true);
    try {
      await onResume(approval.thread_id, approved);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="approval-card">
      <header>{TITLES[approval.action] || "Approval requested"}</header>
      <div className="approval-body">
        {approval.action === "approve_promotion" ? (
          <PromotionDetails approval={approval} />
        ) : (
          <RollbackDetails approval={approval} />
        )}
      </div>
      <div className="approval-actions">
        <button className="approval-approve" disabled={busy} onClick={() => decide(true)}>
          <Check size={15} /> Approve
        </button>
        <button className="approval-reject" disabled={busy} onClick={() => decide(false)}>
          <X size={15} /> Reject
        </button>
      </div>
    </div>
  );
}

function PromotionDetails({ approval }) {
  const comparison = approval.comparison || {};
  const challenger = comparison.challenger?.metrics || {};
  const incumbent = comparison.incumbent?.metrics || null;
  return (
    <>
      <p>
        Model version <strong>{approval.model_version}</strong> for <strong>{approval.machine_id}</strong>
      </p>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Challenger</th>
            <th>{incumbent ? "Incumbent" : "—"}</th>
          </tr>
        </thead>
        <tbody>
          {["f1", "precision", "recall", "pr_auc"].map((key) => (
            <tr key={key}>
              <td>{key.toUpperCase()}</td>
              <td>{formatMetric(challenger[key])}</td>
              <td>{incumbent ? formatMetric(incumbent[key]) : "no incumbent"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function RollbackDetails({ approval }) {
  return (
    <p>
      Roll <strong>{approval.machine_id}</strong> back from version{" "}
      <strong>{approval.current_version ?? "—"}</strong> to version <strong>{approval.target_version}</strong>?
    </p>
  );
}

function formatMetric(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

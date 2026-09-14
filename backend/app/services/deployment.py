"""Read-side views over the MLflow registry: deployment state, comparisons, leaderboards."""

from app.services import readings, registry, training
from app.services.serving import model_server


COMPARED_METRICS = ("f1", "precision", "recall", "pr_auc")


def deployment_status(machine_id: str) -> dict:
    readings.table_for(machine_id)
    return {
        "machine_id": machine_id,
        "serving": model_server.describe(machine_id),
        "failure_model_versions": [registry.describe_version(v) for v in registry.model_versions(machine_id, registry.FAILURE_MODEL)[:8]],
        "anomaly_model_versions": [registry.describe_version(v) for v in registry.model_versions(machine_id, registry.ANOMALY_MODEL)[:4]],
    }


def compare_models(machine_id: str) -> dict:
    readings.table_for(machine_id)
    production = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    staging = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.STAGING)
    previous = previous_production_version(machine_id)
    production_view = registry.describe_version(production) if production else None
    staging_view = registry.describe_version(staging) if staging else None
    return {
        "machine_id": machine_id,
        "production": production_view,
        "staging": staging_view,
        "previous_production": registry.describe_version(previous) if previous else None,
        "staging_minus_production": _delta(staging_view, production_view),
        "note": "Metrics come from each model's own validation run; the quality gate re-scores the incumbent on the challenger's slice.",
    }


def build_comparison(machine_id: str, challenger_version: str, gate_result: dict | None = None) -> dict:
    """Raw challenger-vs-incumbent metrics for the promotion approval step."""
    challenger = registry.get_version(machine_id, registry.FAILURE_MODEL, challenger_version)
    incumbent = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    challenger_view = registry.describe_version(challenger) if challenger else {"version": challenger_version, "metrics": {}}
    incumbent_view = registry.describe_version(incumbent) if incumbent else None
    if gate_result is not None:
        challenger_view["metrics"] = gate_result["challenger_metrics"]
        if incumbent_view is not None and gate_result.get("incumbent_metrics"):
            incumbent_view["metrics"] = gate_result["incumbent_metrics"]
    return {
        "machine_id": machine_id,
        "challenger": challenger_view,
        "incumbent": incumbent_view,
        "gate": gate_result,
        "basis": "same validation slice (quality gate)" if gate_result else "each model's own validation run",
    }


def previous_production_version(machine_id: str):
    """Most recently demoted version that was once in Production."""
    current = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    candidates = [
        version
        for version in registry.model_versions(machine_id, registry.FAILURE_MODEL)
        if (version.tags or {}).get("promoted_at") and (current is None or version.version != current.version)
    ]
    return max(candidates, key=lambda version: version.tags["promoted_at"], default=None)


def training_status(machine_id: str) -> dict:
    readings.table_for(machine_id)
    return {
        "machine_id": machine_id,
        "active_job": training.active_job(machine_id),
        "recent_supervised_runs": registry.search_runs(machine_id, "supervised", limit=5),
        "recent_anomaly_runs": registry.search_runs(machine_id, "anomaly", limit=3),
    }


def leaderboard(machine_id: str | None = None, limit: int = 10) -> list[dict]:
    machine_ids = [machine_id] if machine_id else readings.list_machines()
    rows = []
    for current in machine_ids:
        readings.table_for(current)
        stages = {v.run_id: (str(v.version), v.current_stage) for v in registry.model_versions(current, registry.FAILURE_MODEL)}
        for run in registry.search_runs(current, "supervised", limit=limit, order_by="metrics.f1 DESC"):
            version, stage = stages.get(run["run_id"], (None, None))
            rows.append(
                {
                    "machine_id": current,
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "started_at": run["started_at"],
                    "learner": run["params"].get("learner"),
                    "n_trials": run["params"].get("n_trials"),
                    "trigger": run["tags"].get("trigger"),
                    "metrics": run["metrics"],
                    "model_version": version,
                    "stage": stage,
                }
            )
    rows.sort(key=lambda row: row["metrics"].get("f1", -1), reverse=True)
    return rows[:limit]


def _delta(challenger: dict | None, incumbent: dict | None) -> dict | None:
    if not challenger or not incumbent:
        return None
    return {
        metric: challenger["metrics"][metric] - incumbent["metrics"][metric]
        for metric in COMPARED_METRICS
        if metric in challenger["metrics"] and metric in incumbent["metrics"]
    }

"""Tools exposed to the LLM.

Read-only tools are executed by the `tools` node. Governed tools are never executed directly:
the graph routes them into the retrain / promotion / rollback subflows, which own every
production state change.
"""

import json

from langchain_core.tools import tool

from app.services import analysis, deployment, drift, readings
from app.services.serialization import to_jsonable
from app.services.serving import model_server


# Keys holding UI-only payloads (chart series) that the LLM doesn't need to read.
UI_ONLY_KEYS = {"series"}


@tool
def get_prediction(machine_id: str | None = None) -> dict:
    """Latest live assessment for a machine: sensor values, supervised failure-within-horizon probability,
    Isolation Forest anomaly flag, and the model versions that produced them."""
    return model_server.assess_latest(machine_id)


@tool
def explain_prediction(machine_id: str | None = None) -> dict:
    """SHAP explanation of the Production supervised model's prediction for the latest reading:
    per-sensor contributions pushing toward failure or toward healthy."""
    return model_server.explain_latest(machine_id)


@tool
def profile_dataset(machine_id: str | None = None) -> dict:
    """Profile a machine's stored data: row counts, time range, revealed label counts and positive rate,
    and per-sensor distribution statistics over the rolling training window."""
    return analysis.profile_dataset(machine_id)


@tool
def get_leaderboard(machine_id: str | None = None, limit: int = 10) -> list:
    """MLflow experiment leaderboard of supervised retraining runs ranked by validation F1,
    including learner, trigger, and registry stage. Omit machine_id to rank across all machines."""
    return deployment.leaderboard(machine_id, limit)


@tool
def get_drift_status(machine_id: str | None = None) -> dict:
    """Sensor drift for a machine: mean population stability index between the Production model's
    training window (or the preceding window) and the most recent readings, with per-sensor PSI."""
    return drift.compute_drift(machine_id)


@tool
def plot_sensor_trend(sensor: str, machine_id: str | None = None, hours: float = 6.0) -> dict:
    """Plot one sensor over the last `hours` of simulated time; the chart is shown to the engineer.
    sensor is one of: air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min.
    Returns start/end/min/max and the fitted slope per hour."""
    return analysis.sensor_trend(machine_id, sensor, hours)


@tool
def list_at_risk_machines() -> list:
    """Every machine ranked by current condition (failure risk, anomaly, healthy), with failure probability
    and anomaly score."""
    return analysis.at_risk_machines()


@tool
def get_anomaly_score(machine_id: str | None = None, hours: float = 1.0) -> dict:
    """Isolation Forest anomaly score for the latest reading plus the anomaly rate over the last `hours`."""
    return analysis.anomaly_overview(machine_id, hours)


@tool
def get_sensor_stats(machine_id: str | None = None, hours: float = 6.0) -> dict:
    """Summary statistics per sensor over the last `hours`, with the change in mean versus the previous window."""
    return analysis.sensor_stats(machine_id, hours)


@tool
def compare_models(machine_id: str | None = None) -> dict:
    """Compare the Production, Staging, and previous Production failure models using their MLflow metrics."""
    return deployment.compare_models(machine_id)


@tool
def get_training_status(machine_id: str | None = None) -> dict:
    """Whether a training job is running now, and the most recent supervised and anomaly training runs."""
    return deployment.training_status(machine_id)


@tool
def get_deployment_status(machine_id: str | None = None) -> dict:
    """Model registry versions and stages for the failure and anomaly models, and which versions the
    serving layer currently has loaded."""
    return deployment.deployment_status(machine_id)


@tool
def suggest_action(machine_id: str | None = None) -> dict:
    """Rule-based signals and candidate next actions (inspect, retrain, review a staged model, keep monitoring)
    combining live predictions, drift, label availability, and registry state."""
    return analysis.suggest_action(machine_id)


@tool
def retrain_model(machine_id: str | None = None) -> str:
    """Start the governed retraining workflow for a machine. It checks labeled-data sufficiency, runs FLAML AutoML
    plus an Isolation Forest, applies a fixed quality gate, stages a passing model, and asks the engineer to
    approve promotion. Use only when the engineer asks to retrain."""
    raise RuntimeError("Governed tools are routed by the graph, not executed.")


@tool
def promote_staged_model(machine_id: str | None = None) -> str:
    """Ask the engineer to approve promoting the machine's current Staging model to Production.
    Use only when the engineer asks to promote or deploy."""
    raise RuntimeError("Governed tools are routed by the graph, not executed.")


@tool
def rollback_model(machine_id: str | None = None, target_version: str | None = None) -> str:
    """Ask the engineer to confirm rolling the machine's Production failure model back to an earlier version
    (defaults to the previous Production version). Use only when the engineer asks to roll back."""
    raise RuntimeError("Governed tools are routed by the graph, not executed.")


READ_ONLY_TOOLS = [
    get_prediction,
    explain_prediction,
    profile_dataset,
    get_leaderboard,
    get_drift_status,
    plot_sensor_trend,
    list_at_risk_machines,
    get_anomaly_score,
    get_sensor_stats,
    compare_models,
    get_training_status,
    get_deployment_status,
    suggest_action,
]
READ_ONLY_TOOLS_BY_NAME = {item.name: item for item in READ_ONLY_TOOLS}

RETRAIN = retrain_model.name
PROMOTE = promote_staged_model.name
ROLLBACK = rollback_model.name
GOVERNED_TOOLS = [retrain_model, promote_staged_model, rollback_model]
GOVERNED_TOOL_NAMES = {RETRAIN, PROMOTE, ROLLBACK}

ALL_TOOLS = READ_ONLY_TOOLS + GOVERNED_TOOLS


def run_read_only_tool(name: str, args: dict, default_machine_id: str | None) -> dict | list:
    selected = READ_ONLY_TOOLS_BY_NAME.get(name)
    if selected is None:
        return {"error": f"Unknown tool '{name}'."}
    args = dict(args or {})
    if "machine_id" in selected.args and not args.get("machine_id"):
        if default_machine_id is None:
            return {"error": "No machine is selected. Ask the engineer which machine they mean.", "machines": readings.list_machines()}
        args["machine_id"] = default_machine_id
    try:
        return to_jsonable(selected.invoke(args))
    except readings.UnknownMachineError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def llm_view(result) -> str:
    """Compact JSON for the model, without UI-only payloads."""
    if isinstance(result, dict):
        result = {key: value for key, value in result.items() if key not in UI_ONLY_KEYS}
    return json.dumps(result, default=str)

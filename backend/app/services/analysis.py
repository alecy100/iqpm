"""Read-only analytics over the machine tables, used by the agent tools and the dashboard."""

import numpy as np
import pandas as pd

from app.config import get_settings
from app.services import deployment, drift, readings, registry, training
from app.services.features import FEATURE_COLUMNS, FEATURE_META
from app.services.serialization import iso
from app.services.serving import model_server


CONDITION_SEVERITY = {
    "failure_risk_and_anomaly": 4,
    "failure_risk": 3,
    "anomaly": 2,
    "healthy": 1,
    "no_models": 0,
    "no_data": -1,
}


def rows_for_hours(hours: float) -> int:
    return max(2, int(hours * 3600 / get_settings().stream_step_seconds))


def profile_dataset(machine_id: str) -> dict:
    latest = readings.latest_timestamp(machine_id)
    counts = readings.row_counts(machine_id)
    if latest is None:
        return {"machine_id": machine_id, "rows": 0}
    window = readings.readings_between(machine_id, readings.training_window_start(latest), latest)
    return {
        "machine_id": machine_id,
        "rows_by_source": counts["by_source"],
        "first_timestamp": iso(counts["first_timestamp"]),
        "last_timestamp": iso(counts["last_timestamp"]),
        "rolling_window_days": get_settings().rolling_window_days,
        "rows_in_window": len(window),
        "labels": {
            **readings.label_summary(machine_id, latest),
            "reveal_lag_hours": get_settings().label_reveal_lag_hours,
            "min_rows_for_supervised_training": get_settings().min_supervised_rows,
        },
        "features": {column: _describe(window[column]) for column in FEATURE_COLUMNS},
    }


def sensor_stats(machine_id: str, hours: float = 6.0) -> dict:
    rows = rows_for_hours(hours)
    frame = readings.recent_readings(machine_id, rows * 2)
    current, previous = frame.iloc[-rows:], frame.iloc[:-rows]
    stats = {}
    for column in FEATURE_COLUMNS:
        entry = _describe(current[column])
        entry["latest"] = float(current[column].iloc[-1]) if len(current) else None
        entry["change_in_mean_vs_previous_window"] = (
            float(current[column].mean() - previous[column].mean()) if len(previous) and len(current) else None
        )
        stats[column] = entry
    return {
        "machine_id": machine_id,
        "window_hours": hours,
        "rows": len(current),
        "window_start": iso(current["timestamp"].iloc[0]) if len(current) else None,
        "window_end": iso(current["timestamp"].iloc[-1]) if len(current) else None,
        "features": stats,
    }


def sensor_trend(machine_id: str, sensor: str, hours: float = 6.0, max_points: int = 240) -> dict:
    if sensor not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown sensor '{sensor}'. Choose one of: {', '.join(FEATURE_COLUMNS)}.")
    frame = readings.recent_readings(machine_id, rows_for_hours(hours))
    if frame.empty:
        return {"machine_id": machine_id, "sensor": sensor, "points": 0}
    values = frame[sensor].to_numpy(float)
    elapsed_hours = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600
    slope = float(np.polyfit(elapsed_hours, values, 1)[0]) if len(values) > 2 else 0.0
    stride = max(1, len(frame) // max_points)
    sampled = frame.iloc[::stride]
    return {
        "machine_id": machine_id,
        "sensor": sensor,
        "label": FEATURE_META[sensor]["label"],
        "unit": FEATURE_META[sensor]["unit"],
        "window_hours": hours,
        "points": len(frame),
        "start": float(values[0]),
        "end": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "slope_per_hour": slope,
        # The chart payload is rendered by the UI and stripped from what the LLM sees.
        "series": [{"timestamp": iso(ts), "value": float(value)} for ts, value in zip(sampled["timestamp"], sampled[sensor])],
    }


def anomaly_overview(machine_id: str, hours: float = 1.0) -> dict:
    assessment = model_server.assess_latest(machine_id)
    if assessment["anomaly_model_version"] is None:
        return {"machine_id": machine_id, "error": "No anomaly model is in Production for this machine."}
    frame = readings.recent_readings(machine_id, rows_for_hours(hours))
    scores = model_server.score(machine_id, frame)
    return {
        "machine_id": machine_id,
        "model_version": assessment["anomaly_model_version"],
        "latest_score": assessment["anomaly_score"],
        "latest_is_anomaly": assessment["anomaly_detected"],
        "score_interpretation": "Isolation Forest score; values above 0 are flagged as anomalies.",
        "window_hours": hours,
        "window_rows": len(frame),
        "window_anomaly_rate": float(np.mean(scores["anomaly_detected"])) if len(frame) else None,
        "window_max_score": float(np.max(scores["anomaly_score"])) if len(frame) else None,
        "expected_rate_from_training": get_settings().anomaly_contamination,
    }


def at_risk_machines() -> list[dict]:
    assessments = [model_server.assess_latest(machine_id) for machine_id in readings.list_machines()]
    assessments.sort(
        key=lambda item: (
            CONDITION_SEVERITY.get(item["condition"], 0),
            item.get("failure_probability") or 0,
            item.get("anomaly_score") or -1,
        ),
        reverse=True,
    )
    return [
        {
            "machine_id": item["machine_id"],
            "condition": item["condition"],
            "failure_probability": item.get("failure_probability"),
            "anomaly_score": item.get("anomaly_score"),
            "timestamp": item.get("timestamp"),
        }
        for item in assessments
    ]


def suggest_action(machine_id: str) -> dict:
    settings = get_settings()
    assessment = model_server.assess_latest(machine_id)
    latest = readings.latest_timestamp(machine_id)
    labeled = readings.count_labeled_rows(machine_id, latest) if latest else 0
    staging = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.STAGING)
    drift_status = drift.compute_drift(machine_id)
    job = training.active_job(machine_id)
    anomaly = anomaly_overview(machine_id) if assessment.get("anomaly_model_version") else None

    recommendations = []

    def add(action: str, priority: str, reason: str) -> None:
        recommendations.append({"action": action, "priority": priority, "reason": reason})

    if assessment["condition"] == "no_data":
        add("start_stream", "high", "No readings exist for this machine.")
    if job:
        add("wait_for_training", "info", f"A {job['kind']} training job started at {job['started_at']} is still running.")
    if assessment.get("failure_predicted"):
        add(
            "schedule_inspection",
            "high",
            f"Supervised model gives {assessment['failure_probability']:.0%} probability of a failure within "
            f"{settings.label_horizon_hours:.0f}h.",
        )
    if assessment.get("anomaly_detected") or (anomaly and (anomaly.get("window_anomaly_rate") or 0) > 3 * settings.anomaly_contamination):
        add("check_system", "high", "Isolation Forest flags current behaviour as anomalous.")
    if assessment.get("failure_model_version") is None and labeled >= settings.min_supervised_rows:
        add("retrain", "high", f"No supervised model in Production, and {labeled} labeled rows are available.")
    elif assessment.get("anomaly_model_version") is None:
        add("retrain", "medium", "No anomaly model in Production; retraining will at least fit an Isolation Forest.")
    if drift_status["triggered"]:
        add("retrain", "high", f"Sensor drift score {drift_status['drift_score']:.2f} exceeds {drift_status['threshold']:.2f}.")
    if staging is not None:
        add("review_promotion", "medium", f"Model version {staging.version} is waiting in Staging.")
    if not recommendations:
        add("continue_monitoring", "low", "No risk signals, drift, or pending model changes.")

    return {
        "machine_id": machine_id,
        "signals": {
            "condition": assessment["condition"],
            "failure_probability": assessment.get("failure_probability"),
            "anomaly_detected": assessment.get("anomaly_detected"),
            "drift_score": drift_status["drift_score"],
            "drift_triggered": drift_status["triggered"],
            "revealed_labeled_rows": labeled,
            "staged_version": staging.version if staging else None,
            "production_versions": {
                "failure": assessment.get("failure_model_version"),
                "anomaly": assessment.get("anomaly_model_version"),
            },
        },
        "recommendations": recommendations,
    }


def _describe(series: pd.Series) -> dict:
    if series.empty:
        return {"count": 0}
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()) if series.count() > 1 else 0.0,
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }

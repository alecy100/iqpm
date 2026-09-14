"""Population stability index (PSI) drift between a reference window and the most recent readings."""

from datetime import datetime, timedelta

import numpy as np

from app.config import get_settings
from app.services import readings, registry
from app.services.features import FEATURE_COLUMNS
from app.services.serialization import iso


# Tool wear is a sawtooth counter that resets after every repair, so its distribution shifts by design.
DRIFT_FEATURES = [column for column in FEATURE_COLUMNS if column != "tool_wear_min"]
MIN_REFERENCE_ROWS = 200
MIN_CURRENT_ROWS = 30


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected = np.clip(expected, 1e-4, None)
    actual = np.clip(actual, 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_drift(machine_id: str) -> dict:
    settings = get_settings()
    latest = readings.latest_timestamp(machine_id)
    if latest is None:
        return _result(machine_id, 0.0, {}, "no readings yet", None, None, reason="No readings available.")

    window = timedelta(hours=settings.drift_window_hours)
    production = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    cutoff_tag = (production.tags or {}).get("snapshot_cutoff") if production is not None else None
    if cutoff_tag:
        reference_end = datetime.fromisoformat(cutoff_tag)
        basis = f"training window of production model v{production.version}"
    else:
        reference_end = latest - window
        basis = "preceding rolling window (no production model yet)"

    reference = readings.readings_between(machine_id, readings.training_window_start(reference_end), reference_end)
    current = readings.readings_between(machine_id, latest - window, latest)
    reference_span = {"start": iso(reference["timestamp"].min()) if len(reference) else None, "end": iso(reference_end), "rows": len(reference)}
    current_span = {"start": iso(latest - window), "end": iso(latest), "rows": len(current)}

    if len(reference) < MIN_REFERENCE_ROWS or len(current) < MIN_CURRENT_ROWS:
        return _result(machine_id, 0.0, {}, basis, reference_span, current_span, reason="Not enough rows to compare windows.")

    per_feature = {
        column: population_stability_index(reference[column].to_numpy(float), current[column].to_numpy(float))
        for column in DRIFT_FEATURES
    }
    score = float(np.mean(list(per_feature.values())))
    return _result(machine_id, score, per_feature, basis, reference_span, current_span)


def _result(machine_id, score, per_feature, basis, reference, current, reason=None) -> dict:
    threshold = get_settings().drift_threshold
    return {
        "machine_id": machine_id,
        "drift_score": score,
        "threshold": threshold,
        "triggered": score > threshold,
        "metric": "mean population stability index across sensors (tool wear excluded)",
        "per_feature_psi": per_feature,
        "reference_basis": basis,
        "reference_window": reference,
        "current_window": current,
        "note": reason,
    }

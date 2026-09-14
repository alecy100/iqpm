"""In-memory serving layer: holds each machine's Production models loaded from the MLflow registry."""

import logging
import threading
import time

import numpy as np
import pandas as pd

from app.config import get_settings
from app.services import readings, registry
from app.services.features import FEATURE_COLUMNS, FEATURE_META
from app.services.serialization import iso


logger = logging.getLogger(__name__)

RETRY_AFTER_ERROR_SECONDS = 30


class ModelServer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._deployed: dict[str, dict] = {}
        self._assessments: dict[str, tuple[tuple, dict]] = {}

    def reload(self, machine_id: str) -> dict:
        deployed = {registry.FAILURE_MODEL: None, registry.ANOMALY_MODEL: None, "loaded_at": time.time(), "error": None}
        try:
            for kind in (registry.FAILURE_MODEL, registry.ANOMALY_MODEL):
                version = registry.version_in_stage(machine_id, kind, registry.PRODUCTION)
                if version is not None:
                    deployed[kind] = {
                        "version": str(version.version),
                        "run_id": version.run_id,
                        "model": registry.load_version(machine_id, kind, version.version),
                    }
        except Exception as exc:
            logger.warning("Could not load production models for %s: %s", machine_id, exc)
            deployed["error"] = f"Model registry unavailable: {exc}"
        with self._lock:
            self._deployed[machine_id] = deployed
            self._assessments.pop(machine_id, None)
        return self.describe(machine_id)

    def describe(self, machine_id: str) -> dict:
        deployed = self._get(machine_id)
        return {
            "failure_model_version": _version(deployed[registry.FAILURE_MODEL]),
            "anomaly_model_version": _version(deployed[registry.ANOMALY_MODEL]),
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(deployed["loaded_at"])),
            "error": deployed["error"],
        }

    def model(self, machine_id: str, kind: str) -> dict | None:
        return self._get(machine_id)[kind]

    def score(self, machine_id: str, frame: pd.DataFrame) -> dict:
        """Vectorised scores for a frame of readings; entries are None when that model isn't deployed."""
        deployed = self._get(machine_id)
        x_data = frame[FEATURE_COLUMNS]
        scores = {"failure_probability": None, "failure_predicted": None, "anomaly_score": None, "anomaly_detected": None}
        failure = deployed[registry.FAILURE_MODEL]
        if failure is not None:
            probabilities = failure["model"].predict_proba(x_data)[:, 1]
            scores["failure_probability"] = probabilities
            scores["failure_predicted"] = probabilities >= get_settings().failure_probability_threshold
        anomaly = deployed[registry.ANOMALY_MODEL]
        if anomaly is not None:
            scores["anomaly_score"] = -anomaly["model"].decision_function(x_data)
            scores["anomaly_detected"] = anomaly["model"].predict(x_data) == -1
        return scores

    def assess_latest(self, machine_id: str) -> dict:
        reading = readings.latest_reading(machine_id)
        if reading is None:
            return {"machine_id": machine_id, "condition": "no_data", "timestamp": None, "features": {}}

        deployed = self._get(machine_id)
        key = (reading["timestamp"], _version(deployed[registry.FAILURE_MODEL]), _version(deployed[registry.ANOMALY_MODEL]))
        with self._lock:
            cached = self._assessments.get(machine_id)
        if cached and cached[0] == key:
            return cached[1]

        scores = self.score(machine_id, pd.DataFrame([reading]))
        first = {name: (None if values is None else values[0].item()) for name, values in scores.items()}
        result = {
            "machine_id": machine_id,
            "timestamp": iso(reading["timestamp"]),
            "features": {column: reading[column] for column in FEATURE_COLUMNS},
            "failure_model_version": key[1],
            "anomaly_model_version": key[2],
            **first,
            "condition": _condition(first, deployed),
            "registry_error": deployed["error"],
        }
        with self._lock:
            self._assessments[machine_id] = (key, result)
        return result

    def explain_latest(self, machine_id: str) -> dict:
        import shap

        reading = readings.latest_reading(machine_id)
        if reading is None:
            return {"machine_id": machine_id, "error": "No readings are available for this machine."}
        entry = self.model(machine_id, registry.FAILURE_MODEL)
        if entry is None:
            return {
                "machine_id": machine_id,
                "error": "No supervised failure model is in Production for this machine, so there is no prediction to explain.",
            }

        model = entry["model"]
        with self._lock:
            explainer = entry.get("explainer")
            if explainer is None:
                explainer = entry["explainer"] = shap.TreeExplainer(model)

        x_data = pd.DataFrame([reading])[FEATURE_COLUMNS]
        values = explainer.shap_values(x_data)
        contributions = np.asarray(values[1] if isinstance(values, list) else values, dtype=float)
        if contributions.ndim == 3:
            contributions = contributions[:, :, 1]
        base_value = float(np.asarray(explainer.expected_value, dtype=float).ravel()[-1])

        features = sorted(
            (
                {
                    "feature": column,
                    "label": FEATURE_META[column]["label"],
                    "unit": FEATURE_META[column]["unit"],
                    "value": reading[column],
                    "shap_value": float(value),
                    "pushes_toward": "failure" if value > 0 else "healthy",
                }
                for column, value in zip(FEATURE_COLUMNS, contributions[0])
            ),
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )
        return {
            "machine_id": machine_id,
            "timestamp": iso(reading["timestamp"]),
            "model_version": entry["version"],
            "model_type": type(model).__name__,
            "failure_probability": float(model.predict_proba(x_data)[0, 1]),
            "shap_output_space": "probability" if type(model).__module__.startswith("sklearn") else "log-odds",
            "base_value": base_value,
            "contributions": features,
        }

    def _get(self, machine_id: str) -> dict:
        with self._lock:
            deployed = self._deployed.get(machine_id)
        stale_error = deployed is not None and deployed["error"] and time.time() - deployed["loaded_at"] > RETRY_AFTER_ERROR_SECONDS
        if deployed is None or stale_error:
            self.reload(machine_id)
            with self._lock:
                deployed = self._deployed[machine_id]
        return deployed


def _version(entry: dict | None) -> str | None:
    return entry["version"] if entry else None


def _condition(scores: dict, deployed: dict) -> str:
    if deployed[registry.FAILURE_MODEL] is None and deployed[registry.ANOMALY_MODEL] is None:
        return "no_models"
    failure = bool(scores["failure_predicted"])
    anomaly = bool(scores["anomaly_detected"])
    if failure and anomaly:
        return "failure_risk_and_anomaly"
    if anomaly:
        return "anomaly"
    if failure:
        return "failure_risk"
    return "healthy"


model_server = ModelServer()

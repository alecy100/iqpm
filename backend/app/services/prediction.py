from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import ModelDeployment, PredictionLog, SensorReading
from app.services.features import FEATURE_COLUMNS, reading_to_features

import shap


def machines(db: Session) -> list[str]:
    rows = db.query(SensorReading.machine_id).distinct().order_by(SensorReading.machine_id.asc()).all()
    return [row[0] for row in rows]


def latest_snapshot(db: Session, machine_id: str) -> dict | None:
    reading = _latest_reading(db, machine_id)
    if reading is None:
        return None
    prediction = predict_for_reading(db, reading)
    return {
        "machine_id": machine_id,
        "timestamp": reading.timestamp,
        "features": reading_to_features(reading),
        "status": "online",
        **prediction,
    }


def fleet_risk(db: Session) -> list[dict]:
    return sorted(
        [snapshot for machine_id in machines(db) if (snapshot := latest_snapshot(db, machine_id))],
        key=lambda item: item["risk_score"],
        reverse=True,
    )


def history(db: Session, machine_id: str, limit: int = 120) -> list[dict]:
    predictions = (
        db.query(PredictionLog)
        .filter(PredictionLog.machine_id == machine_id)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": item.id,
            "timestamp": item.timestamp,
            "risk_score": item.risk_score,
            "anomaly_score": item.anomaly_score,
            "health": item.health,
            "model_version": item.model_version,
        }
        for item in reversed(predictions)
    ]


def explain_prediction(db: Session, prediction_id: int) -> dict | None:
    prediction = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).one_or_none()
    if prediction is None:
        return None
    return {
        "prediction_id": prediction.id,
        "machine_id": prediction.machine_id,
        "timestamp": prediction.timestamp,
        "risk_score": prediction.risk_score,
        "top_features": prediction.shap_explanation.get("top_features", []),
        "model_version": prediction.model_version,
    }


def predict_for_reading(db: Session, reading: SensorReading) -> dict:
    deployment = (
        db.query(ModelDeployment)
        .filter(ModelDeployment.machine_id == reading.machine_id, ModelDeployment.stage == "Production")
        .one_or_none()
    )
    model_bundle = _load_bundle(deployment.artifact_path if deployment else None)
    x_frame = pd.DataFrame([reading_to_features(reading)], columns=FEATURE_COLUMNS)
    risk, explanation, version = _supervised_score(model_bundle, x_frame, reading)
    anomaly_score = _anomaly_score(reading.machine_id, x_frame)
    health = max(0.0, min(1.0, 1.0 - risk))
    log = PredictionLog(
        machine_id=reading.machine_id,
        reading_id=reading.id,
        timestamp=reading.timestamp,
        risk_score=risk,
        anomaly_score=anomaly_score,
        health=health,
        shap_explanation=explanation,
        model_version=version,
    )
    db.add(log)
    db.commit()
    return {
        "risk_score": risk,
        "anomaly_score": anomaly_score,
        "health": health,
        "prediction_id": log.id,
        "model_version": version,
        "top_features": explanation.get("top_features", []),
    }


def _latest_reading(db: Session, machine_id: str) -> SensorReading | None:
    return (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )


def _load_bundle(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _supervised_score(model_bundle: dict | None, x_frame: pd.DataFrame, reading: SensorReading) -> tuple[float, dict, str]:
    if model_bundle:
        model = model_bundle["model"]
        if hasattr(model, "predict_proba"):
            risk = float(model.predict_proba(x_frame)[0][-1])
        else:
            risk = float(model.predict(x_frame)[0])
        explanation = _tree_explanation(model, x_frame) or _heuristic_explanation(reading, risk)
        return risk, explanation, model_bundle.get("trained_at", "production")
    risk = _heuristic_risk(reading)
    return risk, _heuristic_explanation(reading, risk), "heuristic"


def _tree_explanation(model, x_frame: pd.DataFrame) -> dict | None:
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_frame)
        row_values = values[-1][0] if isinstance(values, list) else values[0]
        pairs = sorted(
            zip(FEATURE_COLUMNS, [float(value) for value in row_values], strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return {"top_features": [{"feature": name, "impact": value} for name, value in pairs[:5]]}
    except Exception:
        return None


def _anomaly_score(machine_id: str, x_frame: pd.DataFrame) -> float:
    path = Path(get_settings().model_dir) / f"{machine_id}-isolation-forest.joblib"
    if not path.exists():
        return 0.0
    try:
        bundle = joblib.load(path)
        raw_score = float(bundle["model"].decision_function(x_frame)[0])
        return max(0.0, min(1.0, 0.5 - raw_score))
    except Exception:
        return 0.0


def _heuristic_risk(reading: SensorReading) -> float:
    torque = (reading.torque_nm or 0) / 80
    wear = (reading.tool_wear_min or 0) / 260
    speed_penalty = max(0.0, (1800 - (reading.rotational_speed_rpm or 1800)) / 900)
    temp_gap = max(0.0, ((reading.process_temp_k or 300) - (reading.air_temp_k or 300)) / 20)
    return max(0.02, min(0.98, 0.18 * torque + 0.42 * wear + 0.22 * speed_penalty + 0.18 * temp_gap))


def _heuristic_explanation(reading: SensorReading, risk: float) -> dict:
    features = {
        "tool_wear_min": (reading.tool_wear_min or 0) / 260,
        "torque_nm": (reading.torque_nm or 0) / 80,
        "rotational_speed_rpm": max(0.0, (1800 - (reading.rotational_speed_rpm or 1800)) / 900),
        "process_temp_k": (reading.process_temp_k or 300) / 340,
        "air_temp_k": (reading.air_temp_k or 300) / 330,
    }
    top = sorted(features.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "basis": "heuristic fallback" if risk else "model",
        "top_features": [{"feature": name, "impact": float(value)} for name, value in top],
    }

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import ExperimentRun, FailureLabel, ModelDeployment, SensorReading
from app.services.features import FEATURE_COLUMNS


def train_models_for_machine(db: Session, machine_id: str) -> dict:
    settings = get_settings()
    sensor_frame = _sensor_frame(db, machine_id)
    if sensor_frame.empty:
        return {
            "machine_id": machine_id,
            "status": "skipped",
            "message": "No streamed sensor rows are available yet.",
            "metrics": {},
            "promoted": False,
        }

    isolation_path = _train_isolation_forest(machine_id, sensor_frame[FEATURE_COLUMNS])
    frame = _training_frame(db, machine_id)
    result = {
        "machine_id": machine_id,
        "status": "skipped",
        "message": "Isolation Forest updated; supervised model skipped until enough revealed labels exist.",
        "metrics": {"isolation_forest_artifact": isolation_path},
        "promoted": False,
    }

    labeled = frame[frame["keep_for_training"]].dropna(subset=["label_horizon"]) if not frame.empty else frame
    if len(labeled) < settings.min_supervised_rows:
        result["metrics"]["revealed_training_rows"] = int(len(labeled))
        result["metrics"]["minimum_required_rows"] = settings.min_supervised_rows
        return result

    train_frame, validation_frame = _time_split(labeled)
    if validation_frame["label_horizon"].nunique() < 2 or train_frame["label_horizon"].nunique() < 2:
        result["message"] = "Isolation Forest updated; supervised model needs both classes in train and validation windows."
        result["metrics"]["revealed_training_rows"] = int(len(labeled))
        return result

    model = _fit_classifier(train_frame[FEATURE_COLUMNS], train_frame["label_horizon"])
    metrics = _evaluate(model, validation_frame[FEATURE_COLUMNS], validation_frame["label_horizon"])
    run_id = f"{machine_id}-{uuid4().hex[:10]}"
    model_path = _save_model(machine_id, run_id, model, metrics)
    _log_experiment(db, machine_id, run_id, "flaml_or_rf", model_path, metrics)
    _log_mlflow(machine_id, run_id, metrics, model_path)

    promoted = _passes_deployment_gate(db, machine_id, model, validation_frame, metrics)
    _upsert_deployment(db, machine_id, "Staging", "flaml_or_rf", run_id, metrics, model_path)
    if promoted:
        _upsert_deployment(db, machine_id, "Production", "flaml_or_rf", run_id, metrics, model_path)

    return {
        "machine_id": machine_id,
        "status": "finished",
        "message": "Supervised model trained and promoted." if promoted else "Supervised model trained but not promoted.",
        "metrics": metrics | {"isolation_forest_artifact": isolation_path},
        "run_id": run_id,
        "promoted": promoted,
    }


def leaderboard(db: Session, machine_id: str | None = None) -> list[dict]:
    query = db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc())
    if machine_id:
        query = query.filter(ExperimentRun.machine_id == machine_id)
    return [
        {
            "run_id": run.run_id,
            "machine_id": run.machine_id,
            "model_name": run.model_name,
            "status": run.status,
            "metrics": run.metrics,
            "created_at": run.created_at,
        }
        for run in query.limit(50).all()
    ]


def deployment_status(db: Session, machine_id: str) -> list[dict]:
    deployments = (
        db.query(ModelDeployment)
        .filter(ModelDeployment.machine_id == machine_id)
        .order_by(ModelDeployment.promoted_at.desc())
        .all()
    )
    return [
        {
            "machine_id": item.machine_id,
            "stage": item.stage,
            "model_name": item.model_name,
            "model_version": item.model_version,
            "metrics": item.metrics,
            "artifact_path": item.artifact_path,
            "promoted_at": item.promoted_at,
        }
        for item in deployments
    ]


def promote_staging_to_production(db: Session, machine_id: str) -> dict:
    staging = (
        db.query(ModelDeployment)
        .filter(ModelDeployment.machine_id == machine_id, ModelDeployment.stage == "Staging")
        .one_or_none()
    )
    if staging is None:
        return {"status": "missing", "message": "No staging model exists for this machine."}
    _upsert_deployment(
        db,
        machine_id,
        "Production",
        staging.model_name,
        staging.model_version,
        staging.metrics,
        staging.artifact_path,
    )
    return {"status": "promoted", "message": f"{staging.model_version} is now Production."}


def _sensor_frame(db: Session, machine_id: str) -> pd.DataFrame:
    latest = (
        db.query(SensorReading.timestamp)
        .filter(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    if latest is None:
        return pd.DataFrame()

    cutoff = latest[0] - timedelta(days=get_settings().rolling_window_days)
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == machine_id, SensorReading.timestamp >= cutoff)
        .order_by(SensorReading.timestamp.asc())
        .all()
    )
    return pd.DataFrame(
        [
            {
                "timestamp": reading.timestamp,
                **{column: getattr(reading, column) for column in FEATURE_COLUMNS},
            }
            for reading in rows
        ]
    )


def _training_frame(db: Session, machine_id: str) -> pd.DataFrame:
    latest = (
        db.query(SensorReading.timestamp)
        .filter(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    if latest is None:
        return pd.DataFrame()

    cutoff = latest[0] - timedelta(days=get_settings().rolling_window_days)
    rows = (
        db.query(SensorReading, FailureLabel)
        .join(
            FailureLabel,
            (FailureLabel.machine_id == SensorReading.machine_id)
            & (FailureLabel.timestamp == SensorReading.timestamp),
        )
        .filter(SensorReading.machine_id == machine_id, SensorReading.timestamp >= cutoff)
        .order_by(SensorReading.timestamp.asc())
        .all()
    )
    data = []
    for reading, label in rows:
        data.append(
            {
                "timestamp": reading.timestamp,
                **{column: getattr(reading, column) for column in FEATURE_COLUMNS},
                "label_horizon": label.label_horizon,
                "keep_for_training": label.keep_for_training,
            }
        )
    return pd.DataFrame(data)


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = max(1, int(len(frame) * 0.8))
    return frame.iloc[:split_index], frame.iloc[split_index:]


def _fit_classifier(x_train: pd.DataFrame, y_train: pd.Series):
    try:
        from flaml import AutoML

        model = AutoML()
        model.fit(
            X_train=x_train,
            y_train=y_train,
            task="classification",
            metric="f1",
            time_budget=30,
            estimator_list=["lgbm", "xgboost", "rf"],
            split_type="time",
            verbose=0,
        )
        return model
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return model.fit(x_train, y_train)


def _evaluate(model, x_val: pd.DataFrame, y_val: pd.Series) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

    predictions = model.predict(x_val)
    probabilities = _positive_probabilities(model, x_val)
    return {
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "validation_rows": int(len(y_val)),
    }


def _positive_probabilities(model, x_data: pd.DataFrame) -> list[float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_data)
        return [float(row[-1]) for row in probabilities]
    return [float(value) for value in model.predict(x_data)]


def _passes_deployment_gate(db: Session, machine_id: str, model, validation_frame: pd.DataFrame, metrics: dict) -> bool:
    if metrics["f1"] < get_settings().min_f1_for_promotion:
        return False
    production = (
        db.query(ModelDeployment)
        .filter(ModelDeployment.machine_id == machine_id, ModelDeployment.stage == "Production")
        .one_or_none()
    )
    if production is None or not production.artifact_path:
        return True
    try:
        champion = joblib.load(production.artifact_path)
        champion_metrics = _evaluate(
            champion["model"],
            validation_frame[FEATURE_COLUMNS],
            validation_frame["label_horizon"],
        )
    except Exception:
        return True
    return metrics["f1"] > champion_metrics.get("f1", 0)


def _save_model(machine_id: str, run_id: str, model, metrics: dict) -> str:
    model_dir = Path(get_settings().model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{machine_id}-{run_id}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "trained_at": datetime.utcnow().isoformat(),
        },
        path,
    )
    return str(path)


def _train_isolation_forest(machine_id: str, x_data: pd.DataFrame) -> str:
    from sklearn.ensemble import IsolationForest

    model_dir = Path(get_settings().model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{machine_id}-isolation-forest.joblib"
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(x_data)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, path)
    return str(path)


def _log_experiment(
    db: Session,
    machine_id: str,
    run_id: str,
    model_name: str,
    model_path: str,
    metrics: dict,
) -> None:
    db.add(
        ExperimentRun(
            machine_id=machine_id,
            run_id=run_id,
            model_name=model_name,
            status="finished",
            metrics=metrics,
            params={"split": "time", "target": "label_horizon", "features": FEATURE_COLUMNS},
            artifact_path=model_path,
        )
    )
    db.commit()


def _upsert_deployment(
    db: Session,
    machine_id: str,
    stage: str,
    model_name: str,
    model_version: str,
    metrics: dict,
    artifact_path: str | None,
) -> None:
    deployment = (
        db.query(ModelDeployment)
        .filter(ModelDeployment.machine_id == machine_id, ModelDeployment.stage == stage)
        .one_or_none()
    )
    if deployment is None:
        deployment = ModelDeployment(machine_id=machine_id, stage=stage)
        db.add(deployment)
    deployment.model_name = model_name
    deployment.model_version = model_version
    deployment.metrics = metrics
    deployment.artifact_path = artifact_path
    deployment.promoted_at = datetime.utcnow()
    db.commit()


def _log_mlflow(machine_id: str, run_id: str, metrics: dict, model_path: str) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        mlflow.set_experiment(f"iqPM-{machine_id}")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({"target": "label_horizon", "split": "time"})
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
            mlflow.log_artifact(model_path)
    except Exception:
        return

"""Model fitting and MLflow logging. Registry stage changes happen in the agent graph nodes, not here."""

import json
import logging
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from app.config import get_settings
from app.services import readings, registry
from app.services.features import FEATURE_COLUMNS


logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    pass


_active_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
# MLflow's fluent API tracks the active run globally, so runs are logged one at a time.
_mlflow_lock = threading.Lock()


@contextmanager
def training_job(machine_id: str, kind: str):
    with _jobs_lock:
        if machine_id in _active_jobs:
            raise TrainingError(f"A training job is already running for {machine_id}.")
        _active_jobs[machine_id] = {"kind": kind, "started_at": datetime.utcnow().isoformat()}
    try:
        yield
    finally:
        with _jobs_lock:
            _active_jobs.pop(machine_id, None)


def active_job(machine_id: str) -> dict | None:
    with _jobs_lock:
        job = _active_jobs.get(machine_id)
        return dict(job) if job else None


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * 0.8)
    return frame.iloc[:split_index], frame.iloc[split_index:]


def fit_anomaly_model(machine_id: str, cutoff: datetime, trigger: str) -> dict:
    settings = get_settings()
    frame = readings.readings_between(machine_id, readings.training_window_start(cutoff), cutoff)
    if frame.empty:
        raise TrainingError(f"No sensor readings for {machine_id} inside the training window.")

    x_data = frame[FEATURE_COLUMNS]
    params = {"n_estimators": 200, "contamination": settings.anomaly_contamination, "random_state": 42}
    model = IsolationForest(**params).fit(x_data)
    metrics = {
        "training_rows": float(len(x_data)),
        "training_anomaly_rate": float((model.predict(x_data) == -1).mean()),
    }

    with _mlflow_lock, mlflow.start_run(
        experiment_id=registry.experiment_id(machine_id),
        run_name=f"anomaly-{cutoff:%Y%m%d-%H%M}",
        tags=_run_tags(machine_id, cutoff, trigger, "anomaly"),
    ) as run:
        mlflow.log_params({"algorithm": "isolation_forest", "window_days": settings.rolling_window_days, **params})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=x_data.head(3))
    return {"run_id": run.info.run_id, "metrics": metrics}


def fit_failure_model(machine_id: str, cutoff: datetime, trigger: str) -> dict:
    settings = get_settings()
    frame = readings.labeled_frame(machine_id, cutoff)
    train, validation = time_split(frame)
    for name, part in (("training", train), ("validation", validation)):
        if part.empty or part["label_horizon"].nunique() < 2:
            raise TrainingError(
                f"The {name} slice ({len(part)} rows) does not contain both failure-horizon classes, "
                "so a supervised model can't be trained and validated on this window yet."
            )

    x_train, y_train = train[FEATURE_COLUMNS], train["label_horizon"].astype(int)
    x_val, y_val = validation[FEATURE_COLUMNS], validation["label_horizon"].astype(int)

    with _mlflow_lock, mlflow.start_run(
        experiment_id=registry.experiment_id(machine_id),
        run_name=f"retrain-{cutoff:%Y%m%d-%H%M}",
        tags=_run_tags(machine_id, cutoff, trigger, "supervised"),
    ) as run:
        estimator, learner, best_config, n_trials = _fit_automl(x_train, y_train)
        metrics = evaluate(estimator, x_val, y_val)
        mlflow.log_params(
            {
                "learner": learner,
                "n_trials": n_trials,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "time_budget_s": settings.automl_time_budget_seconds,
                "target": "label_horizon",
                "split": "time 80/20",
                **{f"best.{key}": value for key, value in best_config.items()},
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(estimator, artifact_path="model", input_example=x_val.head(3))

    return {
        "run_id": run.info.run_id,
        "learner": learner,
        "n_trials": n_trials,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "metrics": metrics,
    }


def evaluate(model, x_data: pd.DataFrame, y_true: pd.Series) -> dict:
    probabilities = model.predict_proba(x_data)[:, 1]
    predictions = (probabilities >= get_settings().failure_probability_threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "validation_rows": float(len(y_true)),
    }


def get_current_production_metrics(machine_id: str, snapshot_cutoff: str) -> dict | None:
    """Score the current Production model on the challenger's validation slice. None if nothing is in Production."""
    version = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    if version is None:
        return None
    model = registry.load_version(machine_id, registry.FAILURE_MODEL, version.version)
    _, validation = time_split(readings.labeled_frame(machine_id, datetime.fromisoformat(snapshot_cutoff)))
    metrics = evaluate(model, validation[FEATURE_COLUMNS], validation["label_horizon"].astype(int))
    return {**metrics, "model_version": str(version.version)}


def _fit_automl(x_train: pd.DataFrame, y_train: pd.Series):
    settings = get_settings()
    try:
        from flaml import AutoML

        automl = AutoML()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            log_path = Path(tmp) / "flaml_trials.jsonl"
            automl.fit(
                X_train=x_train,
                y_train=y_train,
                task="classification",
                metric="f1",
                time_budget=settings.automl_time_budget_seconds,
                estimator_list=["lgbm", "xgboost", "rf"],
                split_type="time",
                log_file_name=str(log_path),
                mlflow_logging=False,
                seed=42,
                verbose=0,
            )
            n_trials = _log_trials(log_path)
        # Log the fitted underlying estimator (LightGBM / XGBoost / sklearn RF) so SHAP can explain it directly.
        return automl.model.estimator, automl.best_estimator, dict(automl.best_config or {}), n_trials
    except Exception as exc:
        logger.exception("FLAML AutoML failed; falling back to a random forest")
        mlflow.set_tag("automl_error", str(exc)[:500])
        model = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ).fit(x_train, y_train)
        return model, "rf_fallback", {}, 0


def _log_trials(log_path: Path) -> int:
    """Log every FLAML trial as a nested MLflow run under the active retrain run."""
    if not log_path.exists():
        return 0
    count = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "learner" not in record:
            continue
        learner = record["learner"]
        with mlflow.start_run(
            run_name=f"trial-{record.get('record_id', count)}-{learner}",
            nested=True,
            tags={"kind": "trial", "learner": learner},
        ):
            mlflow.log_params({"learner": learner, **(record.get("config") or {})})
            metrics = {
                "trial_time_s": record.get("trial_time"),
                "wall_clock_time_s": record.get("wall_clock_time"),
                "sample_size": record.get("sample_size"),
            }
            loss = record.get("validation_loss")
            if isinstance(loss, (int, float)):
                metrics["validation_loss"] = loss
                metrics["validation_f1"] = 1 - loss
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
        count += 1
    return count


def _run_tags(machine_id: str, cutoff: datetime, trigger: str, kind: str) -> dict:
    return {"kind": kind, "machine_id": machine_id, "snapshot_cutoff": cutoff.isoformat(), "trigger": trigger}

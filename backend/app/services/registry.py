"""MLflow tracking + model registry access. MLflow is the source of truth for model versions and stages."""

import os
import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from app.config import get_settings


warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

FAILURE_MODEL = "failure"
ANOMALY_MODEL = "anomaly"
STAGING = "Staging"
PRODUCTION = "Production"
ARCHIVED = "Archived"

REPORTED_METRICS = ("f1", "precision", "recall", "pr_auc", "validation_rows", "training_rows", "training_anomaly_rate")

_configured = False


def _setup() -> None:
    global _configured
    if not _configured:
        # Fail fast when the tracking server is down instead of retrying for minutes.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        _configured = True


def client() -> MlflowClient:
    _setup()
    return MlflowClient()


def registered_name(machine_id: str, kind: str) -> str:
    return f"iqpm-{machine_id}-{kind}"


def experiment_name(machine_id: str) -> str:
    return f"iqpm-{machine_id}"


def experiment_id(machine_id: str) -> str:
    _setup()
    experiment = mlflow.get_experiment_by_name(experiment_name(machine_id))
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name(machine_id))


def model_versions(machine_id: str, kind: str) -> list:
    """All versions of a registered model, newest first. Empty if the model was never registered."""
    try:
        versions = client().search_model_versions(f"name='{registered_name(machine_id, kind)}'")
    except MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            return []
        raise
    return sorted(versions, key=lambda item: int(item.version), reverse=True)


def version_in_stage(machine_id: str, kind: str, stage: str):
    return next((item for item in model_versions(machine_id, kind) if item.current_stage == stage), None)


def get_version(machine_id: str, kind: str, version: str):
    try:
        return client().get_model_version(registered_name(machine_id, kind), str(version))
    except MlflowException:
        return None


def register_run_model(run_id: str, machine_id: str, kind: str, tags: dict | None = None) -> str:
    _setup()
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        registered_name(machine_id, kind),
        tags={key: str(value) for key, value in (tags or {}).items()},
    )
    return str(result.version)


def transition(machine_id: str, kind: str, version: str, stage: str) -> None:
    """Move a version into `stage`, archiving whatever currently occupies that stage."""
    client().transition_model_version_stage(
        registered_name(machine_id, kind),
        str(version),
        stage,
        archive_existing_versions=True,
    )


def set_version_tag(machine_id: str, kind: str, version: str, key: str, value) -> None:
    client().set_model_version_tag(registered_name(machine_id, kind), str(version), key, str(value))


def load_version(machine_id: str, kind: str, version: str):
    _setup()
    return mlflow.sklearn.load_model(f"models:/{registered_name(machine_id, kind)}/{version}")


def run_summary(run_id: str) -> dict:
    run = client().get_run(run_id)
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "started_at": _from_millis(run.info.start_time),
        "metrics": {key: value for key, value in run.data.metrics.items() if key in REPORTED_METRICS},
        "params": {key: value for key, value in run.data.params.items() if not key.startswith("best.")},
        "tags": {key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.")},
    }


def describe_version(version) -> dict:
    return {
        "version": str(version.version),
        "stage": version.current_stage,
        "run_id": version.run_id,
        "registered_at": _from_millis(version.creation_timestamp),
        "tags": dict(version.tags or {}),
        "metrics": run_summary(version.run_id)["metrics"] if version.run_id else {},
    }


def search_runs(machine_id: str, kind: str, limit: int = 20, order_by: str = "attributes.start_time DESC") -> list[dict]:
    _setup()
    experiment = mlflow.get_experiment_by_name(experiment_name(machine_id))
    if experiment is None:
        return []
    runs = client().search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.kind = '{kind}'",
        order_by=[order_by],
        max_results=limit,
    )
    return [run_summary(run.info.run_id) for run in runs]


def _from_millis(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.utcfromtimestamp(value / 1000).isoformat()

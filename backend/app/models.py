from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    __table_args__ = (
        UniqueConstraint("machine_id", "timestamp", name="uq_sensor_machine_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    air_temp_k: Mapped[float | None] = mapped_column(Float)
    process_temp_k: Mapped[float | None] = mapped_column(Float)
    rotational_speed_rpm: Mapped[float | None] = mapped_column(Float)
    torque_nm: Mapped[float | None] = mapped_column(Float)
    tool_wear_min: Mapped[float | None] = mapped_column(Float)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    label: Mapped["FailureLabel | None"] = relationship(
        "FailureLabel",
        back_populates="reading",
        uselist=False,
        cascade="all, delete-orphan",
    )


class FailureLabel(Base):
    __tablename__ = "failure_labels"
    __table_args__ = (
        UniqueConstraint("machine_id", "timestamp", name="uq_label_machine_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"), index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    machine_failure: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(30))
    label_horizon: Mapped[int] = mapped_column(Integer)
    keep_for_training: Mapped[bool] = mapped_column(Boolean, default=True)
    revealed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    reading: Mapped[SensorReading] = relationship("SensorReading", back_populates="label")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_readings.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    health: Mapped[float] = mapped_column(Float)
    shap_explanation: Mapped[dict] = mapped_column(JSON, default=dict)
    model_version: Mapped[str] = mapped_column(String(120), default="heuristic")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    run_id: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), default="finished")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class ModelDeployment(Base):
    __tablename__ = "model_deployments"
    __table_args__ = (
        UniqueConstraint("machine_id", "stage", name="uq_deploy_machine_stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    machine_id: Mapped[str] = mapped_column(String(80), index=True)
    stage: Mapped[str] = mapped_column(String(30), index=True)
    model_name: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[str] = mapped_column(String(120))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    promoted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

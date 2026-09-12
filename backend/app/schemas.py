from datetime import datetime

from pydantic import BaseModel, Field


class SensorSnapshot(BaseModel):
    machine_id: str
    timestamp: datetime
    features: dict[str, float | None]
    status: str
    risk_score: float
    anomaly_score: float
    health: float
    prediction_id: int | None = None


class StreamStatus(BaseModel):
    running: bool
    rows_streamed: int
    labels_revealed: int
    simulated_time: datetime | None
    pending_labels: int


class TrainingResult(BaseModel):
    machine_id: str
    status: str
    message: str
    metrics: dict = Field(default_factory=dict)
    run_id: str | None = None
    promoted: bool = False


class AgentMessage(BaseModel):
    message: str
    machine_id: str | None = None


class AgentResponse(BaseModel):
    reply: str
    tool: str
    data: dict | list | None = None


class DeploymentResponse(BaseModel):
    machine_id: str
    stage: str
    model_name: str
    model_version: str
    metrics: dict = Field(default_factory=dict)

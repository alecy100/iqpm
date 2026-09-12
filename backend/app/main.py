from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.schemas import AgentMessage, AgentResponse, SensorSnapshot, StreamStatus, TrainingResult
from app.services import agent, analysis, prediction, training
from app.services.streamer import streamer


app = FastAPI(title="iqPM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "iqPM"}


@app.post("/stream/start", response_model=StreamStatus)
async def start_stream() -> dict:
    await streamer.start()
    return streamer.status()


@app.post("/stream/stop", response_model=StreamStatus)
async def stop_stream() -> dict:
    await streamer.stop()
    return streamer.status()


@app.get("/stream/status", response_model=StreamStatus)
def stream_status() -> dict:
    return streamer.status()


@app.get("/machines")
def list_machines(db: Session = Depends(get_db)) -> list[str]:
    return prediction.machines(db)


@app.get("/machines/{machine_id}/current", response_model=SensorSnapshot)
def current_machine(machine_id: str, db: Session = Depends(get_db)) -> dict:
    snapshot = prediction.latest_snapshot(db, machine_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No readings available for this machine.")
    return snapshot


@app.get("/machines/{machine_id}/history")
def machine_history(machine_id: str, limit: int = 120, db: Session = Depends(get_db)) -> list[dict]:
    return prediction.history(db, machine_id, limit)


@app.get("/fleet/risk")
def fleet_risk(db: Session = Depends(get_db)) -> list[dict]:
    return prediction.fleet_risk(db)


@app.get("/analysis/profile")
def profile(machine_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    return analysis.sensor_profile(db, machine_id)


@app.get("/machines/{machine_id}/trend")
def trend(machine_id: str, limit: int = 180, db: Session = Depends(get_db)) -> dict:
    return analysis.trend_chart(db, machine_id, limit)


@app.post("/machines/{machine_id}/predict", response_model=SensorSnapshot)
def predict(machine_id: str, db: Session = Depends(get_db)) -> dict:
    snapshot = prediction.latest_snapshot(db, machine_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No readings available for this machine.")
    return snapshot


@app.post("/machines/{machine_id}/retrain", response_model=TrainingResult)
def retrain(machine_id: str, db: Session = Depends(get_db)) -> dict:
    return training.train_models_for_machine(db, machine_id)


@app.get("/models/leaderboard")
def leaderboard(machine_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    return training.leaderboard(db, machine_id)


@app.get("/machines/{machine_id}/deployment")
def deployment(machine_id: str, db: Session = Depends(get_db)) -> list[dict]:
    return training.deployment_status(db, machine_id)


@app.post("/machines/{machine_id}/deployment/promote")
def promote(machine_id: str, db: Session = Depends(get_db)) -> dict:
    return training.promote_staging_to_production(db, machine_id)


@app.get("/predictions/{prediction_id}/explain")
def explain(prediction_id: int, db: Session = Depends(get_db)) -> dict:
    explanation = prediction.explain_prediction(db, prediction_id)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return explanation


@app.post("/agent/message", response_model=AgentResponse)
def agent_message(payload: AgentMessage, db: Session = Depends(get_db)) -> dict:
    return agent.handle_agent_message(db, payload.message, payload.machine_id)

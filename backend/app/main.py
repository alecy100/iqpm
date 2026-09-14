import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.drift_monitor import drift_monitor
from app.agent.runtime import InvalidThreadIdError, NoPendingApprovalError, ThreadBusyError, runtime
from app.config import get_settings
from app.database import init_db
from app.schemas import AgentMessageIn, DashboardActionIn, ResumeIn
from app.services import deployment, readings
from app.services.features import FEATURE_COLUMNS, FEATURE_META
from app.services.seed import seed_from_csv
from app.services.serialization import to_jsonable
from app.services.serving import model_server
from app.services.streamer import streamer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    await asyncio.to_thread(seed_from_csv)
    runtime.start()
    monitor_task = asyncio.create_task(drift_monitor.run_forever()) if settings.drift_monitor_enabled else None
    if settings.stream_autostart:
        await streamer.start()
    yield
    await streamer.stop()
    if monitor_task is not None:
        monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await monitor_task


app = FastAPI(title="iqPM API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(status: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return handler


app.add_exception_handler(readings.UnknownMachineError, _error(404))
app.add_exception_handler(InvalidThreadIdError, _error(400))
app.add_exception_handler(ThreadBusyError, _error(409))
app.add_exception_handler(NoPendingApprovalError, _error(409))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "iqPM"}


# -- streaming --------------------------------------------------------------------------------


@app.post("/stream/start")
async def start_stream() -> dict:
    await streamer.start()
    return to_jsonable(streamer.status())


@app.post("/stream/stop")
async def stop_stream() -> dict:
    await streamer.stop()
    return to_jsonable(streamer.status())


@app.get("/stream/status")
def stream_status() -> dict:
    return to_jsonable(streamer.status())


# -- machines ---------------------------------------------------------------------------------


@app.get("/machines")
def list_machines() -> list[str]:
    return readings.list_machines()


@app.get("/sensors")
def sensors() -> list[dict]:
    return [{"key": key, **FEATURE_META[key]} for key in FEATURE_COLUMNS]


@app.get("/machines/{machine_id}/current")
def current_machine(machine_id: str) -> dict:
    return to_jsonable(model_server.assess_latest(machine_id))


@app.get("/machines/{machine_id}/readings")
def machine_readings(machine_id: str, limit: int = Query(180, ge=1, le=2000)) -> dict:
    frame = readings.recent_readings(machine_id, limit)
    return {"machine_id": machine_id, "readings": to_jsonable(frame.to_dict("records"))}


@app.get("/machines/{machine_id}/deployment")
def machine_deployment(machine_id: str) -> dict:
    return to_jsonable(deployment.deployment_status(machine_id))


@app.get("/machines/{machine_id}/approvals")
def machine_approvals(machine_id: str) -> list[dict]:
    readings.table_for(machine_id)
    return runtime.open_approvals(machine_id)


@app.post("/machines/{machine_id}/actions/{action}")
def machine_action(machine_id: str, action: Literal["retrain", "promote", "rollback"], payload: DashboardActionIn | None = None) -> dict:
    """Dashboard buttons enter the same LangGraph path as chat requests."""
    readings.table_for(machine_id)
    payload = payload or DashboardActionIn()
    thread_id = runtime.dashboard_action(machine_id, action, payload.thread_id, payload.target_version)
    return runtime.snapshot(thread_id)


@app.get("/models/leaderboard")
def leaderboard(machine_id: str | None = None, limit: int = Query(10, ge=1, le=100)) -> list[dict]:
    return to_jsonable(deployment.leaderboard(machine_id, limit))


@app.post("/drift/check")
async def drift_check(machine_id: str | None = None, force: bool = False) -> list[dict]:
    if machine_id:
        result = [await asyncio.to_thread(drift_monitor.check_machine, machine_id, force)]
    else:
        result = await asyncio.to_thread(drift_monitor.check_all, force)
    return to_jsonable(result)


# -- agent ------------------------------------------------------------------------------------


@app.get("/agent/graph")
def agent_graph() -> dict:
    return {
        "chat": runtime.chat_graph.get_graph().draw_mermaid(),
        "drift": runtime.drift_graph.get_graph().draw_mermaid(),
    }


@app.get("/agent/threads/{thread_id}")
def get_thread(thread_id: str) -> dict:
    return runtime.snapshot(thread_id)


@app.post("/agent/threads/{thread_id}/messages")
def post_message(thread_id: str, payload: AgentMessageIn) -> dict:
    if payload.machine_id:
        readings.table_for(payload.machine_id)
    runtime.send_message(thread_id, payload.message, payload.machine_id)
    return runtime.snapshot(thread_id)


@app.post("/agent/threads/{thread_id}/resume")
def resume_thread(thread_id: str, payload: ResumeIn) -> dict:
    runtime.resume(thread_id, payload.approved)
    return runtime.snapshot(thread_id)

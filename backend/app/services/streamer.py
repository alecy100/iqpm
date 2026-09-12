import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal
from app.models import FailureLabel, SensorReading


@dataclass
class PendingLabel:
    machine_id: str
    timestamp: datetime
    reveal_at: datetime
    machine_failure: int
    state: str
    label_horizon: int
    keep_for_training: bool


class StreamingSimulator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = Lock()
        self.rows_streamed = 0
        self.labels_revealed = 0
        self.simulated_time: datetime | None = None
        self.pending_labels: list[PendingLabel] = []

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    def status(self) -> dict:
        return {
            "running": self.running,
            "rows_streamed": self.rows_streamed,
            "labels_revealed": self.labels_revealed,
            "simulated_time": self.simulated_time,
            "pending_labels": len(self.pending_labels),
        }

    async def _run(self) -> None:
        csv_path = Path(self.settings.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self._stop.is_set():
                    break
                reading_time = datetime.fromisoformat(row["timestamp"])
                self.simulated_time = reading_time
                with SessionLocal() as db:
                    reading = self._insert_sensor_row(db, row, reading_time)
                    if reading is not None:
                        self._queue_label(row, reading_time)
                    self._reveal_due_labels(db, reading_time)
                    db.commit()
                await asyncio.sleep(self.settings.stream_interval_seconds)

    def _insert_sensor_row(self, db: Session, row: dict, reading_time: datetime) -> SensorReading | None:
        reading = SensorReading(
            machine_id=row["machine_id"],
            timestamp=reading_time,
            air_temp_k=_float(row.get("air_temp_k")),
            process_temp_k=_float(row.get("process_temp_k")),
            rotational_speed_rpm=_float(row.get("rotational_speed_rpm")),
            torque_nm=_float(row.get("torque_nm")),
            tool_wear_min=_float(row.get("tool_wear_min")),
        )
        db.add(reading)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            return None
        self.rows_streamed += 1
        return reading

    def _queue_label(self, row: dict, reading_time: datetime) -> None:
        reveal_at = reading_time + timedelta(hours=self.settings.label_reveal_lag_hours)
        self.pending_labels.append(
            PendingLabel(
                machine_id=row["machine_id"],
                timestamp=reading_time,
                reveal_at=reveal_at,
                machine_failure=int(row["machine_failure"]),
                state=row["state"],
                label_horizon=int(row["label_horizon"]),
                keep_for_training=row.get("_keep_for_training", "True") == "True",
            )
        )

    def _reveal_due_labels(self, db: Session, simulated_time: datetime) -> None:
        due = [label for label in self.pending_labels if label.reveal_at <= simulated_time]
        if not due:
            return
        self.pending_labels = [label for label in self.pending_labels if label.reveal_at > simulated_time]
        for label in due:
            reading = (
                db.query(SensorReading)
                .filter(
                    SensorReading.machine_id == label.machine_id,
                    SensorReading.timestamp == label.timestamp,
                )
                .one_or_none()
            )
            if reading is None:
                continue
            db.add(
                FailureLabel(
                    reading_id=reading.id,
                    machine_id=label.machine_id,
                    timestamp=label.timestamp,
                    machine_failure=label.machine_failure,
                    state=label.state,
                    label_horizon=label.label_horizon,
                    keep_for_training=label.keep_for_training,
                    revealed_at=simulated_time,
                )
            )
            self.labels_revealed += 1


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


streamer = StreamingSimulator()

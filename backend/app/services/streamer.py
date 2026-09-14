import asyncio
import logging
from datetime import datetime

from app.config import get_settings
from app.services import readings
from app.services.simulator import MachineSimulator, ReferencePools, load_reference_pools


logger = logging.getLogger(__name__)


class StreamingSimulator:
    """Appends one simulated row per machine per tick. Each machine owns its own RNG and lifecycle."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._pools: ReferencePools | None = None
        self._simulators: dict[str, MachineSimulator] = {}
        self.rows_streamed = 0
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if not self.running:
            return
        self._stop.set()
        await self._task

    def status(self) -> dict:
        times = [sim.timestamp for sim in self._simulators.values()]
        return {
            "running": self.running,
            "rows_streamed": self.rows_streamed,
            "simulated_time": max(times) if times else None,
            "last_error": self.last_error,
        }

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.tick)
                self.last_error = None
            except Exception as exc:  # keep streaming; surface the error in /stream/status
                logger.exception("Streaming tick failed")
                self.last_error = str(exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.stream_interval_seconds)
            except asyncio.TimeoutError:
                pass

    def tick(self) -> None:
        for machine_id in readings.list_machines():
            simulator = self._simulators.get(machine_id) or self._create_simulator(machine_id)
            readings.insert_reading(machine_id, simulator.next_row())
            self.rows_streamed += 1

    def _create_simulator(self, machine_id: str) -> MachineSimulator:
        if self._pools is None:
            self._pools = load_reference_pools(self.settings.ai4i_csv_path)
        last_row = readings.latest_reading(machine_id, include_labels=True)
        simulator = MachineSimulator(
            machine_id=machine_id,
            pools=self._pools,
            step_seconds=self.settings.stream_step_seconds,
            horizon_hours=self.settings.label_horizon_hours,
            last_row=last_row,
        )
        if last_row is None:
            simulator.timestamp = datetime.utcnow()
        self._simulators[machine_id] = simulator
        return simulator


streamer = StreamingSimulator()

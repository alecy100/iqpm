import asyncio
import logging

from app.agent.runtime import runtime
from app.config import get_settings
from app.services import readings, registry, training


logger = logging.getLogger(__name__)


class DriftMonitor:
    """Scheduled drift checks per machine, independent of any open conversation."""

    async def run_forever(self) -> None:
        interval = get_settings().drift_check_interval_seconds
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self.check_all)
            except Exception:
                logger.exception("Drift monitor pass failed")

    def check_all(self, force: bool = False) -> list[dict]:
        return [self.check_machine(machine_id, force) for machine_id in readings.list_machines()]

    def check_machine(self, machine_id: str, force: bool = False) -> dict:
        readings.table_for(machine_id)
        try:
            if training.active_job(machine_id):
                return {"machine_id": machine_id, "skipped": "a training job is running"}
            if runtime.has_open_drift_episode(machine_id):
                return {"machine_id": machine_id, "skipped": "a drift-triggered model is already awaiting approval"}
            # Drift is measured against what the Production model was trained on.
            if not force and registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION) is None:
                return {"machine_id": machine_id, "skipped": "no Production failure model to compare against"}
            return runtime.run_drift_episode(machine_id)
        except Exception as exc:
            logger.exception("Drift check failed for %s", machine_id)
            return {"machine_id": machine_id, "error": f"{type(exc).__name__}: {exc}"}


drift_monitor = DriftMonitor()

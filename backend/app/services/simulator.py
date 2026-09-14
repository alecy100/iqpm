"""Step-by-step port of the lifecycle random walk in generate_synthetic.py.

Each machine repeats HEALTHY -> DEGRADING -> FAILED/REPAIR cycles. Healthy baselines and
failure end-states are sampled from the real AI4I 2020 rows; degradation is an accelerating
walk from the baseline toward the failure state with noise scaled off the real feature spread.

The offline generator derives `label_horizon` by looking ahead over the finished series. The
streamer can't look ahead, so each simulator plans its current and next cycle up front, which
is enough to know when the next failure starts for every row it emits.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.features import FEATURE_COLUMNS


AI4I_COLUMN_MAP = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

# Same fallback stats as generate_synthetic.py, used when the AI4I CSV is not mounted.
FALLBACK_HEALTHY_MEAN = {
    "air_temp_k": 300.0,
    "process_temp_k": 310.0,
    "rotational_speed_rpm": 1540.0,
    "torque_nm": 40.0,
    "tool_wear_min": 100.0,
}
FALLBACK_HEALTHY_STD = {
    "air_temp_k": 2.0,
    "process_temp_k": 1.5,
    "rotational_speed_rpm": 180.0,
    "torque_nm": 10.0,
    "tool_wear_min": 60.0,
}
FALLBACK_FAILURE_MEAN = {
    "air_temp_k": 302.5,
    "process_temp_k": 311.5,
    "rotational_speed_rpm": 1380.0,
    "torque_nm": 58.0,
    "tool_wear_min": 210.0,
}

EXCLUSION_BUFFER_STEPS = 3
REFERENCE_DAYS = 5.0  # generate_synthetic.py sizes lifecycle phases relative to a 5-day run


@dataclass
class ReferencePools:
    healthy: np.ndarray | None
    failed: np.ndarray | None
    noise_scale: dict[str, float]


def load_reference_pools(path: Path | None) -> ReferencePools:
    if path is not None and path.exists():
        frame = pd.read_csv(path, encoding="utf-8-sig").rename(columns=AI4I_COLUMN_MAP)
        healthy = frame[frame["machine_failure"] == 0][FEATURE_COLUMNS]
        failed = frame[frame["machine_failure"] == 1][FEATURE_COLUMNS]
        stds = healthy.std().to_dict()
        return ReferencePools(
            healthy=healthy.to_numpy(dtype=float),
            failed=failed.to_numpy(dtype=float) if len(failed) else None,
            noise_scale={k: max(float(v) * 0.03, 1e-3) for k, v in stds.items()},
        )
    return ReferencePools(
        healthy=None,
        failed=None,
        noise_scale={k: max(v * 0.03, 1e-3) for k, v in FALLBACK_HEALTHY_STD.items()},
    )


@dataclass
class Cycle:
    baseline: dict[str, float]
    target: dict[str, float]
    healthy_len: int
    degrade_len: int
    repair_len: int

    @property
    def failure_start(self) -> int:
        return self.healthy_len + self.degrade_len

    @property
    def length(self) -> int:
        return self.failure_start + self.repair_len


class MachineSimulator:
    def __init__(
        self,
        machine_id: str,
        pools: ReferencePools,
        step_seconds: int,
        horizon_hours: float,
        last_row: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.machine_id = machine_id
        self.pools = pools
        self.step_seconds = step_seconds
        self.rng = rng or np.random.default_rng()
        self.horizon_steps = int(horizon_hours * 3600 / step_seconds)

        total_steps = int(REFERENCE_DAYS * 24 * 3600 / step_seconds)
        self.healthy_bounds = (max(5, int(total_steps * 0.10)), max(10, int(total_steps * 0.25)))
        self.degrade_bounds = (max(3, int(total_steps * 0.05)), max(6, int(total_steps * 0.12)))
        self.repair_bounds = (max(3, int(total_steps * 0.01)), max(6, int(total_steps * 0.04)))

        self.timestamp: datetime = last_row["timestamp"] if last_row else datetime.utcnow()
        self.tool_wear_accum = 0.0
        self.position = 0

        first = self._plan_cycle()
        # Continue from the last stored row instead of jumping to a fresh baseline.
        if last_row and last_row.get("state") in ("healthy", "degrading"):
            first.baseline = {k: float(last_row[k]) for k in FEATURE_COLUMNS}
            self.tool_wear_accum = float(last_row.get("tool_wear_min") or 0.0)
            if last_row["state"] == "degrading":
                first.healthy_len = 0
        self.cycles = deque([first, self._plan_cycle()])

    def next_row(self) -> dict:
        cycle = self.cycles[0]
        if self.position >= cycle.length:
            self.cycles.popleft()
            self.cycles.append(self._plan_cycle())
            self.position = 0
            self.tool_wear_accum = 0.0  # part replaced during repair
            cycle = self.cycles[0]

        pos = self.position
        noise = self.pools.noise_scale
        if pos < cycle.healthy_len:
            state = "healthy"
            point = {k: cycle.baseline[k] + self.rng.normal(0, noise[k]) for k in FEATURE_COLUMNS}
            point["tool_wear_min"] = max(0.0, self.tool_wear_accum)
            self.tool_wear_accum += abs(self.rng.normal(0.5, 0.2))
        elif pos < cycle.failure_start:
            state = "degrading"
            frac = (pos - cycle.healthy_len + 1) / cycle.degrade_len
            drift_frac = frac**1.6  # slow onset, faster near the end
            point = {
                k: cycle.baseline[k] + drift_frac * (cycle.target[k] - cycle.baseline[k]) + self.rng.normal(0, noise[k] * 1.5)
                for k in FEATURE_COLUMNS
                if k != "tool_wear_min"
            }
            point["tool_wear_min"] = max(0.0, self.tool_wear_accum)
            self.tool_wear_accum += abs(self.rng.normal(1.0, 0.4))
        else:
            state = "failed"
            point = {k: cycle.target[k] + self.rng.normal(0, noise[k] * 0.5) for k in FEATURE_COLUMNS}
            point["tool_wear_min"] = cycle.target["tool_wear_min"]

        steps_to_failure = self._steps_to_next_failure(cycle, pos)
        self.position += 1
        self.timestamp = self.timestamp + timedelta(seconds=self.step_seconds)
        return {
            "timestamp": self.timestamp,
            **{k: round(float(v), 3) for k, v in point.items()},
            "machine_failure": int(state == "failed"),
            "state": state,
            "label_horizon": int(1 <= steps_to_failure <= self.horizon_steps),
            "keep_for_training": state != "failed" and steps_to_failure > EXCLUSION_BUFFER_STEPS,
            "source": "stream",
        }

    def _steps_to_next_failure(self, cycle: Cycle, pos: int) -> int:
        if pos < cycle.failure_start:
            return cycle.failure_start - pos
        return (cycle.length - pos) + self.cycles[1].failure_start

    def _plan_cycle(self) -> Cycle:
        return Cycle(
            baseline=self._sample(self.pools.healthy, self._fallback_healthy),
            target=self._sample(self.pools.failed, self._fallback_failure),
            healthy_len=self._randint(*self.healthy_bounds),
            degrade_len=self._randint(*self.degrade_bounds),
            repair_len=self._randint(*self.repair_bounds),
        )

    def _randint(self, low: int, high: int) -> int:
        return int(self.rng.integers(low, high + 1))

    def _sample(self, pool: np.ndarray | None, fallback) -> dict[str, float]:
        if pool is None or len(pool) == 0:
            return fallback()
        row = pool[int(self.rng.integers(0, len(pool)))]
        return {k: float(v) for k, v in zip(FEATURE_COLUMNS, row)}

    def _fallback_healthy(self) -> dict[str, float]:
        return {k: float(self.rng.normal(v, FALLBACK_HEALTHY_STD[k] * 0.3)) for k, v in FALLBACK_HEALTHY_MEAN.items()}

    def _fallback_failure(self) -> dict[str, float]:
        return {k: float(self.rng.normal(v, 3.0)) for k, v in FALLBACK_FAILURE_MEAN.items()}

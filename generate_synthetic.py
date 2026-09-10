"""
generate_synthetic.py

Generates synthetic multivariate time-series sensor data for multiple simulated
machines, seeded from the real AI4I 2020 Predictive Maintenance dataset.

Approach
--------
1. Load AI4I2020.csv (or fall back to built-in approximate stats if not provided).
2. Split AI4I rows into "healthy" (Machine failure == 0) and "failure" (== 1) pools.
3. For each synthetic machine, simulate repeating lifecycles:
      HEALTHY -> DEGRADING (random walk drifting toward a sampled failure profile)
              -> FAILED/REPAIR (flat stretch, label = 1, held roughly constant + noise)
              -> HEALTHY (reset, new baseline)
4. Emit one row per timestamp at a fixed sampling interval, with a raw event label
   (0 = healthy/degrading, 1 = failed/under repair) plus a `state` column for clarity.

Why a random walk toward a sampled real failure row (not a straight line)?
----------------------------------------------------------------------------
AI4I rows are independent snapshots with no time dimension, so there's no real
trajectory to copy. Instead we treat each sampled AI4I "failure" row as a plausible
*end state* and each "healthy" row as a plausible *start state*, then interpolate
between them step by step with:
  - an accelerating drift fraction (slow onset, faster near end -- degradation
    rarely looks linear), and
  - per-step Gaussian noise scaled off the REAL feature spread in the AI4I data
    (so the walk's roughness matches real sensor noise, not an arbitrary guess).

This gives you a physically plausible-looking trajectory grounded in real values
at both endpoints, without claiming to model the true underlying physics.

Output columns
---------------
machine_id, timestamp, air_temp_k, process_temp_k, rotational_speed_rpm,
torque_nm, tool_wear_min, machine_failure, state, label_horizon, _keep_for_training

- `machine_failure`: raw ground-truth event flag. 1 for every row inside a
  contiguous failed/repair span (this is what gives you the "stretches of 1s"
  rather than isolated spikes).
- `state`: healthy / degrading / failed -- useful for filtering/plotting.
- `label_horizon`: derived training label -- "does a failure begin within the
  next --horizon-hours after this row?" This is what your AutoML pipeline
  should actually train on, NOT `machine_failure` directly (training on the
  raw flag would just teach the model to detect a failure that's already
  happening, not predict one that hasn't happened yet).
- `_keep_for_training`: False for rows inside/near a failed span (leakage
  exclusion buffer) -- filter these out before building training windows.

This script is the OFFLINE ORACLE generator only. It does not simulate the
label-reveal lag or the row-by-row streaming into Postgres -- that's the job
of your ingestion/streaming harness, which should read from this CSV but only
progressively reveal `machine_failure`/`label_horizon` to the "live" system
as simulated time advances (see the label-latency discussion in project notes).
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FEATURE_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

AI4I_COLUMN_MAP = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "Type": "type",
}

# Fallback distribution stats if no AI4I CSV is provided (approx. real AI4I stats,
# used only so the script is runnable/demoable before you plug in the real file).
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


def load_ai4i(path):
    df = pd.read_csv(path)
    df = df.rename(columns=AI4I_COLUMN_MAP)
    missing = [c for c in FEATURE_COLS + ["machine_failure"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"AI4I file missing expected columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    healthy = df[df["machine_failure"] == 0][FEATURE_COLS].reset_index(drop=True)
    failed = df[df["machine_failure"] == 1][FEATURE_COLS].reset_index(drop=True)
    if len(failed) == 0:
        raise ValueError("No failure rows (machine_failure == 1) found in AI4I file.")
    return healthy, failed


def sample_healthy_baseline(healthy_pool):
    if healthy_pool is not None and len(healthy_pool) > 0:
        row = healthy_pool.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        return row[FEATURE_COLS].to_dict()
    return {k: float(RNG.normal(v, FALLBACK_HEALTHY_STD[k] * 0.3)) for k, v in FALLBACK_HEALTHY_MEAN.items()}


def sample_failure_target(failed_pool):
    if failed_pool is not None and len(failed_pool) > 0:
        row = failed_pool.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        return row[FEATURE_COLS].to_dict()
    return {k: float(RNG.normal(v, 3.0)) for k, v in FALLBACK_FAILURE_MEAN.items()}


def get_noise_scale(healthy_pool):
    """Per-step noise std, derived from the real feature spread (small fraction of it)."""
    if healthy_pool is not None and len(healthy_pool) > 0:
        stds = healthy_pool[FEATURE_COLS].std().to_dict()
    else:
        stds = FALLBACK_HEALTHY_STD
    return {k: max(float(v) * 0.03, 1e-3) for k, v in stds.items()}


def _make_row(machine_id, start_time, step, interval_seconds, point, label, state):
    ts = start_time + timedelta(seconds=step * interval_seconds)
    row = {"machine_id": machine_id, "timestamp": ts.isoformat()}
    row.update({k: round(float(v), 3) for k, v in point.items()})
    row["machine_failure"] = label
    row["state"] = state
    return row


def simulate_machine(
    machine_id,
    start_time,
    total_steps,
    interval_seconds,
    healthy_pool,
    failed_pool,
    min_healthy_steps,
    max_healthy_steps,
    min_degrade_steps,
    max_degrade_steps,
    min_repair_steps,
    max_repair_steps,
):
    """Simulates one machine's full timeline: HEALTHY -> DEGRADING -> FAILED -> HEALTHY -> ..."""
    noise_scale = get_noise_scale(healthy_pool)
    rows = []
    step = 0
    tool_wear_accum = 0.0

    while step < total_steps:
        # ---- HEALTHY: flat-ish baseline with small noise, tool wear creeps up ----
        healthy_len = int(RNG.integers(min_healthy_steps, max_healthy_steps + 1))
        baseline = sample_healthy_baseline(healthy_pool)
        for _ in range(healthy_len):
            if step >= total_steps:
                break
            point = {k: baseline[k] + RNG.normal(0, noise_scale[k]) for k in FEATURE_COLS}
            point["tool_wear_min"] = max(0.0, tool_wear_accum)
            tool_wear_accum += abs(RNG.normal(0.5, 0.2))
            rows.append(_make_row(machine_id, start_time, step, interval_seconds, point, 0, "healthy"))
            step += 1
        if step >= total_steps:
            break

        # ---- DEGRADING: accelerating random walk toward a sampled real failure row ----
        degrade_len = int(RNG.integers(min_degrade_steps, max_degrade_steps + 1))
        target = sample_failure_target(failed_pool)
        walk_point = dict(baseline)
        for i in range(degrade_len):
            if step >= total_steps:
                break
            frac = (i + 1) / degrade_len
            drift_frac = frac ** 1.6  # concave: slow start, faster near the end
            for k in FEATURE_COLS:
                if k == "tool_wear_min":
                    continue
                drift_target = baseline[k] + drift_frac * (target[k] - baseline[k])
                walk_point[k] = drift_target + RNG.normal(0, noise_scale[k] * 1.5)
            walk_point["tool_wear_min"] = max(0.0, tool_wear_accum)
            tool_wear_accum += abs(RNG.normal(1.0, 0.4))
            rows.append(_make_row(machine_id, start_time, step, interval_seconds, dict(walk_point), 0, "degrading"))
            step += 1
        if step >= total_steps:
            break

        # ---- FAILED / REPAIR: CONTIGUOUS stretch of label == 1 ----
        repair_len = int(RNG.integers(min_repair_steps, max_repair_steps + 1))
        failed_point = {k: target[k] for k in FEATURE_COLS}
        for _ in range(repair_len):
            if step >= total_steps:
                break
            point = {k: failed_point[k] + RNG.normal(0, noise_scale[k] * 0.5) for k in FEATURE_COLS}
            point["tool_wear_min"] = failed_point["tool_wear_min"]
            rows.append(_make_row(machine_id, start_time, step, interval_seconds, point, 1, "failed"))
            step += 1

        tool_wear_accum = 0.0  # part replaced during repair

    return rows


def add_horizon_labels(df, horizon_hours, interval_seconds, exclusion_buffer_steps=3):
    """
    Derives the training-ready label: does a failure span BEGIN within the next
    `horizon_hours` after this row? Also flags rows inside/near a failed span for
    exclusion (label-leakage buffer) -- those rows should not be used as training
    samples at all, since their own features are contaminated by the failure itself.
    """
    horizon_steps = int((horizon_hours * 3600) / interval_seconds)
    out_frames = []
    for machine_id, g in df.groupby("machine_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        state = g["state"].values
        horizon_label = np.zeros(len(g), dtype=int)
        keep = np.ones(len(g), dtype=bool)

        is_failed = state == "failed"
        starts = np.where(is_failed & ~np.r_[False, is_failed[:-1]])[0]

        for s in starts:
            lo = max(0, s - horizon_steps)
            horizon_label[lo:s] = 1
            buf_lo = max(0, s - exclusion_buffer_steps)
            end = s
            while end < len(state) and state[end] == "failed":
                end += 1
            keep[buf_lo:end] = False

        g["label_horizon"] = horizon_label
        g["_keep_for_training"] = keep
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AI4I-seeded time series sensor data.")
    parser.add_argument("--ai4i-path", type=str, default=None,
                         help="Path to real AI4I 2020 CSV. If omitted, uses approximate built-in stats.")
    parser.add_argument("--num-machines", type=int, default=5)
    parser.add_argument("--days", type=float, default=5.0, help="Total simulated days per machine.")
    parser.add_argument("--interval-seconds", type=int, default=30, help="Sampling interval, in seconds.")
    parser.add_argument("--output", type=str, default="synthetic_timeseries.csv")
    parser.add_argument("--horizon-hours", type=float, default=24.0,
                         help="Prediction horizon (hours) for the derived label_horizon column.")
    parser.add_argument("--start-time", type=str, default=None, help="ISO start timestamp; defaults to now.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global RNG
    RNG = np.random.default_rng(args.seed)

    healthy_pool, failed_pool = (None, None)
    if args.ai4i_path:
        healthy_pool, failed_pool = load_ai4i(args.ai4i_path)
        print(f"Loaded AI4I: {len(healthy_pool)} healthy rows, {len(failed_pool)} failure rows.")
    else:
        print("No --ai4i-path provided; using built-in approximate AI4I stats as fallback.")

    total_steps = int((args.days * 24 * 3600) / args.interval_seconds)
    start_time = datetime.fromisoformat(args.start_time) if args.start_time else datetime.now()

    all_rows = []
    for m in range(1, args.num_machines + 1):
        machine_id = f"machine_{m:02d}"
        rows = simulate_machine(
            machine_id=machine_id,
            start_time=start_time,
            total_steps=total_steps,
            interval_seconds=args.interval_seconds,
            healthy_pool=healthy_pool,
            failed_pool=failed_pool,
            min_healthy_steps=max(5, int(total_steps * 0.10)),
            max_healthy_steps=max(10, int(total_steps * 0.25)),
            min_degrade_steps=max(3, int(total_steps * 0.05)),
            max_degrade_steps=max(6, int(total_steps * 0.12)),
            min_repair_steps=max(3, int(total_steps * 0.01)),
            max_repair_steps=max(6, int(total_steps * 0.04)),
        )
        all_rows.extend(rows)
        n_failed = sum(1 for r in rows if r["machine_failure"] == 1)
        print(f"  {machine_id}: {len(rows)} rows, {n_failed} labeled failure rows ({n_failed/len(rows)*100:.2f}%)")

    df = pd.DataFrame(all_rows)
    df = df[["machine_id", "timestamp"] + FEATURE_COLS + ["machine_failure", "state"]]
    df = add_horizon_labels(df, horizon_hours=args.horizon_hours, interval_seconds=args.interval_seconds)

    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nOverall machine_failure positive rate: {df['machine_failure'].mean()*100:.2f}%")
    print(f"Overall label_horizon positive rate:    {df['label_horizon'].mean()*100:.2f}%")
    print(f"Rows excluded from training (leakage buffer): {(~df['_keep_for_training']).sum()}")


if __name__ == "__main__":
    main()
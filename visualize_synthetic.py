"""
visualize_synthetic.py

Interactive visualization for the output of generate_synthetic.py. Plots each
sensor feature over time for a chosen machine, with shaded background regions
showing state (healthy / degrading / failed), so you can visually confirm:

  - the random walk drifting from a healthy baseline toward a failure profile
    during "degrading" spans (shaded yellow)
  - CONTIGUOUS stretches of "failed" (shaded red) rather than isolated spikes
  - the label_horizon column "turning on" before a failure span begins, and
    the exclusion-buffer rows (_keep_for_training == False) around each failure

Uses Plotly (matches the project's chosen viz stack) and writes a single
self-contained interactive HTML file per machine -- open it in a browser,
zoom/pan/hover to inspect exact values at any timestamp.

Usage
-----
    python visualize_synthetic.py --input synthetic_timeseries.csv
    python visualize_synthetic.py --input synthetic_timeseries.csv --machine machine_02
    python visualize_synthetic.py --input synthetic_timeseries.csv --all-machines
"""

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FEATURE_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

STATE_COLORS = {
    "healthy": None,          # no shading
    "degrading": "rgba(255, 193, 7, 0.18)",   # amber
    "failed": "rgba(220, 53, 69, 0.22)",      # red
}

FEATURE_LABELS = {
    "air_temp_k": "Air Temp (K)",
    "process_temp_k": "Process Temp (K)",
    "rotational_speed_rpm": "Rotational Speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool Wear (min)",
}


def find_state_spans(df):
    """Returns list of (state, start_ts, end_ts) contiguous spans, for shading."""
    spans = []
    states = df["state"].values
    timestamps = df["timestamp"].values
    start_idx = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start_idx]:
            spans.append((states[start_idx], timestamps[start_idx], timestamps[i - 1]))
            start_idx = i
    return spans


def plot_machine(df, machine_id, output_path):
    g = df[df["machine_id"] == machine_id].sort_values("timestamp").reset_index(drop=True)
    g["timestamp"] = pd.to_datetime(g["timestamp"])

    n_panels = len(FEATURE_COLS) + 1  # +1 for the label/state indicator panel
    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[FEATURE_LABELS[c] for c in FEATURE_COLS] + ["Labels (machine_failure / label_horizon)"],
        row_heights=[1] * len(FEATURE_COLS) + [0.6],
    )

    spans = find_state_spans(g)

    # Shade degrading/failed regions on every feature panel + the label panel
    for row in range(1, n_panels + 1):
        for state, ts_start, ts_end in spans:
            color = STATE_COLORS.get(state)
            if color is None:
                continue
            fig.add_vrect(
                x0=ts_start, x1=ts_end,
                fillcolor=color, line_width=0,
                row=row, col=1,
            )

    # Feature traces
    for i, col in enumerate(FEATURE_COLS, start=1):
        fig.add_trace(
            go.Scatter(
                x=g["timestamp"], y=g[col],
                mode="lines", name=FEATURE_LABELS[col],
                line=dict(width=1.3),
                showlegend=False,
            ),
            row=i, col=1,
        )

    # Label panel: raw event flag + horizon label as step lines
    label_row = n_panels
    fig.add_trace(
        go.Scatter(
            x=g["timestamp"], y=g["machine_failure"],
            mode="lines", name="machine_failure (event)",
            line=dict(width=2, shape="hv", color="crimson"),
        ),
        row=label_row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=g["timestamp"], y=g["label_horizon"] * 0.9,  # offset slightly so both are visible
            mode="lines", name="label_horizon (training target)",
            line=dict(width=2, shape="hv", color="darkorange", dash="dot"),
        ),
        row=label_row, col=1,
    )
    if "_keep_for_training" in g.columns:
        excluded = g[~g["_keep_for_training"]]
        if len(excluded) > 0:
            fig.add_trace(
                go.Scatter(
                    x=excluded["timestamp"], y=[0.5] * len(excluded),
                    mode="markers", name="excluded (leakage buffer)",
                    marker=dict(size=3, color="gray", symbol="x"),
                ),
                row=label_row, col=1,
            )
    fig.update_yaxes(range=[-0.1, 1.1], row=label_row, col=1)

    n_failures = (g["state"] == "failed").sum()
    failure_runs = sum(1 for s, *_ in spans if s == "failed")

    fig.update_layout(
        height=220 * n_panels,
        title=dict(
            text=(
                f"{machine_id} — synthetic sensor timeline "
                f"({failure_runs} failure event(s), {n_failures} rows under repair, "
                f"{len(g)} total rows)"
            ),
            x=0.02, xanchor="left",
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        margin=dict(t=100),
    )
    fig.update_xaxes(title_text="Timestamp", row=n_panels, col=1)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  Wrote {output_path}  ({failure_runs} failure spans, {len(g)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic time series output as interactive Plotly HTML.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV from generate_synthetic.py")
    parser.add_argument("--machine", type=str, default=None, help="Specific machine_id to plot (default: first one found)")
    parser.add_argument("--all-machines", action="store_true", help="Generate one HTML file per machine")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write HTML file(s) to")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLS + ["machine_id", "timestamp", "state", "machine_failure"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing expected columns: {missing}")

    os.makedirs(args.output_dir, exist_ok=True)
    machine_ids = sorted(df["machine_id"].unique())

    if args.all_machines:
        targets = machine_ids
    elif args.machine:
        if args.machine not in machine_ids:
            raise ValueError(f"machine_id '{args.machine}' not found. Available: {machine_ids}")
        targets = [args.machine]
    else:
        targets = [machine_ids[0]]
        print(f"No --machine specified; defaulting to '{targets[0]}'. "
              f"Available machines: {machine_ids}. Use --all-machines to plot all.")

    print(f"Generating {len(targets)} plot(s)...")
    for mid in targets:
        out_path = os.path.join(args.output_dir, f"{mid}_timeline.html")
        plot_machine(df, mid, out_path)


if __name__ == "__main__":
    main()
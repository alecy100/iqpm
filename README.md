# iqPM

Agentic predictive-maintenance studio for the Theme 1 hackathon track. The app streams the provided synthetic equipment time-series into PostgreSQL, keeps feature rows and revealed labels in separate tables, trains per-machine models, logs experiments to MLflow, and exposes a ChatGPT-like operations UI.

## Architecture

- `frontend/`: React + Vite workspace with resizable chat and live-metrics windows.
- `backend/`: FastAPI API, streaming simulator, prediction service, training/deployment service, and tool-calling agent facade.
- `postgres`: source of truth for streamed readings, delayed labels, predictions, experiments, and deployments.
- `mlflow`: experiment tracking endpoint used by the training service.

The source CSV is only mounted into the streaming simulator. Training and inference query PostgreSQL, so the supervised pipeline cannot read unrevealed future labels from the CSV.

## Data Flow

1. `POST /stream/start` reads `synthetic_timeseries.csv` row by row.
2. Sensor features are inserted immediately into `sensor_readings`.
3. `machine_failure`, `state`, `label_horizon`, and `_keep_for_training` are withheld until `LABEL_REVEAL_LAG_HOURS` simulated hours have elapsed, then written to `failure_labels`.
4. Live prediction queries only the latest `sensor_readings` row for a machine.
5. Supervised training joins `sensor_readings` with revealed `failure_labels`, drops rows where `_keep_for_training` is false, trains on `label_horizon`, and uses a strict time-based validation split.

## Local Run

```bash
docker compose up --build
```

Open:

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

Click the play button in the top bar to begin streaming. Machines appear in the selectors as their first rows arrive.

## Useful API Calls

```bash
curl -X POST http://localhost:8000/stream/start
curl http://localhost:8000/fleet/risk
curl -X POST http://localhost:8000/machines/machine_01/retrain
curl http://localhost:8000/models/leaderboard
curl -X POST http://localhost:8000/machines/machine_01/deployment/promote
```

## Modeling Notes

- Supervised target: `label_horizon`.
- Forbidden as model inputs: `machine_failure`, `state`, `_keep_for_training`.
- Training window: latest five simulated days per machine.
- Split: earliest 80 percent for training, most recent 20 percent for validation.
- Gate: F1 must be at least `MIN_F1_FOR_PROMOTION` and beat the current production model on the same validation slice.
- Unsupervised Isolation Forest updates whenever retraining runs, without a labeled-row minimum.
- SHAP explanations are attempted for tree-compatible production models; the API returns a deterministic feature-contribution fallback until a compatible model is available.

## Agent Tools

The chat endpoint routes natural-language requests onto fixed backend tools:

- Analyze/risk: `fleet_risk`, `current_risk`
- Identify: latest per-machine failure risk and anomaly score
- Recommend/train: `trigger_retrain`, `get_leaderboard`
- Explain: `explain_prediction`
- Deploy: `promote_model`

Multiple chat windows and monitor panes share the same backend state, production model, and live prediction stream for each machine.

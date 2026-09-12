import plotly.graph_objects as go
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import SensorReading
from app.services.features import FEATURE_COLUMNS


def sensor_profile(db: Session, machine_id: str | None = None) -> dict:
    query = db.query(SensorReading)
    if machine_id:
        query = query.filter(SensorReading.machine_id == machine_id)
    rows = query.all()
    stats = {}
    for column in FEATURE_COLUMNS:
        values = [getattr(row, column) for row in rows if getattr(row, column) is not None]
        stats[column] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }
    return {
        "machine_id": machine_id,
        "rows": len(rows),
        "machines": _machine_count(db),
        "features": stats,
    }


def trend_chart(db: Session, machine_id: str, limit: int = 180) -> dict:
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    rows = list(reversed(rows))
    figure = go.Figure()
    for column in FEATURE_COLUMNS:
        figure.add_trace(
            go.Scatter(
                x=[row.timestamp.isoformat() for row in rows],
                y=[getattr(row, column) for row in rows],
                mode="lines",
                name=column,
            )
        )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 32, "r": 12, "t": 16, "b": 32},
        xaxis_title="timestamp",
        yaxis_title="sensor value",
    )
    return figure.to_plotly_json()


def _machine_count(db: Session) -> int:
    return int(db.query(func.count(func.distinct(SensorReading.machine_id))).scalar() or 0)

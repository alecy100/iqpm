"""Data access for the per-machine reading tables.

Every query that touches ground-truth labels goes through `_revealed_label_filter`, which
only exposes labels older than LABEL_REVEAL_LAG_HOURS relative to the snapshot cutoff.
"""

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import Table, and_, func, insert, select

from app.config import get_settings
from app.database import engine
from app.services.features import FEATURE_COLUMNS, LABEL_COLUMNS
from app.tables import machines, reading_table


class UnknownMachineError(LookupError):
    pass


_known_machines: set[str] = set()


def list_machines() -> list[str]:
    with engine.connect() as conn:
        found = [row[0] for row in conn.execute(select(machines.c.machine_id).order_by(machines.c.machine_id))]
    _known_machines.update(found)
    return found


def table_for(machine_id: str) -> Table:
    if machine_id not in _known_machines and machine_id not in list_machines():
        known = ", ".join(sorted(_known_machines)) or "none"
        raise UnknownMachineError(f"Unknown machine '{machine_id}'. Known machines: {known}.")
    return reading_table(machine_id)


def insert_reading(machine_id: str, row: dict) -> None:
    with engine.begin() as conn:
        conn.execute(insert(table_for(machine_id)).values(**row))


def latest_timestamp(machine_id: str) -> datetime | None:
    table = table_for(machine_id)
    with engine.connect() as conn:
        return conn.execute(select(func.max(table.c.timestamp))).scalar()


def latest_reading(machine_id: str, include_labels: bool = False) -> dict | None:
    table = table_for(machine_id)
    columns = ["timestamp", *FEATURE_COLUMNS, *(LABEL_COLUMNS if include_labels else [])]
    stmt = select(*[table.c[name] for name in columns]).order_by(table.c.timestamp.desc()).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


def recent_readings(machine_id: str, limit: int) -> pd.DataFrame:
    table = table_for(machine_id)
    stmt = (
        select(table.c.timestamp, *[table.c[name] for name in FEATURE_COLUMNS])
        .order_by(table.c.timestamp.desc())
        .limit(limit)
    )
    return _frame(stmt).iloc[::-1].reset_index(drop=True)


def readings_between(machine_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    table = table_for(machine_id)
    stmt = (
        select(table.c.timestamp, *[table.c[name] for name in FEATURE_COLUMNS])
        .where(table.c.timestamp > start, table.c.timestamp <= end)
        .order_by(table.c.timestamp.asc())
    )
    return _frame(stmt)


def training_window_start(cutoff: datetime) -> datetime:
    return cutoff - timedelta(days=get_settings().rolling_window_days)


def count_labeled_rows(machine_id: str, cutoff: datetime) -> int:
    table = table_for(machine_id)
    stmt = select(func.count()).select_from(table).where(_revealed_label_filter(table, cutoff))
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar() or 0)


def labeled_frame(machine_id: str, cutoff: datetime) -> pd.DataFrame:
    """Rows usable for supervised training as of `cutoff`, oldest first."""
    table = table_for(machine_id)
    stmt = (
        select(table.c.timestamp, *[table.c[name] for name in FEATURE_COLUMNS], table.c.label_horizon)
        .where(_revealed_label_filter(table, cutoff))
        .order_by(table.c.timestamp.asc())
    )
    return _frame(stmt)


def label_summary(machine_id: str, cutoff: datetime) -> dict:
    table = table_for(machine_id)
    stmt = select(func.count(), func.sum(table.c.label_horizon)).where(_revealed_label_filter(table, cutoff))
    with engine.connect() as conn:
        total, positives = conn.execute(stmt).one()
    total = int(total or 0)
    positives = int(positives or 0)
    return {
        "revealed_labeled_rows": total,
        "positive_rows": positives,
        "positive_rate": positives / total if total else None,
    }


def row_counts(machine_id: str) -> dict:
    table = table_for(machine_id)
    stmt = select(table.c.source, func.count(), func.min(table.c.timestamp), func.max(table.c.timestamp)).group_by(
        table.c.source
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return {
        "by_source": {source: int(count) for source, count, _, _ in rows},
        "first_timestamp": min((row[2] for row in rows), default=None),
        "last_timestamp": max((row[3] for row in rows), default=None),
    }


def _revealed_label_filter(table: Table, cutoff: datetime):
    settings = get_settings()
    return and_(
        table.c.timestamp >= training_window_start(cutoff),
        table.c.timestamp <= cutoff - timedelta(hours=settings.label_reveal_lag_hours),
        table.c.label_horizon.is_not(None),
        table.c.keep_for_training.is_(True),
    )


def _frame(stmt) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return pd.DataFrame(rows, columns=list(stmt.selected_columns.keys()))

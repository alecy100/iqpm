import re

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, func

from app.services.features import FEATURE_COLUMNS


metadata = MetaData()

machines = Table(
    "machines",
    metadata,
    Column("machine_id", String(80), primary_key=True),
    Column("table_name", String(120), nullable=False, unique=True),
    Column("created_at", DateTime, server_default=func.now()),
)

agent_threads = Table(
    "agent_threads",
    metadata,
    Column("thread_id", String(120), primary_key=True),
    Column("kind", String(20), nullable=False),  # chat | ops | drift
    Column("machine_id", String(80), index=True),
    Column("created_at", DateTime, server_default=func.now()),
    Column("updated_at", DateTime, server_default=func.now()),
)

MACHINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,60}$")


def reading_table_name(machine_id: str) -> str:
    if not MACHINE_ID_PATTERN.fullmatch(machine_id):
        raise ValueError(f"Invalid machine id: {machine_id!r}")
    return f"readings_{machine_id.lower()}"


def reading_table(machine_id: str) -> Table:
    """One table per machine: sensor features plus delayed ground-truth labels."""
    name = reading_table_name(machine_id)
    existing = metadata.tables.get(name)
    if existing is not None:
        return existing
    return Table(
        name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime, nullable=False, unique=True),
        *[Column(column, Float) for column in FEATURE_COLUMNS],
        Column("machine_failure", Integer),
        Column("state", String(20)),
        Column("label_horizon", Integer),
        Column("keep_for_training", Boolean),
        Column("source", String(10), nullable=False, default="stream"),
    )

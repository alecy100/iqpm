from sqlalchemy import Table, create_engine, insert, select

from app.config import get_settings
from app.tables import agent_threads, machines, metadata, reading_table


def _connect_args(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


settings = get_settings()
engine = create_engine(
    settings.database_url,
    connect_args=_connect_args(settings.database_url),
    pool_pre_ping=True,
)


def init_db() -> None:
    metadata.create_all(engine, tables=[machines, agent_threads])


def register_machine(machine_id: str) -> Table:
    table = reading_table(machine_id)
    table.create(engine, checkfirst=True)
    with engine.begin() as conn:
        exists = conn.execute(select(machines.c.machine_id).where(machines.c.machine_id == machine_id)).first()
        if exists is None:
            conn.execute(insert(machines).values(machine_id=machine_id, table_name=table.name))
    return table

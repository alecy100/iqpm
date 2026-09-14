import logging

import pandas as pd
from sqlalchemy import func, insert, select

from app.config import get_settings
from app.database import engine, register_machine
from app.services.features import FEATURE_COLUMNS, LABEL_COLUMNS


logger = logging.getLogger(__name__)


def seed_from_csv() -> dict[str, int]:
    """Load the synthetic CSV into one table per machine. Machines that already have rows are skipped."""
    path = get_settings().seed_csv_path
    if not path.exists():
        logger.warning("Seed CSV not found at %s; starting with empty machine tables.", path)
        return {}

    frame = pd.read_csv(path, parse_dates=["timestamp"]).rename(columns={"_keep_for_training": "keep_for_training"})
    frame["keep_for_training"] = frame["keep_for_training"].astype(str).str.lower() == "true"
    columns = ["timestamp", *FEATURE_COLUMNS, *LABEL_COLUMNS]

    seeded: dict[str, int] = {}
    for machine_id, group in frame.groupby("machine_id"):
        table = register_machine(str(machine_id))
        with engine.begin() as conn:
            if conn.execute(select(func.count()).select_from(table)).scalar():
                continue
            records = group.sort_values("timestamp")[columns].assign(source="seed").to_dict("records")
            for record in records:
                record["timestamp"] = record["timestamp"].to_pydatetime()
            conn.execute(insert(table), records)
        seeded[str(machine_id)] = len(records)
        logger.info("Seeded %s rows into %s", len(records), table.name)
    return seeded

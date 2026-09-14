import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def to_jsonable(value):
    """Convert numpy/pandas/datetime values into plain JSON types."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

from app.models import SensorReading


FEATURE_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def reading_to_features(reading: SensorReading) -> dict[str, float | None]:
    return {column: getattr(reading, column) for column in FEATURE_COLUMNS}


def feature_vector(reading: SensorReading) -> list[float]:
    values = []
    for column in FEATURE_COLUMNS:
        value = getattr(reading, column)
        values.append(0.0 if value is None else float(value))
    return values

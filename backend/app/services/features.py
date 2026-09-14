FEATURE_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

FEATURE_META = {
    "air_temp_k": {"label": "Air temperature", "unit": "K"},
    "process_temp_k": {"label": "Process temperature", "unit": "K"},
    "rotational_speed_rpm": {"label": "Rotational speed", "unit": "rpm"},
    "torque_nm": {"label": "Torque", "unit": "N·m"},
    "tool_wear_min": {"label": "Tool wear", "unit": "min"},
}

# Ground-truth columns. They live next to the features in each machine table but are
# only ever read through the label-reveal filter in readings.py.
LABEL_COLUMNS = ["machine_failure", "state", "label_horizon", "keep_for_training"]

import numpy as np


def engineer_features(data):
    data = data.copy()

    data["power_w"] = (
        data["Torque"]
        * (data["Rotational speed"] * 2 * np.pi / 60)
    )

    data["temp_diff_k"] = (
        data["Process temperature"]
        - data["Air temperature"]
    )

    data["wear_torque_interaction"] = (
        data["Tool wear"] * data["Torque"]
    )

    return data


def preprocess_features(data):
    data = data.copy()

    # Convert raw Type: L / M / H
    if "Type" in data.columns:
        valid_types = {"L", "M", "H"}

        if not set(data["Type"].unique()).issubset(valid_types):
            raise ValueError("Type must be one of: L, M, H")

        data["type_L"] = (data["Type"] == "L").astype(int)
        data["type_M"] = (data["Type"] == "M").astype(int)

        data = data.drop(columns=["Type"])

    data = engineer_features(data)

    return data
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess_features


EXPECTED_COLUMNS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "type_L",
    "type_M",
    "power_w",
    "temp_diff_k",
    "wear_torque_interaction",
]


def test_preprocessing_output_schema():
    raw = pd.DataFrame(
        [{
            "Type": "M",
            "Air temperature": 298.5,
            "Process temperature": 309.2,
            "Rotational speed": 1420,
            "Torque": 55.3,
            "Tool wear": 190,
        }]
    )

    result = preprocess_features(raw)

    assert list(result.columns) == EXPECTED_COLUMNS


def test_preprocessing_feature_values():
    raw = pd.DataFrame(
        [{
            "Type": "M",
            "Air temperature": 298.5,
            "Process temperature": 309.2,
            "Rotational speed": 1420,
            "Torque": 55.3,
            "Tool wear": 190,
        }]
    )

    result = preprocess_features(raw).iloc[0]

    expected_power = 55.3 * (1420 * 2 * np.pi / 60)

    assert result["type_L"] == 0
    assert result["type_M"] == 1
    assert result["power_w"] == pytest.approx(expected_power)
    assert result["temp_diff_k"] == pytest.approx(10.7)
    assert result["wear_torque_interaction"] == pytest.approx(190 * 55.3)
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=601)

    df = dataset.data.features.copy()
    df["Machine failure"] = dataset.data.targets["Machine failure"]

    output_path = RAW_DATA_DIR / "ai4i2020_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {df.shape} to {output_path}")


if __name__ == "__main__":
    download_data()
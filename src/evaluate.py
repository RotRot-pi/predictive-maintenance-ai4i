import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocessing import preprocess_features


DATA_PATH = "data/raw/ai4i2020_raw.csv"

MODEL_PATH = "models/xgb_model.joblib"
THRESHOLD_PATH = "models/decision_threshold.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"

TARGET = "Machine failure"

MODEL_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    TARGET,
]


def main():
    # Load raw data
    df = pd.read_csv(DATA_PATH)

    # Keep only safe modeling columns
    df = df[MODEL_COLUMNS].copy()

    # Rebuild the exact original Test split
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET],
    )

    X_test_raw = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Apply the exact same preprocessing used during training
    X_test = preprocess_features(X_test_raw)

    # Load saved artifacts
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    # Guarantee exact training feature order
    X_test = X_test[feature_columns]

    # Model score and final decision
    risk_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (risk_scores >= threshold).astype(int)

    # Metrics
    ap = average_precision_score(y_test, risk_scores)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Decision threshold: {threshold:.8f}")
    print(f"Average Precision (AP): {ap:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")

    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
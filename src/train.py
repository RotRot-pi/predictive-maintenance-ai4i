from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from preprocessing import preprocess_features


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020_raw.csv"
MODELS_DIR = PROJECT_ROOT / "models"


# Columns allowed for modeling
MODEL_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Machine failure",
]

TARGET_COLUMN = "Machine failure"

def load_data():
    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in MODEL_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df[MODEL_COLUMNS].copy()

def split_data(df):
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df[TARGET_COLUMN],
    )

    return train_df, val_df, test_df

def prepare_xy(df):
    y = df[TARGET_COLUMN].copy()

    X_raw = df.drop(columns=[TARGET_COLUMN])

    X = preprocess_features(X_raw)

    return X, y

def train_model(X_train, y_train):
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model

def choose_threshold(model, X_val, y_val, min_recall=0.90):
    val_scores = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        val_scores
    )

    valid = recall[:-1] >= min_recall

    if not valid.any():
        return 0.5, None, None

    best_idx = np.argmax(precision[:-1][valid])

    best_threshold = thresholds[valid][best_idx]
    best_precision = precision[:-1][valid][best_idx]
    best_recall = recall[:-1][valid][best_idx]

    return best_threshold, best_precision, best_recall

def save_artifacts(model, threshold, feature_columns):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "xgb_model.joblib"
    )

    joblib.dump(
        float(threshold),
        MODELS_DIR / "decision_threshold.joblib"
    )

    joblib.dump(
        list(feature_columns),
        MODELS_DIR / "feature_columns.joblib"
    )


if __name__ == "__main__":
    df = load_data()

    train_df, val_df, test_df = split_data(df)

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)

    model = train_model(X_train, y_train)

    threshold, precision, recall = choose_threshold(
        model,
        X_val,
        y_val,
        min_recall=0.90
    )

    print("Model trained successfully")
    print("Chosen threshold:", threshold)
    print("Validation precision:", precision)
    print("Validation recall:", recall)

    save_artifacts(
        model,
        threshold,
        X_train.columns
    )

    print("\nArtifacts saved:")
    print("- xgb_model.joblib")
    print("- decision_threshold.joblib")
    print("- feature_columns.joblib")
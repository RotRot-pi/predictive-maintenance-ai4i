# AI4I 2020 Predictive Maintenance

End-to-end machine-failure classification project built with the
**UCI AI4I 2020 Predictive Maintenance Dataset**.

The project covers data exploration, leakage-safe preprocessing, feature
engineering, imbalanced classification, threshold selection, model
explainability, reproducible evaluation, a FastAPI inference service,
automated tests, Docker containerization, and GitHub Actions CI.

> This project performs **binary machine-failure classification for
> predictive-maintenance decision support**. It does not predict Remaining
> Useful Life (RUL) or time-to-failure.

---

## Final Results

The final reference model is an XGBoost classifier evaluated once on a
held-out Test set after model and threshold selection were completed on
Training/Validation data.

| Metric | Test Result |
|---|---:|
| Average Precision (AP) | **0.848893** |
| Recall | **0.897059** |
| Precision | **0.374233** |
| F1 | **0.528139** |
| Decision threshold | **0.15371445** |

Test set:

- 2,000 samples
- 1,932 healthy machines
- 68 machine failures

Confusion matrix:

```text
[[1830, 102],
 [   7,  61]]
```

This means the final model detected **61 of 68 failures**, missed 7, and
generated 102 false alarms.

The operating point intentionally favors failure detection because missed
failures are assumed to be more costly than additional inspection alerts.

---

## Problem

The AI4I dataset contains approximately 10,000 machine observations, with
only about 3.4% representing failures.

This makes the problem highly imbalanced.

For that reason, overall Accuracy is not treated as the main model-selection
metric. The project focuses primarily on:

- Recall
- Precision
- F1
- Average Precision (AP)

---

## Data Leakage Prevention

The dataset contains five failure-type columns:

```text
TWF
HDF
PWF
OSF
RNF
```

These columns directly reveal information about the target and are therefore
excluded from model features.

`UDI` and `Product ID` are also excluded.

The raw columns used for modeling are:

```text
Type
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
```

Target:

```text
Machine failure
```

---

## Train / Validation / Test Strategy

The dataset is split early using stratified sampling:

```text
Train       64%
Validation  16%
Test        20%
```

The Test set is treated as a **lockbox**.

It is not used for:

- model selection
- hyperparameter selection
- feature selection
- threshold selection
- developmental SHAP analysis

The decision threshold is selected using Validation data only.

The Test set is then opened for final evaluation.

---

## Feature Engineering

All reusable preprocessing is centralized in:

```text
src/preprocessing.py
```

The same preprocessing function is used during both training and inference,
avoiding training-serving skew.

### Type encoding

`Type` is one-hot encoded using `H` as the reference category:

```text
L -> type_L = 1, type_M = 0
M -> type_L = 0, type_M = 1
H -> type_L = 0, type_M = 0
```

### Engineered features

Mechanical power:

```python
power_w = Torque * (Rotational_speed * 2 * pi / 60)
```

Temperature difference:

```python
temp_diff_k = Process_temperature - Air_temperature
```

Tool-wear / torque interaction:

```python
wear_torque_interaction = Tool_wear * Torque
```

The final model receives 10 features:

```text
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
type_L
type_M
power_w
temp_diff_k
wear_torque_interaction
```

---

## Model Development

### Baseline

A Logistic Regression baseline was trained using:

```text
StandardScaler
class_weight="balanced"
```

Validation Average Precision:

```text
0.467575
```

The baseline achieved useful failure recall but generated a large number of
false positives.

### XGBoost

Model selection used `GridSearchCV` with 5-fold `StratifiedKFold`.

The search was optimized for:

```python
scoring="average_precision"
```

Best hyperparameters:

```python
learning_rate = 0.1
max_depth = 3
n_estimators = 200
subsample = 0.8
```

Best cross-validation Average Precision:

```text
0.888037
```

Class imbalance is handled using `scale_pos_weight`.

---

## Decision Threshold

The default threshold of `0.5` was not accepted automatically.

The project defines the following operating rule:

> Require Recall >= 0.90 on the Validation set, then select the candidate
> threshold that provides the highest Precision.

The selected Validation threshold was:

```text
0.15371445
```

Validation performance at that operating point:

```text
Precision: 0.352518
Recall:    0.907407
```

The threshold was fixed before final Test evaluation.

---

## Model Explainability

SHAP analysis is implemented in:

```text
notebooks/02_model_explainability.ipynb
```

SHAP was performed on Validation data rather than the Test set.

The strongest model signals were approximately:

```text
1. Tool wear
2. power_w
3. Rotational speed
4. temp_diff_k
5. wear_torque_interaction
6. Torque
7. Process temperature
8. Air temperature
9. type_L
10. type_M
```

The analysis showed that high tool wear, operating speed, torque, and the
engineered physical features materially influence the model.

SHAP values are used for model interpretation, not as evidence of causality.

---

## Risk Score

The API returns:

```text
risk_score
```

rather than:

```text
failure_probability
```

The XGBoost model uses class weighting through `scale_pos_weight`, and its
output has not been probability-calibrated.

The score is therefore used as a model risk score for ranking and
threshold-based decisions, not as a guaranteed real-world probability of
failure.

---

## Project Structure

```text
predictive-maintenance-ai4i/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── raw/
│       └── ai4i2020_raw.csv       # generated locally, ignored by Git
│
├── models/
│   ├── xgb_model.joblib           # generated, ignored by Git
│   ├── decision_threshold.joblib
│   └── feature_columns.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_explainability.ipynb
│
├── src/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Python 3.12 is recommended for the current dependency set.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/RotRot-pi/predictive-maintenance-ai4i.git
cd predictive-maintenance-ai4i

python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproduce the ML Pipeline

### 1. Download the dataset

```bash
python -m src.download_data
```

This retrieves the AI4I 2020 dataset and creates:

```text
data/raw/ai4i2020_raw.csv
```

### 2. Train the model

```bash
python -m src.train
```

This:

1. loads the raw data
2. rebuilds the stratified Train / Validation / Test split
3. applies canonical preprocessing
4. trains XGBoost using the selected hyperparameters
5. selects the threshold using Validation data
6. saves the model artifacts

Generated artifacts:

```text
models/xgb_model.joblib
models/decision_threshold.joblib
models/feature_columns.joblib
```

### 3. Evaluate

```bash
python -m src.evaluate
```

The evaluation script rebuilds the exact Test split, loads the saved model
artifacts, applies the same preprocessing, and reports:

```text
Average Precision
Precision
Recall
F1
Confusion Matrix
```

It performs evaluation only and does not tune the model using Test results.

---

## Run the API

After generating the model artifacts:

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Example request:

```json
{
  "air_temperature": 298.5,
  "process_temperature": 309.2,
  "rotational_speed": 1420,
  "torque": 55.3,
  "tool_wear": 190,
  "product_type": "M"
}
```

Example response from the reference model:

```json
{
  "risk_score": 0.025340067,
  "failure_predicted": false,
  "decision_threshold": 0.15371445
}
```

---

## Tests

The project includes tests for both the API and preprocessing logic.

Run:

```bash
python -m pytest -v
```

The current test suite verifies:

- health endpoint behavior
- valid prediction requests
- invalid product types
- invalid negative values
- preprocessing output schema
- engineered feature calculations

---

## Docker

The Docker image is self-contained and trains the model during the image
build because generated `.joblib` artifacts are intentionally excluded from
Git.

Build:

```bash
docker build -t predictive-maintenance-api .
```

Run:

```bash
docker run --rm -p 8000:8000 predictive-maintenance-api
```

Then:

```text
http://127.0.0.1:8000/docs
```

### Reproducibility note

The reference metrics documented above correspond to the locked reference
model version used for final evaluation.

Because the Docker image regenerates the XGBoost model during build rather
than packaging a fixed binary model artifact, numerical model outputs may
vary across execution platforms even when the same code and dependency
versions are used.

The container is therefore currently validated for reproducible pipeline
execution and API functionality, rather than bit-for-bit identical model
artifacts across operating systems.

A future production version could use explicit model artifact versioning to
guarantee identical deployed model binaries.

---

## Continuous Integration

GitHub Actions validates the containerized application automatically.

The CI workflow:

```text
checkout repository
        ↓
build Docker image
        ↓
run container
        ↓
test /health
        ↓
test /predict
        ↓
cleanup
```

This verifies that the repository can build a working container and serve
predictions in a clean Linux environment without requiring Docker to be
installed on the development machine.

---

## Key Engineering Decisions

Several decisions were intentionally made to keep the evaluation credible:

1. Failure-type columns are excluded to prevent target leakage.
2. The Test set is isolated before model development.
3. Stratified splitting is used because failures are rare.
4. Average Precision is used instead of Accuracy for model selection.
5. Threshold selection is performed on Validation data only.
6. Feature engineering is centralized in one reusable preprocessing module.
7. SHAP development analysis uses Validation rather than Test data.
8. API model outputs are called risk scores rather than calibrated
   probabilities.
9. Test evaluation is separated from training in `src/evaluate.py`.
10. API and preprocessing behavior are covered by automated tests.

---

## Dataset

UCI Machine Learning Repository:

**AI4I 2020 Predictive Maintenance Dataset — Dataset ID 601**

The dataset is synthetic and was designed to reflect common industrial
machine operating conditions and failure mechanisms.

---

## Current Scope

This project focuses on classification of the current machine state:

```text
normal vs failure
```

It does not model temporal degradation or estimate how long remains before a
machine fails.

Natural extensions include:

- probability calibration
- model/data drift monitoring
- model artifact versioning
- deployment to a public cloud service
- temporal predictive-maintenance problems such as Remaining Useful Life
  estimation
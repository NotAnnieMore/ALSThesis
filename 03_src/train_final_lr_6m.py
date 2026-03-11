"""
Train final Logistic Regression model (6-month horizon) for ALS progression.

This script:
  1) loads the processed 6m dataset (one row per subject)
  2) defines the binary target "rapid" as the worst 30% slopes (more negative)
  3) fits a preprocessing + Logistic Regression pipeline
  4) saves the trained model (joblib) and a small metadata JSON

Assumptions:
  - dataset path: 01_data/processed/dataset_6m_v1.csv
  - slope column: slope_180d_per_30d
  - subject id column: subject_id
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Configuration (edit here if needed)
# ----------------------------
DATASET_REL_PATH = os.path.join("01_data", "processed", "dataset_6m_v1.csv")

ID_COL = "subject_id"
SLOPE_COL = "slope_180d_per_30d"

# "rapid" = worst 30% slopes (more negative)
RAPID_FRAC = 0.30

# Final decision threshold (chosen from CV mean threshold in Step 5)
FINAL_THRESHOLD = 0.30

# Output artefacts
MODELS_DIR = "models"
MODEL_FILENAME = "final_lr_6m.joblib"
META_FILENAME = "final_lr_6m_metadata.json"


@dataclass
class TrainMeta:
    model_name: str
    horizon: str
    created_utc: str

    dataset_path: str
    n_samples: int
    rapid_prevalence: float

    slope_col: str
    slope_cutoff_30pct: float

    decision_threshold: float

    feature_cols: List[str]
    num_cols: List[str]
    cat_cols: List[str]


def project_root() -> str:
    """Return the current working directory as the project root."""
    return os.getcwd()


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    for required in (ID_COL, SLOPE_COL):
        if required not in df.columns:
            raise KeyError(f"Missing required column '{required}' in dataset.")

    # Basic cleaning for slope
    df[SLOPE_COL] = pd.to_numeric(df[SLOPE_COL], errors="coerce")
    df = df.dropna(subset=[SLOPE_COL]).copy()

    if df.empty:
        raise ValueError("No rows with valid slope after cleaning.")

    return df


def build_target(df: pd.DataFrame) -> Tuple[pd.Series, float]:
    """Create y (rapid=1) and return the slope cutoff used."""
    cutoff = float(df[SLOPE_COL].quantile(RAPID_FRAC))
    y = (df[SLOPE_COL] <= cutoff).astype(int)
    return y, cutoff


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return X and list of feature columns."""
    drop_cols = {ID_COL, SLOPE_COL}

    # If the dataset contains 3m slope too, drop it (avoid leakage/target contamination)
    if "slope_90d_per_30d" in df.columns:
        drop_cols.add("slope_90d_per_30d")

    feat_cols = [c for c in df.columns if c not in drop_cols]
    if not feat_cols:
        raise ValueError("No feature columns left after dropping ID/target columns.")

    X = df[feat_cols].copy()
    return X, feat_cols


def split_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical based on dtypes."""
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """Preprocessing + Logistic Regression."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
    )

    return Pipeline(steps=[("prep", preproc), ("clf", clf)])


def save_artifacts(model: Pipeline, meta: TrainMeta) -> Tuple[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    meta_path = os.path.join(MODELS_DIR, META_FILENAME)

    joblib.dump(model, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return model_path, meta_path


def main() -> None:
    root = project_root()
    dataset_path = os.path.join(root, DATASET_REL_PATH)

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Loading dataset: {dataset_path}")

    df = load_dataset(dataset_path)
    print(f"[INFO] Loaded rows: {len(df)}")

    y, slope_cutoff = build_target(df)
    X, feat_cols = build_features(df)
    num_cols, cat_cols = split_types(X)

    print(f"[INFO] rapid prevalence: {y.mean():.3f}")
    print(f"[INFO] slope cutoff (30%): {slope_cutoff:.4f}")
    print(f"[INFO] features: {len(feat_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")

    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X, y)

    # Sanity check
    proba = pipe.predict_proba(X)[:, 1]
    preds = (proba >= FINAL_THRESHOLD).astype(int)
    print(f"[INFO] proba: min={proba.min():.3f} mean={proba.mean():.3f} max={proba.max():.3f}")
    print(f"[INFO] positives @ threshold {FINAL_THRESHOLD:.2f}: {preds.sum()} / {len(preds)}")

    meta = TrainMeta(
        model_name="LogisticRegression(class_weight=balanced)",
        horizon="6m",
        created_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        dataset_path=DATASET_REL_PATH,
        n_samples=int(len(df)),
        rapid_prevalence=float(y.mean()),
        slope_col=SLOPE_COL,
        slope_cutoff_30pct=float(slope_cutoff),
        decision_threshold=float(FINAL_THRESHOLD),
        feature_cols=feat_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    model_path, meta_path = save_artifacts(pipe, meta)

    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
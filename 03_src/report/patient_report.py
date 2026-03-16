"""
Patient Risk Report Generator — ALS Rapid-Progression Screening.

Loads the trained XGBoost pipeline and generates a textual report for a
single patient, including:
  1. Probability of rapid progression (6-month horizon)
  2. Binary classification at F1-optimal and F2-optimal thresholds
  3. Top-5 contributing features (SHAP local explanation)
  4. Monitoring suggestions based on population percentile comparison

Usage
-----
  # From workspace root:
  python 03_src/report/patient_report.py                       # demo patient
  python 03_src/report/patient_report.py --patient-id 12345    # from TEST set
  python 03_src/report/patient_report.py --json patient.json   # from JSON file

Outputs
-------
  Printed textual report to stdout.
  Optionally saved to 04_outputs/reports/<patient_id>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# Paths (relative to workspace root)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "final_xgb_6m.joblib")
META_PATH  = os.path.join("models", "final_xgb_6m_metadata.json")
DATA_PATH  = os.path.join("01_data", "processed", "dataset_6m_v2.csv")
SPLIT_PATH = os.path.join("01_data", "processed", "holdout_split_6m.csv")
OUT_DIR    = os.path.join("04_outputs", "reports")

# Thresholds from DEV OOF sweep (evaluate_holdout.py)
THR_F1 = 0.21   # F1-optimal
THR_F2 = 0.21   # F2-optimal (≈ same in this dataset; F2 curve is flat 0.15–0.21)

N_TOP = 5       # number of SHAP features to show

# ─────────────────────────────────────────────────────────────
# Human-readable feature labels
# ─────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "ALSFRS_R_t0":                     "ALSFRS-R total (baseline)",
    "Age":                             "Age (years)",
    "Sex":                             "Sex",
    "Ethnicity":                       "Ethnicity",
    "Race_Caucasian":                  "Race: Caucasian",
    "Race_Black_African_American":     "Race: Black/African-American",
    "Race_Asian":                      "Race: Asian",
    "Race_Americ_Indian_Alaska_Native":"Race: Am. Indian/Alaska Native",
    "Race_Hawaiian_Pacific_Islander":  "Race: Hawaiian/Pacific Isl.",
    "Race_Other":                      "Race: Other",
    "Race_Unknown":                    "Race: Unknown",
    "R_1_Dyspnea":                     "ALSFRS-R item: Dyspnea",
    "R_2_Orthopnea":                   "ALSFRS-R item: Orthopnea",
    "R_3_Respiratory_Insufficiency":   "ALSFRS-R item: Resp. Insufficiency",
    "Mode_of_Administration":          "Mode of Administration",
    "Weight_kg_t0":                    "Weight (kg)",
    "Height_cm_t0":                    "Height (cm)",
    "BMI_t0":                          "BMI",
    "Pulse":                           "Pulse (bpm)",
    "Respiratory_Rate":                "Respiratory Rate",
    "Temperature":                     "Temperature (°C)",
    "Blood_Pressure_Systolic":         "BP Systolic (mmHg)",
    "Blood_Pressure_Diastolic":        "BP Diastolic (mmHg)",
    "Baseline_Standing_BP_Systolic":   "Standing BP Systolic",
    "Baseline_Standing_BP_Diastolic":  "Standing BP Diastolic",
    "Baseline_Supine_BP_Systolic":     "Supine BP Systolic",
    "Baseline_Supine_BP_Diastolic":    "Supine BP Diastolic",
    "FVC_Liters_best_t0":             "FVC (litres)",
    "FVC_pctNormal_best_t0":          "FVC (% predicted normal)",
    "subject_normal":                  "FVC subject normal flag",
    "riluzole_pre_t0":                 "Riluzole (pre-baseline)",
    "study_arm":                       "Study arm",
    "n_conmeds_pre_t0":                "Concomitant medications (n)",
    "alt_t0":                          "ALT (U/L, baseline)",
    "creatinine_t0":                   "Creatinine (µmol/L, baseline)",
}

# Clinical monitoring rules — keyed by feature name.
# Each entry: (condition_description, suggestion)
# Only triggered when the feature is in the top-N SHAP contributors.
MONITORING_RULES = {
    "ALSFRS_R_t0": {
        "low_p25": 35.0,
        "suggestion": "ALSFRS-R is below the 25th population percentile. "
                      "Recommend monthly functional reassessment.",
    },
    "FVC_Liters_best_t0": {
        "low_p25": 2.78,
        "suggestion": "FVC is below the 25th population percentile (2.78 L). "
                      "Recommend spirometry follow-up every 4 weeks.",
    },
    "FVC_pctNormal_best_t0": {
        "low_p25": 77.0,
        "suggestion": "FVC % predicted is below the 25th percentile (77%). "
                      "Recommend respiratory function monitoring.",
    },
    "Respiratory_Rate": {
        "high_p75": 20.0,
        "suggestion": "Respiratory rate above the 75th percentile (20). "
                      "Monitor respiratory function; consider NIV evaluation.",
    },
    "BMI_t0": {
        "low_p25": 23.3,
        "suggestion": "BMI is below the 25th percentile (23.3). "
                      "Recommend nutritional assessment and weight monitoring.",
    },
    "Age": {
        "high_p75": 65.0,
        "suggestion": "Patient age above the 75th percentile (65 years). "
                      "Higher age is associated with faster decline.",
    },
    "R_1_Dyspnea": {
        "low_p25": 3.0,
        "suggestion": "Dyspnea sub-score at or below P25. "
                      "Recommend respiratory symptom monitoring.",
    },
    "R_2_Orthopnea": {
        "low_p25": 4.0,
        "suggestion": "Orthopnea sub-score below baseline normative value. "
                      "Monitor nocturnal respiratory support needs.",
    },
    "R_3_Respiratory_Insufficiency": {
        "low_p25": 4.0,
        "suggestion": "Respiratory insufficiency sub-score below baseline. "
                      "Evaluate need for respiratory intervention.",
    },
    "Weight_kg_t0": {
        "low_p25": 68.0,
        "suggestion": "Weight below the 25th percentile (68 kg). "
                      "Recommend nutritional support evaluation.",
    },
    "Pulse": {
        "high_p75": 83.0,
        "suggestion": "Pulse above the 75th percentile. "
                      "Monitor cardiovascular parameters.",
    },
    "creatinine_t0": {
        "low_p25": 53.0,
        "suggestion": "Low creatinine may reflect reduced muscle mass. "
                      "Consider as a marker of disease burden.",
    },
    "alt_t0": {
        "high_p75": 45.0,
        "suggestion": "ALT above the 75th percentile (45 U/L). "
                      "Monitor liver function; review concomitant medications.",
    },
}


def readable(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


# ─────────────────────────────────────────────────────────────
# Load model & metadata
# ─────────────────────────────────────────────────────────────
def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]
    slope_cutoff = meta["slope_cutoff_30pct"]
    return pipe, meta, feat_cols, slope_cutoff


# ─────────────────────────────────────────────────────────────
# Build patient vector
# ─────────────────────────────────────────────────────────────
def patient_from_dataset(patient_id: int, feat_cols: list[str]):
    """Load a patient from the processed dataset by subject_id."""
    df = pd.read_csv(DATA_PATH)
    row = df[df["subject_id"] == patient_id]
    if row.empty:
        sys.exit(f"ERROR: patient {patient_id} not found in dataset.")
    row = row.iloc[0]
    X = row[feat_cols].to_frame().T.reset_index(drop=True)
    slope_val = row.get("slope_180d_per_30d", None)
    return X, slope_val


def patient_from_json(json_path: str, feat_cols: list[str]):
    """Load a patient from a JSON file with feature key–value pairs."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [c for c in feat_cols if c not in data]
    if missing:
        print(f"WARNING: {len(missing)} features missing from JSON — "
              f"will be NaN (imputed by pipeline): {missing[:5]}...")
    X = pd.DataFrame([{c: data.get(c, np.nan) for c in feat_cols}])
    return X, None


# ─────────────────────────────────────────────────────────────
# SHAP local explanation
# ─────────────────────────────────────────────────────────────
def compute_shap(pipe, X, feat_cols):
    """Compute SHAP values for a single patient."""
    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]
    X_trans = prep.transform(X)

    explainer = shap.TreeExplainer(clf)
    sv = explainer(X_trans)

    # Map transformed feature names back to original
    if hasattr(prep, "get_feature_names_out"):
        trans_names = list(prep.get_feature_names_out())
    else:
        trans_names = feat_cols

    vals  = sv.values[0]   # shape: (n_transformed_features,)
    base  = sv.base_values[0]

    # Aggregate one-hot encoded features back to original
    agg = {}
    for i, tn in enumerate(trans_names):
        orig = _map_to_original(tn, feat_cols)
        agg[orig] = agg.get(orig, 0.0) + vals[i]

    shap_series = pd.Series(agg)
    return shap_series, base


def _map_to_original(trans_name: str, feat_cols: list[str]) -> str:
    """Map a transformed feature name back to the original feature name."""
    # Direct match
    if trans_name in feat_cols:
        return trans_name
    # One-hot: "cat__Sex_Male" → try "Sex"
    for fc in feat_cols:
        if trans_name.startswith(f"cat__{fc}") or trans_name.startswith(f"num__{fc}"):
            return fc
        if fc in trans_name:
            return fc
    return trans_name


# ─────────────────────────────────────────────────────────────
# Monitoring suggestions
# ─────────────────────────────────────────────────────────────
def generate_suggestions(top_features: list[tuple], X: pd.DataFrame):
    """Given top-N (feature, shap_value) pairs, generate monitoring text."""
    suggestions = []
    for feat, shap_val in top_features:
        rule = MONITORING_RULES.get(feat)
        if rule is None:
            continue
        val = X[feat].iloc[0] if feat in X.columns else None
        if val is None or pd.isna(val):
            continue

        triggered = False
        if "low_p25" in rule and val <= rule["low_p25"]:
            triggered = True
        if "high_p75" in rule and val >= rule["high_p75"]:
            triggered = True

        if triggered:
            suggestions.append({
                "feature": feat,
                "readable": readable(feat),
                "value": val,
                "shap": shap_val,
                "suggestion": rule["suggestion"],
            })
    return suggestions


# ─────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────
def format_report(patient_id, proba, shap_series, base_value, X,
                  slope_val, slope_cutoff):
    """Build the full textual report."""
    W = 62  # column width
    lines = []

    # ── Header ──
    lines.append("=" * W)
    lines.append("  ALS RAPID-PROGRESSION RISK REPORT")
    lines.append("  Model: XGBoost (Optuna-tuned) — 6-month horizon")
    lines.append("=" * W)

    # ── 1. Risk assessment ──
    lines.append("")
    lines.append("─── RISK ASSESSMENT " + "─" * (W - 20))
    lines.append(f"  Patient ID:         {patient_id}")
    lines.append(f"  P(rapid, 6 months): {proba:.4f}  ({proba*100:.1f}%)")

    # Risk tier
    if proba >= 0.50:
        tier = "HIGH"
    elif proba >= THR_F1:
        tier = "MODERATE"
    else:
        tier = "LOW"
    lines.append(f"  Risk tier:          {tier}")
    lines.append("")

    # Classification at both thresholds
    cls_f1 = "RAPID" if proba >= THR_F1 else "SLOW"
    cls_f2 = "RAPID" if proba >= THR_F2 else "SLOW"
    sym_f1 = "⚠" if cls_f1 == "RAPID" else "✓"
    sym_f2 = "⚠" if cls_f2 == "RAPID" else "✓"
    lines.append(f"  Classification (F1-optimal, t={THR_F1}):")
    lines.append(f"    {sym_f1} {cls_f1} PROGRESSOR")
    lines.append(f"  Classification (F2-optimal, t={THR_F2}):")
    lines.append(f"    {sym_f2} {cls_f2} PROGRESSOR")

    if slope_val is not None:
        true_label = "RAPID" if slope_val <= slope_cutoff else "SLOW"
        lines.append(f"  True label (known): {true_label}  "
                     f"(slope={slope_val:.4f}, cutoff={slope_cutoff:.4f})")

    lines.append("─" * W)

    # ── 2. Top contributing features ──
    lines.append("")
    lines.append("─── TOP-5 CONTRIBUTING FEATURES (SHAP) " + "─" * (W - 39))
    lines.append(f"  Base value (population average log-odds): {base_value:.4f}")
    lines.append("")

    top_abs = shap_series.abs().sort_values(ascending=False).head(N_TOP)
    top_feats = [(feat, shap_series[feat]) for feat in top_abs.index]

    lines.append(f"  {'#':<3} {'Feature':<32} {'Value':>8}  {'SHAP':>7}  Direction")
    lines.append(f"  {'─'*3} {'─'*32} {'─'*8}  {'─'*7}  {'─'*15}")

    for i, (feat, sv) in enumerate(top_feats, 1):
        val = X[feat].iloc[0] if feat in X.columns else "N/A"
        val_str = f"{val:.1f}" if isinstance(val, (int, float, np.floating)) and not pd.isna(val) else str(val)
        direction = "↑ increases risk" if sv > 0 else "↓ decreases risk"
        lines.append(f"  {i:<3} {readable(feat):<32} {val_str:>8}  {sv:>+7.4f}  {direction}")

    lines.append("─" * W)

    # ── 3. Monitoring suggestions ──
    lines.append("")
    lines.append("─── MONITORING SUGGESTIONS " + "─" * (W - 27))

    suggestions = generate_suggestions(top_feats, X)
    if suggestions:
        lines.append("  Based on the top risk drivers for this patient:")
        lines.append("")
        for s in suggestions:
            val_str = f"{s['value']:.1f}" if isinstance(s['value'], (int, float, np.floating)) else str(s['value'])
            lines.append(f"  • {s['readable']} = {val_str}")
            lines.append(f"    → {s['suggestion']}")
            lines.append("")
    else:
        lines.append("  No specific monitoring triggers activated.")
        lines.append("  The top contributing features are within normal")
        lines.append("  population ranges or do not have predefined rules.")
        lines.append("")

    # General recommendation
    if tier == "HIGH":
        lines.append("  General: HIGH risk classification.")
        lines.append("  → Recommend multidisciplinary team review and")
        lines.append("    intensified follow-up schedule (monthly).")
    elif tier == "MODERATE":
        lines.append("  General: MODERATE risk classification.")
        lines.append("  → Standard follow-up; reassess in 3 months.")
    else:
        lines.append("  General: LOW risk classification.")
        lines.append("  → Standard care pathway; reassess in 6 months.")

    lines.append("─" * W)

    # ── Disclaimer ──
    lines.append("")
    lines.append("DISCLAIMER: This report is generated by a machine-learning")
    lines.append("model trained on PRO-ACT data for research purposes only.")
    lines.append("It does NOT constitute medical advice. Clinical decisions")
    lines.append("should always be made by qualified healthcare professionals.")
    lines.append("=" * W)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Demo patient (default)
# ─────────────────────────────────────────────────────────────
def get_demo_patient_id():
    """Pick a patient from the TEST set for demonstration."""
    hs = pd.read_csv(SPLIT_PATH)
    test_ids = hs.loc[hs["partition"] == "test", "subject_id"].tolist()
    df = pd.read_csv(DATA_PATH)
    df_test = df[df["subject_id"].isin(test_ids)]
    slope_col = "slope_180d_per_30d"
    cutoff = -1.2475
    rapids = df_test[df_test[slope_col] <= cutoff]
    if not rapids.empty:
        return int(rapids.iloc[0]["subject_id"])
    return int(test_ids[0])


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate ALS rapid-progression risk report for a patient."
    )
    parser.add_argument("--patient-id", type=int, default=None,
                        help="subject_id from the dataset")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to JSON file with patient features")
    parser.add_argument("--save", action="store_true",
                        help="Save report to 04_outputs/reports/")
    args = parser.parse_args()

    print("Loading model...")
    pipe, meta, feat_cols, slope_cutoff = load_model()

    # Build patient vector
    if args.json:
        X, slope_val = patient_from_json(args.json, feat_cols)
        patient_id = "JSON_input"
    elif args.patient_id:
        X, slope_val = patient_from_dataset(args.patient_id, feat_cols)
        patient_id = args.patient_id
    else:
        patient_id = get_demo_patient_id()
        print(f"No patient specified — using demo patient {patient_id} from TEST set.")
        X, slope_val = patient_from_dataset(patient_id, feat_cols)

    # Predict
    proba = float(pipe.predict_proba(X)[:, 1][0])

    # SHAP explanation
    print("Computing SHAP explanation...")
    shap_series, base_value = compute_shap(pipe, X, feat_cols)

    # Generate report
    report = format_report(patient_id, proba, shap_series, base_value,
                           X, slope_val, slope_cutoff)
    print()
    print(report)

    # Optionally save
    if args.save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"report_{patient_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()

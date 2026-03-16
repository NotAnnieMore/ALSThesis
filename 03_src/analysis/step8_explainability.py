"""
Step 8 — Explainability: SHAP (global + local) + LIME (local) + LR coefficients.

Protocol
--------
1. Load final XGBoost model and TEST set (holdout_split_6m)
2. SHAP global: mean-|SHAP| bar plot + beeswarm (summary) plot
3. SHAP local:  waterfall plots for 4 representative patients (TP, FP, FN, TN)
4. LIME local:  explanations for the same 4 patients (side-by-side with SHAP)
5. LR coefficients: bar chart of standardised coefficients from final LR model
6. SHAP vs LIME concordance table (top-N feature overlap)

Usage
-----
  cd g:\\TESE\\CodigoMain
  python 03_src/analysis/step8_explainability.py

Outputs
-------
  04_outputs/figures/step8_xai/shap_bar_global.pdf / .png
  04_outputs/figures/step8_xai/shap_beeswarm.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_TP.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_FP.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_FN.pdf / .png
  04_outputs/figures/step8_xai/shap_waterfall_TN.pdf / .png
  04_outputs/figures/step8_xai/lime_local_TP.pdf / .png
  04_outputs/figures/step8_xai/lime_local_FP.pdf / .png
  04_outputs/figures/step8_xai/lime_local_FN.pdf / .png
  04_outputs/figures/step8_xai/lime_local_TN.pdf / .png
  04_outputs/figures/step8_xai/lr_coefficients.pdf / .png
  04_outputs/tables/step8_xai/shap_feature_importance.csv
  04_outputs/tables/step8_xai/shap_vs_lime_concordance.csv
  04_outputs/tables/step8_xai/lr_coefficients.csv
  04_outputs/tables/step8_xai/case_study_profiles.csv
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import shap
from lime.lime_tabular import LimeTabularExplainer

from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
PROCESSED    = os.path.join("01_data", "processed")
OUT_FIG      = os.path.join("04_outputs", "figures", "step8_xai")
OUT_TAB      = os.path.join("04_outputs", "tables", "step8_xai")
OVERLEAF_FIG = os.path.join("overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join("overleaf", "images", "tables")

MODEL_XGB_PATH = os.path.join("models", "final_xgb_6m.joblib")
MODEL_LR_PATH  = os.path.join("models", "final_lr_6m.joblib")
META_XGB_PATH  = os.path.join("models", "final_xgb_6m_metadata.json")
META_LR_PATH   = os.path.join("models", "final_lr_6m_metadata.json")

HORIZON    = "6m"
SEED       = 42
RAPID_FRAC = 0.30
THRESHOLD  = 0.21          # F1-optimal threshold from Step 6

N_TOP_FEATURES  = 15       # for bar chart / concordance
N_LIME_SAMPLES  = 2000     # perturbation samples for LIME

DROP_COLS = {
    "subject_id",
    "t0_delta_days",
    "vitals_delta_days",
    "fvc_delta_days",
    "slope_90d_per_30d",
    "slope_180d_per_30d",
    "ALSFRS_Responded_By",
}

# Readable feature names for plots
FEATURE_LABELS = {
    "ALSFRS_R_t0":                     "ALSFRS-R total (baseline)",
    "Age":                             "Age",
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
    "Pulse":                           "Pulse",
    "Respiratory_Rate":                "Respiratory Rate",
    "Temperature":                     "Temperature",
    "Blood_Pressure_Systolic":         "BP Systolic",
    "Blood_Pressure_Diastolic":        "BP Diastolic",
    "Baseline_Standing_BP_Systolic":   "Standing BP Systolic",
    "Baseline_Standing_BP_Diastolic":  "Standing BP Diastolic",
    "Baseline_Supine_BP_Systolic":     "Supine BP Systolic",
    "Baseline_Supine_BP_Diastolic":    "Supine BP Diastolic",
    "FVC_Liters_best_t0":             "FVC (litres)",
    "FVC_pctNormal_best_t0":          "FVC (% normal)",
    "subject_normal":                  "FVC subject normal flag",
    "riluzole_pre_t0":                 "Riluzole (pre-baseline)",
    "study_arm":                       "Study arm",
    "n_conmeds_pre_t0":                "Concomitant medications (n)",
    "alt_t0":                          "ALT (baseline)",
    "creatinine_t0":                   "Creatinine (baseline)",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in (OUT_FIG, OUT_TAB, OVERLEAF_FIG, OVERLEAF_TAB):
        os.makedirs(d, exist_ok=True)


def save_fig(fig, name):
    """Save figure to outputs + overleaf, both PDF and PNG."""
    for d in (OUT_FIG, OVERLEAF_FIG):
        fig.savefig(os.path.join(d, f"{name}.pdf"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def readable(name):
    """Return a readable label for a feature name."""
    return FEATURE_LABELS.get(name, name)


def load_test_data():
    """Load TEST partition with features and labels."""
    ds_path = os.path.join(PROCESSED, f"dataset_{HORIZON}_v2.csv")
    hs_path = os.path.join(PROCESSED, f"holdout_split_{HORIZON}.csv")

    df = pd.read_csv(ds_path)
    hs = pd.read_csv(hs_path)

    slope_col = "slope_180d_per_30d"
    df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
    df = df.dropna(subset=[slope_col])

    dev_ids  = set(hs.loc[hs["partition"] == "dev",  "subject_id"])
    test_ids = set(hs.loc[hs["partition"] == "test", "subject_id"])

    df_dev  = df[df["subject_id"].isin(dev_ids)].copy()
    df_test = df[df["subject_id"].isin(test_ids)].copy()

    # Label threshold from DEV only
    cutoff = float(df_dev[slope_col].quantile(RAPID_FRAC))

    feat_cols = [c for c in df.columns if c not in DROP_COLS]

    X_test  = df_test[feat_cols].reset_index(drop=True)
    y_test  = (df_test[slope_col] <= cutoff).astype(int).reset_index(drop=True)
    ids_test = df_test["subject_id"].reset_index(drop=True)

    # Also return DEV features for LIME background
    X_dev = df_dev[feat_cols].reset_index(drop=True)

    print(f"  TEST n={len(X_test)}, Rapid prevalence={y_test.mean():.3f}")
    print(f"  Slope cutoff (DEV P30): {cutoff:.4f}")
    return X_test, y_test, ids_test, X_dev, feat_cols, cutoff


def pick_case_studies(y_true, y_pred, proba):
    """
    Select one representative patient from each confusion-matrix quadrant.
    Pick the patient closest to the quadrant's median predicted probability,
    so the case study is 'typical' rather than extreme.
    """
    cases = {}
    for label, mask_fn in [
        ("TP", lambda yt, yp: (yt == 1) & (yp == 1)),
        ("FP", lambda yt, yp: (yt == 0) & (yp == 1)),
        ("FN", lambda yt, yp: (yt == 1) & (yp == 0)),
        ("TN", lambda yt, yp: (yt == 0) & (yp == 0)),
    ]:
        mask = mask_fn(y_true, y_pred)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"  [WARN] No {label} cases found — skipping")
            continue
        # pick patient closest to quadrant median probability
        med = np.median(proba[idxs])
        best = idxs[np.argmin(np.abs(proba[idxs] - med))]
        cases[label] = int(best)
    return cases


# ─────────────────────────────────────────────────────────────
# SHAP analysis
# ─────────────────────────────────────────────────────────────
def run_shap(pipe, X_test, feat_cols, case_indices):
    """Compute SHAP values using TreeExplainer on the XGBoost model."""
    print("\n── SHAP analysis ──")

    # Extract the preprocessor and the XGBoost booster from the pipeline
    preproc = pipe.named_steps["prep"]
    clf     = pipe.named_steps["clf"]

    X_processed = preproc.transform(X_test)

    # Get transformed feature names
    transformed_names = []
    for name, trans, cols in preproc.transformers_:
        if name == "num":
            transformed_names.extend(cols)
        elif name == "cat":
            if hasattr(trans, "named_steps"):
                ohe = (trans.named_steps.get("onehot")
                       or trans.named_steps.get("oh"))
            else:
                ohe = trans[-1]
            transformed_names.extend(ohe.get_feature_names_out(cols).tolist())

    if isinstance(X_processed, np.ndarray):
        X_df = pd.DataFrame(X_processed, columns=transformed_names)
    else:
        X_df = pd.DataFrame(X_processed.toarray(), columns=transformed_names)

    # TreeSHAP (exact, fast)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_df)

    # ── Global: mean |SHAP| bar chart ──
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": transformed_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.to_csv(os.path.join(OUT_TAB, "shap_feature_importance.csv"), index=False)

    top_n = importance_df.head(N_TOP_FEATURES)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        [readable(f) for f in top_n["feature"].values[::-1]],
        top_n["mean_abs_shap"].values[::-1],
        color="#2b7bba", edgecolor="white", height=0.65,
    )
    # annotate values
    for bar, val in zip(bars, top_n["mean_abs_shap"].values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="black")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance — XGBoost (6-month, TEST set)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "shap_bar_global")

    # ── Global: beeswarm ──
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=N_TOP_FEATURES, show=False)
    plt.title("SHAP Beeswarm — XGBoost (6-month, TEST set)",
              fontsize=12, fontweight="bold")
    fig = plt.gcf()
    fig.tight_layout()
    save_fig(fig, "shap_beeswarm")

    # ── Local: waterfall plots for case studies ──
    for label, idx in case_indices.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        plt.title(f"SHAP Waterfall — {label} (patient index {idx})",
                  fontsize=11, fontweight="bold")
        fig = plt.gcf()
        fig.tight_layout()
        save_fig(fig, f"shap_waterfall_{label}")

    return shap_values, importance_df, X_df, transformed_names


# ─────────────────────────────────────────────────────────────
# LIME analysis
# ─────────────────────────────────────────────────────────────
def run_lime(pipe, X_test, X_dev, feat_cols, case_indices, y_test,
             transformed_names):
    """Generate LIME explanations for the same 4 case-study patients.

    LIME requires numeric input, so we operate on the preprocessed
    (imputed + encoded) feature space and wrap only the classifier
    for predictions.
    """
    print("\n── LIME analysis ──")

    preproc = pipe.named_steps["prep"]
    clf     = pipe.named_steps["clf"]

    # Transform train/test to numeric arrays
    X_dev_proc  = preproc.transform(X_dev)
    X_test_proc = preproc.transform(X_test)

    if not isinstance(X_dev_proc, np.ndarray):
        X_dev_proc  = X_dev_proc.toarray()
    if not isinstance(X_test_proc, np.ndarray):
        X_test_proc = X_test_proc.toarray()

    readable_names = [readable(f) for f in transformed_names]

    # Predict wrapper — clf expects already-transformed data
    def predict_fn(X):
        return clf.predict_proba(X)

    explainer = LimeTabularExplainer(
        training_data=X_dev_proc,
        feature_names=readable_names,
        class_names=["Slow", "Rapid"],
        mode="classification",
        random_state=SEED,
        discretize_continuous=True,
    )

    lime_rankings = {}

    for label, idx in case_indices.items():
        instance = X_test_proc[idx]

        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=N_TOP_FEATURES,
            num_samples=N_LIME_SAMPLES,
            labels=(1,),
        )

        # Get the explanation for the positive class (Rapid = 1)
        exp_list = explanation.as_list(label=1)
        lime_rankings[label] = [feat for feat, _ in exp_list]

        # Generate LIME figure
        fig = explanation.as_pyplot_figure(label=1)
        fig.set_size_inches(9, 6)
        fig.suptitle(f"LIME Explanation — {label} (patient index {idx})",
                     fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_fig(fig, f"lime_local_{label}")

    return lime_rankings


# ─────────────────────────────────────────────────────────────
# Concordance: SHAP vs LIME
# ─────────────────────────────────────────────────────────────
def concordance_table(shap_importance_df, lime_rankings, case_indices,
                      shap_values, top_k=10):
    """Compare top-K feature rankings from SHAP (global) vs LIME (local)."""
    print("\n── SHAP vs LIME concordance ──")

    shap_top = shap_importance_df.head(top_k)["feature"].tolist()

    rows = []
    for label in case_indices:
        if label not in lime_rankings:
            continue
        # LIME feature names are readable; convert SHAP to readable for comparison
        shap_readable = [readable(f) for f in shap_top]

        # Extract the feature name from LIME's compound expressions (e.g. "BMI > 25.3")
        lime_feats_raw = lime_rankings[label][:top_k]
        lime_feats_clean = []
        for expr in lime_feats_raw:
            # extract feature name (before <, >, <=, >=, =)
            for op in [" <= ", " >= ", " < ", " > ", " = "]:
                if op in expr:
                    lime_feats_clean.append(expr.split(op)[0].strip())
                    break
            else:
                lime_feats_clean.append(expr.strip())

        overlap = set(shap_readable) & set(lime_feats_clean)
        rows.append({
            "case": label,
            "shap_top_k": ", ".join(shap_readable),
            "lime_top_k": ", ".join(lime_feats_clean),
            "overlap_count": len(overlap),
            "overlap_features": ", ".join(sorted(overlap)),
        })

    conc_df = pd.DataFrame(rows)
    conc_df.to_csv(os.path.join(OUT_TAB, "shap_vs_lime_concordance.csv"), index=False)
    print(f"  [OK] Concordance table saved ({len(rows)} cases, top-{top_k})")
    return conc_df


# ─────────────────────────────────────────────────────────────
# LR coefficient analysis
# ─────────────────────────────────────────────────────────────
def run_lr_coefficients():
    """Extract and plot LR coefficients from the final LR model."""
    print("\n── Logistic Regression coefficients ──")

    if not os.path.exists(MODEL_LR_PATH):
        print("  [SKIP] LR model not found — skipping coefficient analysis")
        return

    pipe_lr = joblib.load(MODEL_LR_PATH)
    preproc = pipe_lr.named_steps["prep"]
    clf     = pipe_lr.named_steps["clf"]

    # Get transformed feature names
    transformed_names = []
    for name, trans, cols in preproc.transformers_:
        if name == "num":
            transformed_names.extend(cols)
        elif name == "cat":
            if hasattr(trans, "named_steps"):
                ohe = (trans.named_steps.get("onehot")
                       or trans.named_steps.get("oh"))
            else:
                ohe = trans[-1]
            transformed_names.extend(ohe.get_feature_names_out(cols).tolist())

    coefs = clf.coef_[0]
    coef_df = pd.DataFrame({
        "feature": transformed_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    coef_df.to_csv(os.path.join(OUT_TAB, "lr_coefficients.csv"), index=False)

    # Plot top coefficients
    top = coef_df.head(N_TOP_FEATURES).copy()
    top = top.sort_values("abs_coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d9534f" if c > 0 else "#3274a1" for c in top["coefficient"]]
    bars = ax.barh(
        [readable(f) for f in top["feature"]],
        top["coefficient"],
        color=colors, edgecolor="white", height=0.65,
    )
    # annotate
    for bar, val in zip(bars, top["coefficient"]):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8, color="black")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Standardised Coefficient", fontsize=11)
    ax.set_title("Logistic Regression Coefficients — Top Features (6-month)",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#d9534f", label="Increases rapid-progression risk"),
        Patch(color="#3274a1", label="Decreases rapid-progression risk"),
    ], loc="lower right", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "lr_coefficients")
    print(f"  [OK] LR coefficients: {len(coef_df)} features")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    ensure_dirs()

    # ── Load model & data ──
    print("Loading model and data...")
    pipe_xgb = joblib.load(MODEL_XGB_PATH)
    X_test, y_test, ids_test, X_dev, feat_cols, cutoff = load_test_data()

    # ── Predictions on TEST ──
    proba = pipe_xgb.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    print(f"  Predictions: Rapid={y_pred.sum()}, Slow={len(y_pred) - y_pred.sum()}")

    # ── Select case studies ──
    case_indices = pick_case_studies(y_test, y_pred, proba)
    print(f"  Case studies: {case_indices}")

    # Save case-study profiles
    profiles = []
    for label, idx in case_indices.items():
        row = X_test.iloc[idx].to_dict()
        row["case"] = label
        row["y_true"] = int(y_test.iloc[idx])
        row["y_pred"] = int(y_pred[idx])
        row["proba_rapid"] = float(proba[idx])
        row["subject_id"] = int(ids_test.iloc[idx])
        profiles.append(row)
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(os.path.join(OUT_TAB, "case_study_profiles.csv"), index=False)

    # ── SHAP ──
    shap_values, shap_imp, X_proc, transformed_names = run_shap(
        pipe_xgb, X_test, feat_cols, case_indices)

    # ── LIME ──
    lime_rankings = run_lime(
        pipe_xgb, X_test, X_dev, feat_cols, case_indices, y_test,
        transformed_names)

    # ── Concordance ──
    concordance_table(shap_imp, lime_rankings, case_indices, shap_values)

    # ── LR coefficients ──
    run_lr_coefficients()

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Step 8 complete in {elapsed:.1f}s")
    print(f"  Figures: {OUT_FIG}")
    print(f"  Tables:  {OUT_TAB}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

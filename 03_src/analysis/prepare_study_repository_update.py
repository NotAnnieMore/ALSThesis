"""Export the scoped supplementary release to the two existing study checkouts.

No model fits, patient tables, predictions, split membership or local explanation
weights are exported. README edits remain under each repository's own history.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "04_outputs/reviewer_followup/revised_analyses"
STUDY1 = ROOT / "study-repositories/explainable-als-progression"
STUDY2 = ROOT / "study-repositories/als-survival-imbalance-ml"


def records(name):
    frame = pd.read_csv(RESULTS / name)
    forbidden = {"subject_id", "patient_id", "y_true", "y_pred", "y_proba", "probability",
                 "native_prediction", "proba_rapid", "model_probability", "surrogate_probability"}
    if forbidden.intersection(frame.columns):
        raise ValueError(f"Patient-level columns in proposed aggregate export: {name}")
    return json.loads(frame.drop(columns=["seconds"], errors="ignore").to_json(orient="records", double_precision=15))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_tests(destination, class_names, study):
    source = (ROOT / "03_src/analysis/test_reviewer_revisions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name: ast.get_source_segment(source, node)
               for node in tree.body if isinstance(node, ast.ClassDef)}
    header = '''"""Synthetic reviewer-regression fixtures; no study models are retrained.

Exported from the dissertation's test_reviewer_revisions.py.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

'''
    header += f"ROOT = Path(__file__).resolve().parents[{2 if study == 1 else 1}]\n"
    header += ("from race_encoding_sensitivity import RACE_COLUMNS, represent_race\n" if study == 1
               else "from reviewer_temporal_cohorts import bmi_codes, extract_site, selected_before\n")
    body = "\n\n".join(classes[name] for name in class_names)
    if study == 2:
        body = body.replace('ROOT / "replication/02_scripts/step5_modeling.py"',
                            'ROOT / "02_scripts/step5_modeling.py"')
    text = header + "\n\n" + body + '\n\n\nif __name__ == "__main__":\n    unittest.main(verbosity=2)\n'
    compile(text, str(destination), "exec")
    destination.write_text(text, encoding="utf-8")


def prepare_study1():
    names = ["step8_explainability.py", "lime_categorical.py", "review_lime_categorical.py",
             "repeated_stratified_validation.py", "race_encoding_sensitivity.py"]
    for name in names:
        shutil.copy2(ROOT / "03_src/analysis" / name, STUDY1 / "03_src/analysis" / name)
    export_tests(STUDY1 / "03_src/analysis/test_supplementary_revision.py",
                 ["RaceRepresentationTests", "LimeRepresentationTests"], 1)
    parameter_files = sorted((ROOT / "04_outputs/tables/step5_tuning").glob("*_best.json"))
    assert len(parameter_files) == 21
    target = STUDY1 / "04_outputs/tables/step5_tuning"
    target.mkdir(parents=True, exist_ok=True)
    for path in parameter_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert {"model", "mode", "horizon", "full_params", "best_value"}.issubset(payload)
        assert "subject_id" not in json.dumps(payload)
        shutil.copy2(path, target / path.name)
    lime_meta = json.loads((RESULTS / "s1_lime_metadata.json").read_text(encoding="utf-8"))
    fidelity = pd.read_csv(RESULTS / "s1_lime_fidelity.csv")
    payload = {
        "analysis_date": "2026-10-24",
        "scope": "Supplementary fixed-parameter analyses; primary models and thresholds retained.",
        "repeated_validation_6m": records("s1_repeated_summary_6m.csv"),
        "repeated_validation_3m": records("s1_repeated_summary_3m.csv"),
        "repeated_validation_fit_count": 315,
        "race_encoding": {"summary": records("s1_race_summary.csv"), "fit_count": 90,
                          "test_evaluated": False,
                          "audit": json.loads((RESULTS / "s1_race_audit.json").read_text(encoding="utf-8"))},
        "lime": {key: lime_meta[key] for key in ["seed", "neighbours_per_case", "n_original_variables",
                 "n_transformed_columns", "exact_encode_decode", "original_test_predictions_reproduced",
                 "concordance_unit", "limitation"]},
    }
    payload["lime"].update({"local_weighted_r2_min": float(fidelity.local_weighted_r2.min()),
                            "local_weighted_r2_max": float(fidelity.local_weighted_r2.max())})
    write_json(STUDY1 / "assets/reviewer_followup/results.json", payload)
    shutil.copy2(ROOT / "overleaf/images/figures/study1_pipeline.png", STUDY1 / "assets/study1_pipeline.png")


def prepare_study2():
    for name in ["step2_preprocessing.py", "step5_modeling.py", "reviewer_temporal_cohorts.py",
                 "reviewer_temporal_evaluation.py"]:
        source = ROOT / "replication/02_scripts" / name
        destination = STUDY2 / "02_scripts" / name
        if name == "reviewer_temporal_evaluation.py":
            text = source.read_text(encoding="utf-8")
            assert text.count("ROOT = STUDY.parent") == 1
            text = text.replace("ROOT = STUDY.parent", "ROOT = STUDY")
            text = text.replace('ROOT / "04_outputs/reviewer_followup/revised_analyses"',
                                'ROOT / "03_outputs/reviewer_followup/revised_analyses"')
            destination.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(source, destination)
    export_tests(STUDY2 / "02_scripts/test_supplementary_revision.py",
                 ["TemporalEncodingTests", "FailureAuditTests"], 2)
    target = STUDY2 / "03_outputs/step5"
    target.mkdir(parents=True, exist_ok=True)
    configs = ROOT / "replication/03_outputs/step5/step5_best_configs.json"
    assert len(json.loads(configs.read_text(encoding="utf-8"))) == 7
    shutil.copy2(configs, target / configs.name)
    cv_path = ROOT / "replication/03_outputs/step5/step5_cv_results.csv"
    cv = pd.read_csv(cv_path)
    expected = {"classifier", "imbalance_technique", "n_trials", "best_trial", "best_params", "time_seconds"}
    expected.update(f"{prefix}_{metric}" for prefix in ["mean", "std"]
                    for metric in ["auc", "f1", "gmean", "precision", "recall", "specificity"])
    assert set(cv.columns) == expected and len(cv) == 70
    shutil.copy2(cv_path, target / cv_path.name)
    status = json.loads((RESULTS / "s2_revised_status.json").read_text(encoding="utf-8"))
    exceptions = [{key: row[key] for key in ["cohort", "classifier", "status", "phase", "reason"] if key in row}
                  for row in status if row["status"] != "completed"]
    metadata = json.loads((RESULTS / "s2_revised_metadata.json").read_text(encoding="utf-8"))
    payload = {
        "analysis_date": "2026-10-24", "primary_results_replaced": False, "new_optuna_search": False,
        "reference": records("s2_timing_reference.csv"),
        "cohorts": records("s2_revised_cohorts.csv"),
        "metrics": records("s2_revised_metrics.csv"),
        "threshold_metrics": records("s2_revised_thresholds.csv"),
        "fit_count": metadata["n_completed_fits"], "exceptions": exceptions,
        "limitations": metadata["limitations"],
        "reference_source": "Original saved probabilities, validated against the original labels and confusion matrices.",
    }
    write_json(STUDY2 / "assets/reviewer_followup/results.json", payload)
    for extension in ["png", "pdf"]:
        path = RESULTS / f"review_s2_timing_sensitivity.{extension}"
        shutil.copy2(path, STUDY2 / "assets/reviewer_followup" / path.name)
    shutil.copy2(ROOT / "overleaf/images/figures/experimental_pipeline.png", STUDY2 / "assets/experimental_pipeline.png")


def main():
    for path in [STUDY1, STUDY2]:
        if not (path / ".git").is_dir():
            raise ValueError(f"Expected an existing study checkout: {path}")
    prepare_study1()
    prepare_study2()
    print("Prepared both study checkouts: scoped code, synthetic tests, selected parameters and aggregate results.")


if __name__ == "__main__":
    main()

"""Correct two plot-display issues without changing primary model choices."""

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    evaluation = load_module("review_holdout_plot", ROOT / "03_src/evaluation/evaluate_holdout.py")
    x, y, groups, *_ = evaluation.load_data()
    params = evaluation.load_best_params("XGB", "unbalanced")
    best, thresholds, sweep, probabilities = evaluation.oof_threshold_sweep(
        "XGB", params, False, x, y, groups)
    assert np.isclose(best["f1"]["threshold"], .21)
    assert np.isclose(best["youden"]["threshold"], .35)
    assert np.isclose(best["f2"]["threshold"], .05)
    out = ROOT / "04_outputs/reviewer_followup"
    out.mkdir(parents=True, exist_ok=True)
    evaluation.OUT_FIGURES = str(out)
    evaluation.plot_threshold_sweep(thresholds, sweep, best)
    # Cache only under the ignored patient-data directory.
    pd.DataFrame({"subject_id":groups,"y_true":y,"probability":probabilities}).to_csv(
        ROOT / "01_data/interim/review_s1_oof.csv",index=False)
    (out / "review_s1_threshold_diagnostics.json").write_text(json.dumps(best,indent=2)+"\n",encoding="utf-8")
    for ext in ["pdf","png"]:
        shutil.copy2(out / f"step6_threshold_sweep.{ext}",
                     ROOT / f"overleaf/images/figures/step6_threshold_sweep.{ext}")
    xai = load_module("review_xai_plot", ROOT / "03_src/analysis/step8_explainability.py")
    coefficients = pd.read_csv(ROOT / "04_outputs/tables/step8_xai/lr_coefficients.csv")
    xai.plot_lr_coefficients(coefficients)
    print("Plot-only corrections completed; selected thresholds are unchanged.")


if __name__ == "__main__":
    main()

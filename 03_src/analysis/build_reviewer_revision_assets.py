"""Build supplementary tables/figures from completed reviewer analyses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/reviewer_followup/revised_analyses"


def table(name, caption, label, headers, rows, note):
    lines = [r"\begin{table}[ht!]", r"\centering\small", rf"\caption[{caption}]{{{caption}}}",
             rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * (len(headers)-1) + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}}", r"\par\smallskip",
                  r"\begin{minipage}{0.98\linewidth}\footnotesize", note,
                  r"\end{minipage}", r"\end{table}"])
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def study1():
    rows = []
    for horizon in ["6m", "3m"]:
        rows.append(r"\multicolumn{4}{l}{\textbf{" + ("Six months" if horizon == "6m" else "Three months") + r"}} \\")
        data = pd.read_csv(OUT / f"s1_repeated_summary_{horizon}.csv")
        for _, row in data.iterrows():
            label = row.model + (" (bal.)" if row["mode"] == "balanced" else " (unbal.)")
            rows.append([label, f"{row.original_pr_auc:.3f}",
                         f"${row.pr_auc_mean:.3f}\\pm{row.pr_auc_population_sd:.3f}$",
                         f"{row.repeat_mean_min:.3f}--{row.repeat_mean_max:.3f}"])
    table("review_s1_repeated_cv", "Study I fixed-parameter repeated validation",
          "tab:review_s1_repeated_cv", ["Configuration", "Original AP", "Repeated AP", "Repeat-mean range"], rows,
          r"AP denotes average precision. Repeated values are mean $\pm$ population SD across 15 validation folds. The range covers the three five-fold repetition means, not a confidence interval. Parameters were selected in the original development analysis and were not retuned; this is a partition-stability check, not nested validation. TEST was not accessed.")
    data = pd.read_csv(OUT / "s1_lime_fidelity.csv")
    rows = [[r.case, f"{r.local_weighted_r2:.3f}", f"{r.model_probability:.3f}",
             f"{r.surrogate_probability:.3f}", f"{r.absolute_probability_error:.3f}"] for r in data.itertuples()]
    table("review_s1_lime_fidelity", "Local fidelity of categorical-aware LIME",
          "tab:review_s1_lime_fidelity", ["Case", r"Weighted $R^2$", "Model probability", "LIME probability", "Absolute error"], rows,
          r"Probabilities refer to the rapid-progression class at the original case-study input. Weighted $R^2$ describes the surrogate's fit to sampled neighbours, not predictive accuracy. Valid categorical sampling does not guarantee clinically plausible joint combinations.")


def study1_race():
    data = pd.read_csv(OUT / "s1_race_summary.csv")
    labels = {"primary_median": "Numeric median", "explicit_recording": "Recorded states",
              "race_omitted": "Race omitted"}
    rows = [[r.model + (" (bal.)" if r.mode == "balanced" else " (unbal.)"),
             labels[r.strategy], f"${r.pr_auc_mean:.3f}\\pm{r.pr_auc_population_sd:.3f}$",
             f"${r.pr_auc_delta_mean:+.4f}$", f"{r.roc_auc_mean:.3f}", f"{r.brier_mean:.4f}"]
            for r in data.itertuples()]
    table("review_s1_race_sensitivity", "Study I race-encoding sensitivity",
          "tab:review_s1_race_sensitivity", ["Configuration", "Race coding", "PR-AUC", r"$\Delta$ AP", "ROC-AUC", "Brier"], rows,
          r"Six-month DEV only ($n=1{,}113$), with fixed parameters and identical stratified five-fold splits repeated three times. PR-AUC is mean $\pm$ population fold SD; the other scores are fold means. $\Delta$ AP is the mean paired change from the numeric-median reference. Recorded states distinguish explicit zero, explicit one and no recorded value for each indicator; all other preprocessing is unchanged. No new Optuna search, TEST evaluation, threshold selection or significance test was performed.")


def cohort_table():
    meta = json.loads((OUT / "s2_revised_metadata.json").read_text())["cohort_details"]
    train = pd.read_csv(ROOT / "replication/01_data/processed/step4_train_unscaled.csv")
    test = pd.read_csv(ROOT / "replication/01_data/processed/step4_test_unscaled.csv")
    records = []
    for name, details in meta.items():
        data = pd.read_csv(ROOT / f"replication/01_data/interim/reviewer_revision/{name}.csv")
        tr = data.loc[data.subject_id.isin(train.subject_id)]
        te = data.loc[data.subject_id.isin(test.subject_id)]
        records.append({"cohort": name, "n":len(data), "short":int(data.Survival_Group.sum()),
                        "n_train":len(tr), "train_short":int(tr.Survival_Group.sum()),
                        "n_test":len(te), "test_short":int(te.Survival_Group.sum()),
                        "test_prevalence":float(te.Survival_Group.mean()), **details})
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "s2_revised_cohorts.csv",index=False)
    labels = {"bmi_boundary_check":"BMI boundaries only", "entry_18m":r"Entry $\leq18$ months",
              "landmark_18m":"18-month landmark", "entry_12m":r"Entry $\leq12$ months",
              "landmark_12m":"12-month landmark"}
    rows = [["Original reference", "1,502", "174", "1,201", "301", "35"]]
    rows.extend([[labels[r.cohort], str(r.n), str(r.short), str(r.n_train), str(r.n_test), str(r.test_short)]
                 for r in frame.itertuples()])
    table("review_s2_timed_cohorts", "Study II supplementary cohort sizes",
          "tab:review_s2_timed_cohorts", ["Analysis", "$N$", "Positive", "Training", "TEST", "TEST positive"], rows,
          r"Original train/test membership and ordering were preserved. Entry-only rows retain the retrospective predictors and onset-based class. Landmark rows include only patients alive at that month and use dated predictors no later than that time; a positive outcome is subsequent death by month 24. The 12-month landmark has only four negative cases in total and cannot support the planned five-fold comparison.")
    return frame


def study2_tables():
    data = pd.read_csv(OUT / "s2_revised_metrics.csv")
    rows = []
    for name, label in [("entry_18m","Entry at/before 18 months: TEST 91, positives 34"),
                        ("entry_12m","Entry at/before 12 months: TEST 28, positives 22")]:
        rows.append(r"\multicolumn{5}{l}{\textbf{" + label + r"}} \\")
        subset = data.loc[data.cohort.eq(name)].set_index("classifier")
        for classifier in ["KNN","NaiveBayes","DecisionTree","RandomForest","SVM","MLP","LightGBM"]:
            if classifier not in subset.index:
                rows.append([classifier, "--", "--", "--", "--"])
                continue
            r = subset.loc[classifier]
            rows.append([classifier, f"{r.roc_auc:.3f} [{r.roc_auc_lower:.3f}, {r.roc_auc_upper:.3f}]",
                         f"{r.average_precision:.3f}", f"{r.sensitivity:.3f}", f"{r.specificity:.3f}"])
    table("review_s2_entry_restricted", "Study II entry-only sensitivity",
          "tab:review_s2_entry_restricted", ["Classifier", "ROC-AUC [95\\% CI]", "AP", "Sensitivity", "Specificity"], rows,
          r"Saved classifier/resampling configurations were refitted without new tuning. Native prediction rules were retained, including SVM's decision-function rule. Intervals use 2,000 patient resamples and condition on each fitted model. AP reference prevalences are 0.374 and 0.786. The 12-month KNN+RUS configuration failed because undersampling left 22 fitting observations for $k=31$; no fallback score was assigned. These restrictions do not bound predictor dates.")
    subset = data.loc[data.cohort.eq("landmark_18m")]
    rows = []
    for r in subset.itertuples():
        rows.append([r.classifier, f"{r.roc_auc:.3f} [{r.roc_auc_lower:.3f}, {r.roc_auc_upper:.3f}]",
                     f"{r.average_precision:.3f}", f"{r.sensitivity:.3f}/{r.specificity:.3f}",
                     f"{r.constrained_sensitivity:.3f}/{r.constrained_specificity:.3f}"])
    table("review_s2_landmark18", "Study II time-bounded 18-month landmark",
          "tab:review_s2_landmark18", ["Classifier", "ROC-AUC [95\\% CI]", "AP", r"\shortstack{Native\\sens./spec.}", r"\shortstack{OOF-constrained\\sens./spec.}"], rows,
          r"TEST contains 33 participants and five deaths between months 18 and 24; AP reference prevalence is 0.152. The alternative threshold is the highest score attaining sensitivity at least 0.90 in pooled repeated-CV development predictions. This target is not guaranteed on TEST. Complete-case selection, five test events and fixed transferred hyperparameters limit interpretation; these are not diagnosis-time predictions.")


def timing_figure(probabilities_path, boots=2000):
    sys.path.insert(0, str(ROOT / "replication/02_scripts"))
    spec = importlib.util.spec_from_file_location("temporal_plot_metrics", ROOT / "replication/02_scripts/reviewer_temporal_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probabilities = json.loads(probabilities_path.read_text())
    reference = pd.read_csv(ROOT / "replication/03_outputs/step6/step6_test_results.csv").set_index("classifier")
    arrays = np.load(ROOT / "replication/01_data/processed/step4_arrays_unscaled.npz", allow_pickle=True)
    originals = []
    for name in ["LightGBM", "MLP"]:
        d = probabilities[name]
        y, p, native = np.array(d["y_true"]), np.array(d["y_proba"]), np.array(d["y_pred"])
        assert np.array_equal(y, arrays["y_test"])
        metrics = module.metrics(y,p,native)
        assert [metrics[k] for k in ["TN","FP","FN","TP"]] == [int(reference.loc[name,k]) for k in ["TN","FP","FN","TP"]]
        originals.append({"cohort":"reference", "classifier":name, "n_test":len(y),
                          "test_short":int(y.sum()), "test_prevalence":y.mean(),
                          **metrics, **module.intervals(y,p,native,boots)})
    pd.DataFrame(originals).to_csv(OUT / "s2_timing_reference.csv",index=False)
    data = pd.concat([pd.DataFrame(originals), pd.read_csv(OUT / "s2_revised_metrics.csv")])
    variants = ["reference","entry_18m","landmark_18m","entry_12m"]
    labels = ["Reference\n301 / 35", "Entry $\\leq18$m\n91 / 34", "18m landmark\n33 / 5", "Entry $\\leq12$m\n28 / 22"]
    plt.rcdefaults()
    navy, cyan, magenta, gold = "#16324f", "#00b0be", "#f54f74", "#ffb255"
    plt.rcParams.update({"text.color":navy, "axes.labelcolor":navy,"axes.edgecolor":navy,
                         "xtick.color":navy,"ytick.color":navy,"font.size":9})
    fig, axes = plt.subplots(1,2,figsize=(10.6,4.7),sharey=True)
    for axis, metric, title in zip(axes,["roc_auc","average_precision"],["ROC-AUC","Average precision"]):
        for offset, name, colour in [(-.10,"LightGBM",cyan),(.10,"MLP",magenta)]:
            selected = data.loc[data.classifier.eq(name)].set_index("cohort").loc[variants]
            centre = selected[metric].to_numpy()
            errors = np.vstack([centre-selected[metric+"_lower"],selected[metric+"_upper"]-centre])
            axis.errorbar(np.arange(4)+offset,centre,yerr=errors,fmt="o",color=colour,
                          capsize=3,markersize=5,label="LightGBM+ROS" if name=="LightGBM" else name)
        axis.set(xticks=np.arange(4),xticklabels=labels,ylim=(0,1.04),title=title)
        axis.grid(axis="y",alpha=.15,color=navy)
        axis.spines[["top","right"]].set_visible(False)
    axes[0].axhline(.5,color=navy,ls=":",lw=1)
    prevalences = data.loc[data.classifier.eq("MLP")].set_index("cohort").loc[variants,"test_prevalence"]
    axes[1].scatter(np.arange(4),prevalences,color=gold,marker="x",s=45,label="AP prevalence reference",zorder=5)
    fig.supxlabel("Analysis: TEST participants / positive outcomes", y=.015)
    fig.suptitle("Study II: temporal sensitivity with fixed configurations",color=navy,fontweight="bold")
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles,labels_legend,loc="upper center",bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.84))
    for extension in ["pdf","png"]:
        fig.savefig(OUT / f"review_s2_timing_sensitivity.{extension}",dpi=300,bbox_inches="tight")
    plt.close(fig)


def publish():
    for path in OUT.glob("review_*.tex"):
        shutil.copy2(path, ROOT / "overleaf/images/tables" / path.name)
    for path in OUT.glob("review_s2_timing_sensitivity.*"):
        if path.suffix in [".png",".pdf"]:
            shutil.copy2(path, ROOT / "overleaf/images/figures" / path.name)
    for case in ["TP","FP","FN","TN"]:
        for extension in ["png","pdf"]:
            name = f"lime_local_{case}.{extension}"
            for folder in [ROOT / "overleaf/images/figures", ROOT / "04_outputs/figures/step8_xai"]:
                shutil.copy2(OUT/name,folder/name)
    shutil.copy2(OUT / "s1_lime_concordance.csv", ROOT / "04_outputs/tables/step8_xai/shap_vs_lime_concordance.csv")
    shutil.copy2(OUT / "s1_lime_fidelity.csv", ROOT / "04_outputs/tables/step8_xai/lime_local_fidelity.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study2-probabilities",type=Path)
    parser.add_argument("--publish",action="store_true")
    parser.add_argument("--race-only",action="store_true",
                        help="Build/publish only the new race table, preserving existing layout edits")
    args = parser.parse_args()
    if args.race_only:
        study1_race()
        if args.publish:
            name = "review_s1_race_sensitivity.tex"
            shutil.copy2(OUT / name, ROOT / "overleaf/images/tables" / name)
        print("Race sensitivity table built" + (" and published." if args.publish else "."))
        return
    if not args.study2_probabilities:
        parser.error("--study2-probabilities is required unless --race-only is used")
    study1()
    if (OUT / "s1_race_summary.csv").exists():
        study1_race()
    cohort_table()
    study2_tables()
    timing_figure(args.study2_probabilities)
    if args.publish:
        publish()
    print("Tables and figures built" + (" and published to active thesis paths." if args.publish else "."))


if __name__ == "__main__":
    main()

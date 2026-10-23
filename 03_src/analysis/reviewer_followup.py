"""Post-hoc reviewer checks; never refit models or change saved predictions.

All exported results are aggregate summaries. Patient-level inputs remain local.
Run from the repository root with the verified Study I Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest, chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
INPUTS: dict[str, str] = {}


def fingerprint(path: Path) -> None:
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    fingerprint(path)
    return pd.read_csv(path)


def write_table(out: Path, name: str, caption: str, label: str,
                headers: list[str], rows: list[list[str]], note: str = "") -> None:
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            rf"\caption[{caption}]{{{caption}}}", rf"\label{{{label}}}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
            " & ".join(headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        body.extend([r"\par\smallskip", r"\begin{minipage}{0.98\linewidth}\footnotesize",
                     note, r"\end{minipage}"])
    body.append(r"\end{table}")
    (out / f"{name}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def summary(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "--"
    q = s.quantile([0.25, 0.5, 0.75])
    return (r"\shortstack{" + f"{q.loc[0.5]:.1f} [{q.loc[0.25]:.1f}, {q.loc[0.75]:.1f}]"
            + r" \\ $n=" + str(len(s)) + "$}")


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict:
    logit = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
    def loss(beta):
        eta = beta[0] + beta[1] * logit
        return np.sum(np.logaddexp(0, eta) - y * eta)
    fit = minimize(loss, [0.0, 1.0], method="BFGS", tol=1e-7)
    # A small numerical gradient is acceptable when BFGS reports precision loss.
    eta = fit.x[0] + fit.x[1] * logit
    grad = np.array([np.sum(expit(eta) - y), np.sum((expit(eta) - y) * logit)])
    if not np.isfinite(fit.fun) or np.linalg.norm(grad) > 1e-3:
        raise RuntimeError(f"Calibration diagnostic did not converge: {fit.message}")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1])}


def study1(out: Path, boots: int) -> dict:
    base = read_csv(ROOT / "01_data/processed/features_baseline_v1.csv")
    six = read_csv(ROOT / "01_data/processed/dataset_6m_v2.csv")
    three = read_csv(ROOT / "01_data/processed/dataset_3m_v2.csv")
    pred = read_csv(ROOT / "04_outputs/tables/step8_xai/test_predictions.csv")
    split = read_csv(ROOT / "01_data/processed/holdout_split_6m.csv")
    assert len(base) == 5436 and len(six) == 1392 and len(three) == 2407
    assert len(set(three.subject_id) & set(six.subject_id)) == 720
    columns = ["Age", "Sex", "ALSFRS_R_t0", "FVC_Liters_best_t0", "FVC_pctNormal_best_t0"]
    for cohort in [three, six]:
        assert cohort.subject_id.is_unique
        current = cohort.set_index("subject_id")[columns]
        pd.testing.assert_frame_equal(base.set_index("subject_id").loc[current.index, columns],
                                      current, check_dtype=False)
    assert not pred.subject_id.duplicated().any()
    assert set(pred.subject_id) == set(split.loc[split.partition.eq("test"), "subject_id"])
    test = pred.merge(six, on="subject_id", validate="one_to_one")
    y, p = test.y_true.to_numpy(), test.proba_rapid.to_numpy()
    assert confusion_matrix(y, p >= 0.21).ravel().tolist() == [62, 133, 12, 72]
    dev = six.loc[six.subject_id.isin(split.loc[split.partition.eq("dev"), "subject_id"])]
    cutoff = dev.slope_180d_per_30d.quantile(0.30)
    assert np.array_equal(y, (test.slope_180d_per_30d <= cutoff).astype(int))
    cohorts = [base.loc[base.subject_id.isin(three.subject_id)],
               base.loc[base.subject_id.isin(six.subject_id)],
               base.loc[~base.subject_id.isin(six.subject_id)]]
    rows = [[r"Participants", *[f"{len(c):,}" for c in cohorts]]]
    for col, label in [("Age", "Age (years)"), ("ALSFRS_R_t0", "Baseline ALSFRS-R"),
                       ("FVC_Liters_best_t0", "FVC (litres)"),
                       ("FVC_pctNormal_best_t0", r"FVC (\% normal)")]:
        rows.append([label, *[summary(c[col]) for c in cohorts]])
    rows.append(["Male / recorded sex", *[f"{c.Sex.eq('Male').sum()} / {c.Sex.notna().sum()}" for c in cohorts]])
    write_table(out, "review_s1_baseline", "Study I baseline characteristics and eligibility",
                "tab:review_s1_baseline", ["Characteristic", r"\shortstack{3-month\\eligible}", r"\shortstack{6-month\\eligible}", r"\shortstack{6-month\\excluded}"], rows,
                "Continuous values are median [interquartile range]; number observed. Exclusions are relative to the 5,436 participants with a valid ALSFRS-R assessment. The two eligible cohorts overlap (720 participants).")
    baseline_tests = []
    for col in ["Age", "ALSFRS_R_t0", "FVC_Liters_best_t0", "FVC_pctNormal_best_t0"]:
        a, b = cohorts[1][col].dropna(), cohorts[2][col].dropna()
        u, pv = mannwhitneyu(a, b, alternative="two-sided")
        baseline_tests.append({"feature": col, "n_eligible": len(a), "n_excluded": len(b),
                               "p": float(pv), "rank_biserial": float(2*u/(len(a)*len(b))-1)})
    pd.DataFrame(baseline_tests).to_csv(out / "review_s1_selection_tests.csv", index=False)

    # Match the fixed cut-points used by the original supplementary script.
    groups = {
        "Age": pd.cut(test.Age, [0, 49, 57, 65, np.inf],
                      labels=["<=49", "50--57", "58--65", ">65"]).astype(object).fillna("Missing"),
        "Sex": test.Sex.fillna("Missing"),
        "ALSFRS-R": pd.cut(test.ALSFRS_R_t0, [-np.inf, 35, 39, 42, np.inf],
                           labels=["<=35", "36--39", "40--42", ">42"]),
        "FVC": pd.cut(test.FVC_pctNormal_best_t0, [-np.inf, 77, 87, 98, np.inf],
                      labels=["<=77", "78--87", "88--98", ">98"]).astype(object).fillna("Missing"),
    }
    rng = np.random.default_rng(42)
    subgroup_rows, records = [], []
    for family, strata in groups.items():
        for group in pd.unique(strata):
            mask = strata.eq(group).to_numpy()
            ys, ps = y[mask], p[mask]
            n, positives = len(ys), int(ys.sum())
            if n < 10 or positives < 3:
                continue
            indices = rng.integers(0, n, size=(boots, n))
            by, bp = ys[indices], ps[indices] >= .21
            denominators = by.sum(axis=1)
            recall = (by*bp).sum(axis=1)[denominators > 0] / denominators[denominators > 0]
            lo, hi = np.quantile(recall, [.025, .975])
            est = float(np.sum(ys*(ps >= .21))/positives)
            records.append(dict(stratification=family, group=group, n=n, positives=positives,
                                recall=est, lower=lo, upper=hi, valid_bootstraps=len(recall)))
            display = str(group).replace("<=", r"$\leq$").replace(">", r"$>$")
            subgroup_rows.append([f"{family}: {display}", str(n), str(positives),
                                  f"{est:.3f} [{lo:.3f}, {hi:.3f}]"])
    order = {
        "Age": ["<=49", "50--57", "58--65", ">65", "Missing"],
        "Sex": ["Female", "Male", "Missing"],
        "ALSFRS-R": ["<=35", "36--39", "40--42", ">42"],
        "FVC": ["<=77", "78--87", "88--98", ">98", "Missing"],
    }
    indices = sorted(range(len(records)), key=lambda i: (
        list(order).index(records[i]["stratification"]),
        order[records[i]["stratification"]].index(records[i]["group"])))
    records = [records[i] for i in indices]
    subgroup_rows = [subgroup_rows[i] for i in indices]
    pd.DataFrame(records).to_csv(out / "review_s1_subgroup_ci.csv", index=False)
    write_table(out, "review_s1_subgroup_ci", "Study I subgroup recall with bootstrap intervals",
                "tab:review_s1_subgroup_ci", ["Stratum", "$N$", "Rapid", "Recall [95\\% CI]"], subgroup_rows,
                f"Pointwise percentile intervals from {boots:,} within-stratum patient resamples at the fixed threshold 0.21. Single-class resamples without rapid progressors are omitted. These are not simultaneous intervals or tests of subgroup differences.")

    thresholds = np.round(np.arange(.01, .61, .01), 2)
    n = len(y)
    idx = rng.integers(0, n, size=(boots, n))
    by, bp = y[idx], p[idx]
    dca = []
    for t in thresholds:
        decisions = bp >= t
        nb = ((by*decisions).sum(axis=1) - ((1-by)*decisions).sum(axis=1)*t/(1-t))/n
        treat_all = by.mean(axis=1) - (1-by.mean(axis=1))*t/(1-t)
        delta = nb - treat_all
        pred_t = p >= t
        point = (np.sum(y*pred_t)-np.sum((1-y)*pred_t)*t/(1-t))/n
        lo, hi = np.quantile(nb, [.025, .975])
        dl, dh = np.quantile(delta, [.025, .975])
        dca.append(dict(threshold=t, net_benefit=point, lower=lo, upper=hi,
                        difference_vs_all=point-(y.mean()-(1-y.mean())*t/(1-t)),
                        difference_lower=dl, difference_upper=dh))
    dc = pd.DataFrame(dca)
    dc.to_csv(out / "review_s1_dca_ci.csv", index=False)
    from study1_style import apply_study1_style, PALETTE
    apply_study1_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.fill_between(dc.threshold, dc.lower, dc.upper, color=PALETTE["soft_mauve"], alpha=.2, label="Pointwise 95% interval")
    ax.plot(dc.threshold, dc.net_benefit, color=PALETTE["velvet_purple"], label="XGBoost (fixed fit)")
    ax.plot(thresholds, y.mean()-(1-y.mean())*thresholds/(1-thresholds), color=PALETTE["deep_rose"], ls="--", label="Treat all")
    ax.axhline(0, color="0.4", ls=":", label="Treat none")
    ax.axvline(.21, color="0.4", ls=":", lw=.8)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-.1, .36), xlim=(.01, .6))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "review_s1_dca_ci.pdf", bbox_inches="tight")
    fig.savefig(out / "review_s1_dca_ci.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    continuous = ["ALSFRS_R_t0", "Age", "slope_180d_per_30d", "creatinine_t0",
                  "alt_t0", "BMI_t0", "Weight_kg_t0", "n_conmeds_pre_t0"]
    observed = six.FVC_Liters_best_t0.notna()
    effects = []
    for col in continuous:
        a, b = six.loc[observed, col].dropna(), six.loc[~observed, col].dropna()
        u, pv = mannwhitneyu(a, b, alternative="two-sided")
        effects.append([col.replace("_", r"\_"), str(len(a)), str(len(b)),
                        f"{2*u/(len(a)*len(b))-1:+.3f}"])
    write_table(out, "review_s1_rank_effects", "Rank-based effects in the FVC missingness comparison",
                "tab:review_s1_rank_effects", ["Variable", "FVC observed", "FVC absent", "Rank-biserial $r$"], effects,
                "$r=2U/(n_1n_2)-1$, with the observed-FVC group first. Positive values indicate higher ranks in that group. Missing comparison values are omitted separately for each variable.")
    categorical_rows = []
    for col, label in [("Sex", "Sex"), ("study_arm", "Study arm"), ("riluzole_pre_t0", "Riluzole")]:
        table = pd.crosstab(six[col], observed)
        chi_uncorrected, _, _, _ = chi2_contingency(table, correction=False)
        v = np.sqrt(chi_uncorrected / (table.to_numpy().sum()*min(np.array(table.shape)-1)))
        categorical_rows.append([label, str(int(table.to_numpy().sum())), f"{v:.3f}"])
    write_table(out, "review_s1_cramers_v", "Uncorrected Cramer's V for FVC availability",
                "tab:review_s1_cramers_v", ["Variable", "$N$", r"Cram\'er's $V$"], categorical_rows,
                "Calculated from the uncorrected Pearson statistic. Original reported test p-values retain the Yates correction for two-by-two tables.")

    arm = pd.crosstab(six.study_arm, six.slope_180d_per_30d <= six.slope_180d_per_30d.quantile(.3))
    assert sorted(arm.sum(axis=1).tolist()) == [506, 762]
    cy, py, _, _ = chi2_contingency(arm, correction=True)
    cu, pu, _, _ = chi2_contingency(arm, correction=False)
    odds, pf = fisher_exact(arm)
    race = []
    for col in [c for c in dev if c.startswith("Race_")]:
        race.append({"feature": col, "missing_dev": int(dev[col].isna().sum()),
                     "median_dev": float(dev[col].median()), "recorded_dev": int(dev[col].notna().sum())})
    pd.DataFrame(race).to_csv(out / "review_s1_race_encoding.csv", index=False)
    null = y.mean()*(1-y.mean())
    return dict(n_test=len(y), average_precision=average_precision_score(y,p),
                brier=brier_score_loss(y,p), null_brier=null,
                scaled_brier=1-brier_score_loss(y,p)/null,
                probability_range=[float(p.min()), float(p.max())],
                calibration=calibration_diagnostics(y,p),
                treatment_tests=dict(yates_chi2=cy, yates_p=py, uncorrected_chi2=cu,
                                     uncorrected_p=pu, fisher_p=pf),
                dca_at_021=dc.loc[dc.threshold.eq(.21)].iloc[0].to_dict())


def study2(out: Path, probabilities: Path, boots: int) -> dict:
    plt.rcdefaults()
    plt.rcParams.update({"axes.labelcolor":"#16324f", "axes.edgecolor":"#16324f",
                         "axes.titlecolor":"#16324f", "axes.titleweight":"bold",
                         "xtick.color":"#16324f", "ytick.color":"#16324f"})
    fingerprint(probabilities)
    data = json.loads(probabilities.read_text(encoding="utf-8"))
    arrays_path = ROOT / "replication/01_data/processed/step4_arrays_unscaled.npz"
    fingerprint(arrays_path)
    arrays = np.load(arrays_path, allow_pickle=True)
    reference = read_csv(ROOT / "replication/03_outputs/step6/step6_test_results.csv").set_index("classifier")
    rows, metrics = [], []
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharex=True, sharey=True)
    for i, (name, d) in enumerate(data.items()):
        y, p, pred = np.array(d["y_true"]), np.array(d["y_proba"]), np.array(d["y_pred"])
        assert np.array_equal(y, arrays["y_test"]), name
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        r = reference.loc[name]
        assert [tn,fp,fn,tp] == [int(r.TN),int(r.FP),int(r.FN),int(r.TP)], name
        assert abs(roc_auc_score(y,p)-r.test_auc) <= 5.01e-5, name
        assert d["imbalance_technique"] == (r.imbalance_technique if pd.notna(r.imbalance_technique) else "None")
        brier, ap = brier_score_loss(y,p), average_precision_score(y,p)
        record = dict(classifier=name, average_precision=ap, brier=brier,
                      mean_probability=p.mean(), prevalence=y.mean(), **calibration_diagnostics(y,p))
        metrics.append(record)
        rows.append([name, f"{ap:.3f}", f"{brier:.3f}", f"{p.mean():.3f}"])
        observed, predicted = calibration_curve(y, p, n_bins=5, strategy="quantile")
        ax = axes.flat[i]
        ax.plot([0,1],[0,1], ls="--", color="0.6", lw=.8)
        ax.plot(predicted, observed, "o-", color="#00b0be", ms=4)
        ax.set(title=name, xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=.15)
    axes.flat[-1].axis("off")
    fig.supxlabel("Mean predicted probability (five quantile bins)")
    fig.supylabel("Observed Short Survivor frequency")
    fig.tight_layout()
    fig.savefig(out / "review_s2_calibration.pdf", bbox_inches="tight")
    fig.savefig(out / "review_s2_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(out / "review_s2_probability_metrics.csv", index=False)
    write_table(out, "review_s2_probability_metrics", "Study II held-out probability assessment",
                "tab:review_s2_probability_metrics", ["Classifier", "Average precision", "Brier score", "Mean probability"], rows,
                "Selected configurations are unchanged from the primary analysis. Test prevalence is 35/301 (0.116); a constant prediction equal to that prevalence has Brier score 0.103. Lower Brier scores are better. No probability recalibration was performed.")

    a = read_csv(ROOT / "replication/01_data/interim/step2_patient_features.csv")
    final = read_csv(ROOT / "replication/01_data/processed/step3_coded_unscaled.csv")
    dead = a.Event_Dead.astype(bool)
    short = dead & a.Event_Dead_Time_from_Onset.notna() & a.Event_Dead_Time_from_Onset.le(24)
    standard = (dead & a.Event_Dead_Time_from_Onset.gt(24)) | (~dead & a.Last_Visit_from_Onset.ge(24))
    censored = ~dead & a.Last_Visit_from_Onset.notna() & a.Last_Visit_from_Onset.lt(24)
    unknown = ~(short|standard|censored)
    assert [int(short.sum()),int(standard.sum()),int(censored.sum()),int(unknown.sum())] == [656,5464,1396,1967]
    eligible = a.loc[(short|standard) & ~a.Site_of_Onset.isin(["Other","Limb_and_Bulbar"])].copy()
    assert len(eligible) == 5802 and len(final) == 1502
    retained, excluded = eligible.loc[eligible.subject_id.isin(final.subject_id)], eligible.loc[~eligible.subject_id.isin(final.subject_id)]
    rows = [["Participants", f"{len(retained):,}", f"{len(excluded):,}"]]
    for col, label in [("Age_at_Onset", "Age at onset (years)"), ("Diagnosis_Delay", "Diagnostic delay (months)"),
                       ("FVC_at_Diagnosis", r"FVC (\% predicted)"), ("BMI_at_Diagnosis", "BMI")]:
        rows.append([label, summary(retained[col]), summary(excluded[col])])
    for col, value, label in [("Sex","Male","Male / recorded sex"), ("Site_of_Onset","Bulbar","Bulbar / recorded site")]:
        rows.append([label,*[f"{c[col].eq(value).sum()} / {c[col].notna().sum()}" for c in [retained,excluded]]])
    rows.append(["Short Survivors",*[str(int(short.loc[c.index].sum())) for c in [retained,excluded]]])
    write_table(out, "review_s2_selection", "Study II complete cases and excluded participants",
                "tab:review_s2_selection", ["Characteristic", "Complete cases", "Incomplete cases"], rows,
                "Continuous values are median [interquartile range]; number observed. Both groups belong to the 5,802 outcome-eligible participants after the two onset-site exclusions. Measurements retain the original closest-to-diagnosis definition.")

    # PRO-ACT defines delta zero as trial onset; it is not a verified admission date.
    timing = retained.merge(final[["subject_id","Survival_Group"]], on="subject_id", validate="one_to_one")
    timing["entry_months"] = -timing.Onset_Delta / 30.44
    timing_rows, timing_records = [], []
    for value, label in [(1,"Short Survivor"),(0,"Standard Survival")]:
        s = timing.loc[timing.Survival_Group.eq(value),"entry_months"].dropna()
        record = dict(group=label, n=len(s), q25=s.quantile(.25), median=s.median(), q75=s.quantile(.75),
                      at_or_after_24=int(s.ge(24).sum()), min=s.min(), max=s.max())
        timing_records.append(record)
        timing_rows.append([label, str(len(s)), f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]",
                            f"{s.ge(24).sum()} ({100*s.ge(24).mean():.1f}\\%)"])
    write_table(out, "review_s2_delayed_entry", "Study II symptom onset to the source-trial reference day",
                "tab:review_s2_delayed_entry", ["Outcome", "$N$", "Months: median [IQR]", "Entry $\\geq24$ months"], timing_rows,
                "Calculated as minus the recorded symptom-onset offset divided by 30.44. PRO-ACT defines day zero as trial onset; it is used here as an entry-time proxy, not a verified individual enrolment date. The audit does not correct left truncation or establish when every predictor became available.")

    positive = arrays["y_test"].astype(bool)
    lgb, svm = np.array(data["LightGBM"]["y_pred"])[positive], np.array(data["SVM"]["y_pred"])[positive]
    lgb_only, svm_only = int(((lgb==1)&(svm==0)).sum()), int(((lgb==0)&(svm==1)).sum())
    paired_p = binomtest(lgb_only, lgb_only+svm_only, .5).pvalue if lgb_only+svm_only else 1.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0,len(lgb),size=(boots,len(lgb)))
    differences = (lgb[indices]-svm[indices]).mean(axis=1)
    lower, upper = np.quantile(differences,[.025,.975])

    history = read_csv(ROOT / "01_data/raw/PROACT_ALSHISTORY.csv")
    text = history.Site_of_Onset.astype("string").str.strip()
    recognised = text.str.contains("Bulbar|Other", regex=True, na=False) | text.isin(["Onset: Limb","Limb","Onset: Spine","Spine","Spinal"])
    fallback = history.loc[text.notna() & ~recognised,"subject_id"]
    return dict(probability_metrics=metrics, timing=timing_records,
                unknown_outcomes=int(unknown.sum()), unknown_missing_onset=int(a.loc[unknown,"Onset_Delta"].isna().sum()),
                unknown_missing_death_time=int((dead[unknown]&a.loc[unknown,"Event_Dead_Time_from_Onset"].isna()).sum()),
                unknown_missing_followup=int((~dead[unknown]&a.loc[unknown,"Last_Visit_from_Onset"].isna()).sum()),
                site_fallback_unique_patients=int(fallback.nunique()),
                site_fallback_in_final=int(final.subject_id.isin(fallback).sum()),
                paired_sensitivity=dict(lgb_only=lgb_only,svm_only=svm_only,p=paired_p,
                                        difference=float((lgb-svm).mean()),lower=lower,upper=upper))


def trials(out: Path) -> dict:
    records = []
    for path in sorted((ROOT / "04_outputs/tables/step5_tuning").glob("*_trials.csv")):
        d = read_csv(path)
        records.append(dict(search=path.stem.replace("_trials",""),n=len(d),
                            zero_scores=int(d.value.eq(0).sum()), missing_scores=int(d.value.isna().sum()),
                            states=d.state.value_counts().to_dict(),best_trial_zero_based=int(d.loc[d.value.idxmax(),"number"])))
    pd.DataFrame(records).to_csv(out / "review_s1_trial_audit.csv", index=False)
    return dict(study1=records,
                study2="Stored aggregate trial means cannot establish whether an individual fold received the 0.5 failure fallback; no fold-level failure ledger is available in this workspace.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study2-probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "04_outputs/reviewer_followup")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result = {"scope":"Post-hoc checks with fixed cohorts, predictions and thresholds; no primary model refits.",
              "bootstrap_resamples":args.bootstrap,
              "study1":study1(args.output,args.bootstrap),
              "study2":study2(args.output,args.study2_probabilities,args.bootstrap),
              "trials":trials(args.output)}
    result["input_sha256"] = INPUTS
    result["packages"] = {p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","matplotlib"]}
    (args.output/"review_followup_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["study1","study2"]},indent=2),flush=True)


if __name__ == "__main__":
    main()

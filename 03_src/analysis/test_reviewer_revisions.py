"""Focused regression checks; synthetic fixtures do not train predictive models."""

import importlib.util
import hashlib
import json
import sys
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "replication/02_scripts"))
from reviewer_temporal_cohorts import bmi_codes, extract_site, selected_before
from race_encoding_sensitivity import RACE_COLUMNS, represent_race


class RaceRepresentationTests(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame(np.nan, index=range(3), columns=RACE_COLUMNS)
        frame.loc[0, "Race_Caucasian"] = 1
        frame.loc[0, "Race_Other"] = 1
        frame.loc[1, "Race_Caucasian"] = 0
        frame.loc[1, "Race_Unknown"] = 1
        frame["Age"] = [40., 50., 60.]
        return frame

    def test_missing_zero_one_and_multiple_affirmatives_remain_distinct(self):
        source = self.fixture()
        result = represent_race(source, "explicit_recording")
        self.assertEqual(result.Race_Caucasian.tolist(),
                         ["recorded_one", "recorded_zero", "unrecorded"])
        self.assertEqual(result.loc[0, "Race_Other"], "recorded_one")
        self.assertEqual(result.loc[1, "Race_Unknown"], "recorded_one")
        self.assertEqual(result.loc[2, "Race_Unknown"], "unrecorded")
        self.assertFalse(result.isna().any().any())
        pd.testing.assert_frame_equal(source, self.fixture())

    def test_omission_removes_only_race(self):
        source = self.fixture()
        result = represent_race(source, "race_omitted")
        pd.testing.assert_frame_equal(result, source[["Age"]])
        pd.testing.assert_frame_equal(represent_race(source, "primary_median"), source)

    def test_invalid_values_are_not_silently_recoded(self):
        source = self.fixture()
        source.loc[0, "Race_Asian"] = 2
        with self.assertRaisesRegex(ValueError, "Unexpected race value"):
            represent_race(source, "explicit_recording")


class TemporalEncodingTests(unittest.TestCase):
    def test_bmi_boundaries(self):
        values = pd.Series([18.49, 18.5, 24.999, 25, 29.999, 30])
        self.assertEqual(bmi_codes(values).tolist(), [0, 1, 1, 2, 2, 3])

    def test_unknown_onset_is_not_limb(self):
        self.assertTrue(pd.isna(extract_site(pd.Series({"Site_of_Onset": "unrecognised"}))))
        self.assertEqual(extract_site(pd.Series({"Site_of_Onset": "Onset: Limb"})), "Limb/Spinal")
        self.assertEqual(extract_site(pd.Series({"Site_of_Onset": "Onset: Bulbar"})), "Bulbar")
        self.assertTrue(pd.isna(extract_site(pd.Series({"Site_of_Onset": pd.NA,
                                                       "Site_of_Onset___Bulbar": pd.NA}))))

    def test_future_and_unknown_dates_excluded(self):
        records = pd.DataFrame({"subject_id": [1, 1, 1, 2],
                                "date": [-2, 6, np.nan, 5], "value": [10, 99, 88, 20]})
        cohort = pd.DataFrame({"subject_id": [1, 2], "Diagnosis_Delta": [-10, 0],
                               "landmark_delta": [5, 5]})
        selected = selected_before(records, "date", cohort, ["value"]).set_index("subject_id")
        self.assertEqual(selected.value.to_dict(), {1: 10, 2: 20})


class LimeRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from lime_categorical import CategoricalView
            cls.view_class = CategoricalView
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_exact_roundtrip_and_mutually_exclusive_categories(self):
        development = pd.DataFrame({"Age": [40., 50., 60.], "flag": [0., 1., 0.],
                                     "Sex": ["Female", "Male", "Female"]})
        prep = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), ["Age", "flag"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["Sex"]),
        ])
        processed = prep.fit_transform(development)
        view = self.view_class(prep, processed)
        np.testing.assert_array_equal(view.decode(view.encode(processed)), processed)
        other = development.iloc[[0]].assign(Sex="unseen")
        unknown = prep.transform(other)
        np.testing.assert_array_equal(view.decode(view.encode(unknown)), unknown)
        samples = np.tile(view.encode(processed)[0], (30, 1))
        rng = np.random.default_rng(42)
        for column in view.categorical_indices:
            samples[:, column] = rng.integers(0, len(view.groups[column].domain), len(samples))
        restored = view.decode(samples)
        self.assertTrue(np.isin(restored[:, 1], [0, 1]).all())
        self.assertTrue(np.isin(restored[:, 2:].sum(axis=1), [0, 1]).all())


class FailureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("failure_test_modeling", ROOT / "replication/02_scripts/step5_modeling.py")
        cls.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.module)
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))

    def test_resampling_failure_is_logged_and_raised(self):
        class BrokenSampler:
            def fit_resample(self, x, y):
                raise RuntimeError("synthetic resampling failure")
        module = self.module
        with tempfile.TemporaryDirectory(prefix="revision-tests-", dir=ROOT / "01_data/interim") as directory:
            self.assertEqual(Path(directory).resolve().parent, (ROOT / "01_data/interim").resolve())
            with patch.object(module, "OUT_DIR", directory), patch.object(module, "N_SPLITS", 2), \
                 patch.object(module, "N_REPEATS", 1), patch.object(module, "make_sampler", return_value=BrokenSampler()):
                with self.assertRaisesRegex(RuntimeError, "synthetic resampling failure"):
                    module.run_cv("Synthetic", {}, "Synthetic", np.arange(20).reshape(10, 2),
                                  np.array([0, 1] * 5), trial_number=7)
            records = [json.loads(line) for line in (Path(directory) / "step5_failures.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["phase"], "resampling")
            self.assertEqual(records[0]["trial"], 7)
            self.assertFalse((Path(directory) / "step5_fold_metrics.csv").exists())

    def test_success_ledger_has_context_and_metrics(self):
        with tempfile.TemporaryDirectory(prefix="revision-tests-", dir=ROOT / "01_data/interim") as directory, \
             patch.object(self.module, "OUT_DIR", directory):
            self.assertEqual(Path(directory).resolve().parent, (ROOT / "01_data/interim").resolve())
            self.module.record_fold("Synthetic", "None", 3, 2, "test", {"auc": .7, "f1": .4})
            frame = pd.read_csv(Path(directory) / "step5_fold_metrics.csv")
            self.assertEqual(frame.loc[0, "trial"], 3)
            self.assertEqual(frame.loc[0, "fold"], 2)
            self.assertEqual(frame.loc[0, "auc"], .7)

    def test_fit_failure_is_logged_and_raised(self):
        class BrokenModel:
            def fit(self, x, y):
                raise RuntimeError("synthetic fit failure")
        module = self.module
        with tempfile.TemporaryDirectory(prefix="revision-tests-", dir=ROOT / "01_data/interim") as directory:
            self.assertEqual(Path(directory).resolve().parent, (ROOT / "01_data/interim").resolve())
            with patch.object(module, "OUT_DIR", directory), patch.object(module, "N_SPLITS", 2), \
                 patch.object(module, "N_REPEATS", 1), patch.object(module, "make_sampler", return_value=None), \
                 patch.object(module, "make_classifier", return_value=BrokenModel()):
                with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                    module.run_cv("Synthetic", {}, "None", np.arange(20).reshape(10,2), np.array([0,1]*5))
            record = json.loads((Path(directory) / "step5_failures.jsonl").read_text())
            self.assertEqual(record["phase"], "fit_predict_score")

    def test_valid_metrics_unchanged(self):
        result = self.module.compute_fold_metrics(np.array([0,0,1,1]),
                                                  np.array([0,0,0,1]), np.array([.1,.4,.3,.9]))
        self.assertEqual(result['auc'], .75)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['specificity'], 1.)

    def test_undefined_auc_has_no_half_score_fallback(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                self.module.compute_fold_metrics(np.ones(3), np.ones(3), np.array([.1,.5,.9]))


class CompletedAnalysisTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "04_outputs/reviewer_followup/revised_analyses/s1_race_metadata.json").exists(),
                         "Requires locally completed race sensitivity")
    def test_race_sensitivity_preserves_reference_and_uses_no_test(self):
        folder = ROOT / "04_outputs/reviewer_followup/revised_analyses"
        meta = json.loads((folder / "s1_race_metadata.json").read_text())
        self.assertEqual(meta["n_fits"], 90)
        self.assertFalse(meta["test_evaluated"])
        self.assertFalse(meta["hyperparameters_retuned"])
        self.assertFalse(meta["primary_results_replaced"])
        self.assertTrue(meta["original_repeated_scores_reproduced"])
        for path, expected in meta["input_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)
        scores = pd.read_csv(folder / "s1_race_folds.csv")
        self.assertEqual(len(scores), 90)
        self.assertTrue(scores.groupby(["model", "strategy"]).size().eq(15).all())
        for (_, _, _), group in scores.groupby(["model", "repeat", "fold"]):
            self.assertEqual(group.target_cutoff.nunique(), 1)
            self.assertEqual(group.validation_positive.nunique(), 1)
        summary = pd.read_csv(folder / "s1_race_summary.csv")
        for (model, strategy), group in scores.groupby(["model", "strategy"]):
            stored = summary.loc[summary.model.eq(model) & summary.strategy.eq(strategy)].iloc[0]
            self.assertAlmostEqual(stored.pr_auc_mean, group.pr_auc.mean(), places=12)
        coefficients = pd.read_csv(folder / "s1_race_lr_coefficients.csv")
        self.assertTrue(coefficients.loc[coefficients.strategy.eq("explicit_recording"), "coefficient"].eq(0).all())
        published = (ROOT / "overleaf/images/tables/review_s1_race_sensitivity.tex").read_text()
        for row in summary.itertuples():
            self.assertIn(f"${row.pr_auc_mean:.3f}\\pm{row.pr_auc_population_sd:.3f}$", published)
            self.assertIn(f"${row.pr_auc_delta_mean:+.4f}$", published)

    @unittest.skipUnless((ROOT / "04_outputs/reviewer_followup/revised_analyses/s2_revised_metrics.csv").exists(),
                         "Requires locally completed supplementary analyses")
    def test_temporal_metrics_and_thresholds_reproduce_saved_predictions(self):
        from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
        folder = ROOT / "04_outputs/reviewer_followup/revised_analyses"
        private = ROOT / "replication/01_data/interim/reviewer_revision"
        scores = pd.read_csv(folder / "s2_revised_metrics.csv")
        thresholds = pd.read_csv(folder / "s2_revised_thresholds.csv")
        original_test = pd.read_csv(ROOT / "replication/01_data/processed/step4_test_unscaled.csv")
        original_train = pd.read_csv(ROOT / "replication/01_data/processed/step4_train_unscaled.csv")
        for row in scores.itertuples():
            predictions = pd.read_csv(private / f"predictions_{row.cohort}_{row.classifier}.csv")
            cohort = pd.read_csv(private / f"{row.cohort}.csv").set_index("subject_id")
            expected_ids = original_test.loc[original_test.subject_id.isin(cohort.index), "subject_id"]
            np.testing.assert_array_equal(predictions.subject_id, expected_ids)
            self.assertTrue(set(predictions.subject_id).isdisjoint(original_train.subject_id))
            np.testing.assert_array_equal(predictions.y_true, cohort.loc[expected_ids, "Survival_Group"])
            y = predictions.y_true.to_numpy()
            probability = predictions.probability.to_numpy()
            self.assertTrue(np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all())
            for metric, actual in [("roc_auc", roc_auc_score(y, probability)),
                                   ("average_precision", average_precision_score(y, probability)),
                                   ("brier", brier_score_loss(y, probability))]:
                self.assertAlmostEqual(getattr(row, metric), actual, places=12)
            rules = thresholds.loc[thresholds.cohort.eq(row.cohort) & thresholds.classifier.eq(row.classifier)]
            self.assertEqual(len(rules), 5)
            for rule in rules.itertuples():
                predicted = (predictions.native_prediction.to_numpy() if rule.rule == "native"
                             else probability >= rule.threshold)
                tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
                self.assertEqual([rule.TN, rule.FP, rule.FN, rule.TP], [tn, fp, fn, tp])
                self.assertAlmostEqual(rule.sensitivity, tp / (tp + fn), places=12)
                self.assertAlmostEqual(rule.specificity, tn / (tn + fp), places=12)
            constrained = rules.loc[rules.rule.eq("sensitivity_0_90")].iloc[0]
            self.assertAlmostEqual(row.sensitivity_threshold, constrained.threshold, places=12)
            self.assertAlmostEqual(row.constrained_sensitivity, constrained.sensitivity, places=12)
            self.assertAlmostEqual(row.constrained_specificity, constrained.specificity, places=12)
        meta = json.loads((folder / "s2_revised_metadata.json").read_text())
        for path, expected in meta["input_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "replication" / path).read_bytes()).hexdigest(), expected)

    @unittest.skipUnless((ROOT / "04_outputs/reviewer_followup/revised_analyses/s1_repeated_folds_6m.csv").exists(),
                         "Requires locally completed supplementary analyses")
    def test_repeated_folds_preserve_training_only_cutoffs(self):
        for horizon, count in [("6m", 195), ("3m", 120)]:
            scores = pd.read_csv(ROOT / f"04_outputs/reviewer_followup/revised_analyses/s1_repeated_folds_{horizon}.csv")
            self.assertEqual(len(scores), count)
            membership = pd.read_csv(ROOT / f"01_data/interim/reviewer_revision/s1_repeated_folds_{horizon}.csv")
            data = pd.read_csv(ROOT / f"01_data/processed/dataset_{horizon}_v2.csv").set_index("subject_id")
            split = pd.read_csv(ROOT / f"01_data/processed/holdout_split_{horizon}.csv")
            development = set(split.loc[split.partition.eq("dev"), "subject_id"])
            self.assertEqual(set(membership.subject_id), development)
            slope = "slope_180d_per_30d" if horizon == "6m" else "slope_90d_per_30d"
            for (repeat, fold), block in scores.groupby(["repeat", "fold"]):
                validation = set(membership.loc[membership.repeat.eq(repeat) & membership.fold.eq(fold), "subject_id"])
                training = development - validation
                cutoff = data.loc[list(training), slope].quantile(.30)
                self.assertTrue(np.allclose(block.target_cutoff, cutoff, rtol=0, atol=1e-12))
                self.assertTrue((block.n_train == len(training)).all())
                self.assertTrue((block.n_validation == len(validation)).all())

    @unittest.skipUnless((ROOT / "04_outputs/reviewer_followup/revised_analyses/s2_revised_metrics.csv").exists(),
                         "Requires locally completed supplementary analyses")
    def test_temporal_results_and_infeasibility_are_consistent(self):
        folder = ROOT / "04_outputs/reviewer_followup/revised_analyses"
        scores = pd.read_csv(folder / "s2_revised_metrics.csv")
        self.assertEqual(len(scores), 27)
        self.assertFalse(scores.duplicated(["cohort", "classifier"]).any())
        self.assertTrue((scores.TP + scores.FN == scores.test_short).all())
        self.assertTrue((scores.TP + scores.FP + scores.FN + scores.TN == scores.n_test).all())
        self.assertFalse(((scores.cohort == "entry_12m") & (scores.classifier == "KNN")).any())
        self.assertFalse(scores.cohort.eq("landmark_12m").any())
        cohorts = pd.read_csv(folder / "s2_revised_cohorts.csv").set_index("cohort")
        self.assertEqual(int(cohorts.loc["landmark_12m", "n"]), 41)
        self.assertEqual(int(cohorts.loc["landmark_18m", "test_short"]), 5)
        meta = json.loads((folder / "s2_revised_metadata.json").read_text())
        self.assertEqual(meta["n_completed_fits"], 432)
        self.assertFalse(meta["new_optuna_search"])
        self.assertFalse(meta["original_results_replaced"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

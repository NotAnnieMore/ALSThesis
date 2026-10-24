# Results of the Methodological Follow-Up

Analyses performed on 24 October 2026 under `ANALYSIS_PLAN_20261024.md`.
The author chose to retain the original Study II replication reference and add
the revised analyses separately. No new Optuna search or clean-environment
reproduction was performed.

## Study I Repeated Validation

Three repetitions of stratified five-fold CV evaluated 13 six-month and eight
three-month configurations, giving 315 completed fixed-parameter fits. DEV
membership is unchanged. Provisional labels allocate folds; each training fold
still supplies its own target cutoff and preprocessing estimates. TEST is not
used. Selected parameters came from the original development folds, so this is
partition sensitivity, not nested validation.

- Six-month RF balanced: AP 0.421598, population fold SD 0.033908.
- Six-month RF unbalanced: AP 0.414655, SD 0.038997.
- Six-month XGBoost unbalanced: AP 0.414495, SD 0.033131.
- Three-month XGBoost unbalanced: AP 0.416141, SD 0.034119.
- Original selected XGBoost TEST AP remains 0.456462; threshold remains 0.21.

The leading rank is not stable. No new significance test treats overlapping folds
as independent replications, and no replacement model is selected using TEST.

## Study I Race-Encoding Sensitivity

An additional author-approved check preserves the primary representation and
uses only the six-month DEV partition. It compares numeric median imputation,
explicit per-indicator recorded-zero/recorded-one/unrecorded categories, and
omission of all seven indicators. The XGBoost-unbalanced and LR-balanced saved
configurations use the same 15 repeated stratified folds, giving 90 fits. Target
cutoffs and all fitted preprocessing remain training-fold estimates. No TEST
evaluation, new Optuna, threshold selection or new significance test is performed.

| Configuration | Numeric median AP | Recorded states AP | Race omitted AP |
| --- | ---: | ---: | ---: |
| XGBoost unbalanced | 0.414495 | 0.415943 | 0.417128 |
| LR balanced | 0.397072 | 0.397376 | 0.397381 |

The median references reproduce the previous repeated-CV AP and ROC-AUC scores
within absolute tolerance 1e-12. Input data, splits, saved parameters, primary
test predictions and LR coefficients retain their SHA-256 hashes. The DEV audit
matches the raw source: 59 patients have all seven fields unrecorded, while three
have an affirmative Unknown indicator. Missing fields are not relabelled as zeros.

All explicit-state LR race coefficients are zero across the 15 folds. The numeric
Black/African-American indicator is non-zero in 14 folds, ranging from -0.106981
to zero. Different scaling and dummy representation change how the fixed penalty
acts; this is not evidence that race is uninformative or a fairness assessment.
Ninety compatibility/deprecation warnings from the 45 LR fits are recorded; no
race-analysis fit fails. The primary pipeline and all explanation figures remain
unchanged. New aggregate outputs use the `s1_race_*` prefix.

## Categorical-Aware LIME

The saved XGBoost configuration was refitted on DEV only. All stored test
probabilities reproduce within absolute tolerance 1e-7; the confusion matrix and
four original case-study identifiers are unchanged. Exact encode/decode checks
pass for DEV and TEST. One-hot groups are sampled as original categorical
variables; binary/discrete fields retain observed categories. Classifier inputs
therefore retain valid categorical groups rather than continuous dummy values.

SHAP contributions are summed within original categorical variables before
absolute ranking, matching the units of the 35-variable LIME representation.
TP/FP/FN/TN top-ten overlaps are 5/7/6/7. Local weighted R2 values are
0.326/0.198/0.226/0.456. For FN, the surrogate probability is 0.247 versus model
probability 0.169, crossing threshold 0.21. This is reported as limited fidelity,
not concealed by the corrected representation or feature overlap.

Only the four LIME plots, concordance and fidelity outputs replace their earlier
explanation counterparts. Global/local SHAP figures, LR coefficients, original
test predictions and primary classification metrics are not replaced.

## Study II Timing and Encoding

All supplementary fits reuse the seven saved classifier/resampling configurations
and original train/test membership/order. Thresholds use the original repeated
development-CV protocol, not TEST. Scaling and resampling are fitted inside each
training fold. No tuning is repeated.

| Task | N | Training | TEST | TEST positive |
| --- | ---: | ---: | ---: | ---: |
| Original reference / BMI boundary check | 1,502 | 1,201 | 301 | 35 |
| Entry proxy at/before 18 months | 424 | 333 | 91 | 34 |
| Time-bounded 18-month landmark | 161 | 128 | 33 | 5 |
| Entry proxy at/before 12 months | 124 | 96 | 28 | 22 |
| Time-bounded 12-month landmark | 41 | 32 | 9 | 8 |

The entry-only tasks retain retrospective predictors. The landmark tasks include
observed patients alive at that time and dated predictors no later than the
landmark; their outcome is subsequent death by month 24. At month 18, 344 are
initially eligible and 161 remain complete. At month 12, 106 are initially eligible
and 41 remain complete. Missingness/recruitment selection remains; these are not
diagnosis-time predictions or a delayed-entry survival model.

| Task | LightGBM ROC-AUC | MLP ROC-AUC |
| --- | ---: | ---: |
| Original reference | 0.923 | 0.918 |
| Entry at/before 18 months | 0.828 | 0.823 |
| 18-month landmark | 0.693 | 0.764 |
| Entry at/before 12 months | 0.576 | 0.765 |

The differences do not isolate the causal effect of left truncation: population,
class balance, refitted models and, for landmarking, the prediction question
change. The 18-month landmark has only five positive test outcomes. Its LightGBM
ROC-AUC interval is [0.507, 0.863] and MLP's [0.600, 0.910]. Native detection is
3/5 and 2/5; the OOF sensitivity thresholds detect all five but reduce specificity
to 0.321 and 0.107. These are not validated clinical operating points.

The 12-month entry restriction has TEST prevalence 0.786. LightGBM AP 0.841 and
MLP AP 0.941 must be read against that baseline, not compared directly with the
original AP estimates. KNN+RUS fails in its first fold because only 22 observations
remain for k=31. The 12-month landmark has fewer than five development negatives
and is infeasible for the predeclared five-fold protocol. Neither receives a
fallback score or a post-hoc change of hyperparameters/folds.

Twenty-seven classifier/cohort combinations completed 15 development folds and a
final training fit, totalling 432 completed evaluations. One final MLP fit in the
18-month entry-restricted cohort emitted a convergence warning at the retained
1,000-iteration limit. The warning and other library warnings are saved explicitly.

BMI boundary correction changes two original codes; LightGBM/MLP reference ranking
and native confusion matrices remain unchanged at the reported precision.
Landmark riluzole is dated past use; no known dated record means no recorded use,
not confirmed non-treatment. Unknown onset text now remains missing; the current
extract had no records taking the old default. No reliable clinical-trial ID was
found; measurement `Test_trial` and treatment `Study_Arm` are not trial membership.

## Output and Privacy

Aggregate tables, figures, metadata, warnings and status records are in
`revised_analyses/`. Patient-level data, predictions, fold assignments and the
case-specific LIME weights remain in ignored `01_data/interim/` directories.
The revised Step 5 code records future trials/folds and stops on errors, without
claiming to recover missing historical evidence. A local pre-analysis snapshot
remains outside the Git worktree. Subsequent repository publication is recorded
separately in `REPOSITORY_SYNC_20261024.md`; it did not repeat model training.

## Verification

The expanded focused suite passes 16 tests in the Study I runtime. The Study II runtime
passes the 15 applicable tests and skips the LIME-only fixture because LIME is
not installed there. The checks cover category reconstruction, future/unknown-date
exclusion, BMI boundaries, unknown onset, failure propagation/logging, valid metric
behaviour, training-only CV cutoffs and consistency of completed/infeasible results.
All 27 temporal classifier/cohort summaries and their 135 threshold rows reproduce
the saved patient predictions. Original test membership/order and source labels
are checked, together with the original Study II input hashes. Race tests check
three-state preservation, omission, invalid-value rejection and primary-score
reproduction.

Input hashes for both studies and all saved configurations remain unchanged.
Original test predictions, case profiles, SHAP importance and LR coefficient tables
match the pre-analysis snapshot byte for byte. Active LIME and timing figure copies
match their generated counterparts in both PDF and PNG. Patient-level outputs are
confirmed Git-ignored. Static thesis checks resolve 285 labels and 49 cited entries
across 40 active TeX files. Abstract counts including titles are 138 EN and 147 PT.
The current author's page controls are compared with a fresh pre-edit snapshot,
not the older Git commit. All pre-existing external table and figure files match
that snapshot byte for byte; only six chapters and one new race table change in
this pass. Study II report-item closure is mapped in `STUDY2_CLOSURE.md`.
No local LaTeX compilation or whole-document pagination check was performed.

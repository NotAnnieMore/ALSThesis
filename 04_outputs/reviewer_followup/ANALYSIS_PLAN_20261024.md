# Supplementary Reviewer Analyses

Plan fixed on 24 October 2026, before the new performance evaluations.

## Scope and Preservation

The author approved retaining the original Study II replication and adding the
revised analyses separately. Original data partitions, tuning histories, primary
test results and thresholds remain reference artefacts. No new Optuna search or
clean-environment reproduction is planned. A local pre-analysis ZIP snapshot is
stored outside the Git worktree.

## Study I: Repeated Stratified Stability Check

- Reuse the original DEV identifiers and saved hyperparameters for all 13 six-month
  and eight three-month configurations.
- Use stratified five-fold CV repeated three times, seed 42. A provisional DEV
  30th-percentile label is used only for fold allocation. Recompute the operative
  target cutoff from each fold's training slopes, then apply it to validation.
- Fit imputation, encoding, scaling and class weights within each training fold.
- Do not use TEST or replace the original selected model/threshold.
- Report fold and repetition summaries as a partition-stability check of already
  selected configurations, not independent or nested validation. Do not treat the
  overlapping folds as independent observations for new significance claims.
- Log errors and abort instead of substituting a metric for a failed fit.

## Study I: Race-Encoding Sensitivity (Additional Author Approval)

Plan added before evaluating the new encodings. Preserve all primary results and
the current LIME. Use only the original six-month DEV partition and the same
three repetitions of stratified five-fold CV described above. Retain the saved
unbalanced XGBoost and balanced LR parameters; do not repeat Optuna.

Compare (1) primary numeric median imputation; (2) each race indicator as an
explicit three-state categorical field, distinguishing recorded zero, recorded
one and no recorded value; and (3) omission of all seven race indicators. Keep
multiple affirmative indicators and the separate Unknown indicator. Do not
interpret a missing checkbox as an absent identity or force one race category.
Audit the DEV fields against raw demographics before fitting. Fit all remaining
preprocessing and the target cutoff on each training fold. Check that the median
reference reproduces the earlier repeated-validation scores. Report paired
descriptive changes in average precision, ROC-AUC and Brier score, without new
significance tests or a preferred encoding selected from TEST. Do not evaluate
TEST, change the primary threshold, or regenerate explanation figures.

## Study I: Categorical-Aware LIME

- Refit only the saved XGBoost configuration on the original DEV partition if a
  verified fitted DEV artefact is unavailable. Verify predictions against the
  saved test probabilities and preserve the same four case-study identifiers.
- Explain continuous fields and categorical/binary fields in a representation
  that reconstructs the exact classifier input, keeping one-hot groups valid.
- Keep 2,000 neighbours per case and seed 42 for the displayed explanations.
- For SHAP/LIME concordance, compare the same original-variable units. Sum SHAP
  contributions within a one-hot group before ranking their absolute values.
- Record local surrogate fidelity and check exact encode/decode reconstruction.
  Correct categorical sampling does not validate clinical plausibility or causality.

## Study II: Temporal and Encoding Sensitivity

- Keep the original 1,502-patient complete-case analysis as the replication
  reference. Retain its train/test assignments in restricted-cohort comparisons.
- Evaluate predeclared onset-to-source-trial-reference limits of 18 and 12 months.
  This timestamp is an entry-time proxy, not a verified individual enrolment date.
  Before fitting, the original complete-case subset counts were 424/165 Short at
  18 months and 124/104 Short at 12 months.
- Reuse the saved seven classifier/resampling configurations without retuning.
  Fit scaling and resampling inside each training fold. Derive any alternative
  threshold from development predictions only.
- Audit when the selected predictors were available. Distinguish an entry-only
  restriction from a genuinely time-bounded predictor analysis; do not describe
  the former as resolving every temporal bias or left truncation.
- Evaluate codification changes separately where feasible, so they are not
  confused with a change of population. Investigate trial identifiers, but do not
  infer trial identity from a measurement-trial number or missingness pattern.
- Record convergence warnings and failures explicitly. Do not fabricate a score
  for an infeasible resampling/model combination or erase a failed attempt.
- Include cohort-specific prevalence and uncertainty, particularly for the very
  small 12-month subset. None of these analyses constitutes external validation.

### Landmark Protocol, Fixed Before Performance Evaluation

In addition to entry-only restrictions, construct conditional prediction tasks at
12 and 18 months after symptom onset. Retain participants whose trial-reference
proxy is no later than the landmark and who are alive at that time. Their outcome
is subsequent death by month 24, giving a common future horizon within each task.
Use only dated ALSFRS, FVC, weight, height and riluzole records at or before the
landmark. Closest-to-diagnosis selection is retained within that permissible past;
static onset/demographic history is assumed available, and diagnosis must already
have occurred. Unknown dates are not silently treated as day zero.

Use WHO-aligned left-closed BMI intervals in the landmark tasks; retain the
documented age/delay/slope conventions. Independently compare BMI-boundary changes
on the original cohort, so population change is not mistaken for a binning effect.
No record of timely riluzole use means no recorded timely use, not proven absence
of treatment; this assumption is reported. Onset-site text has no default Limb
classification for unrecognised patterns.

Preserve original train/test membership and ordering. Use the original five-fold
CV repeated three times, without new tuning. Derive thresholds from concatenated
development-fold probabilities as in the original protocol. If either development
class has fewer than five patients, report the cohort as infeasible for the
predeclared validation rather than changing the number of folds after inspection.

Before fitting, dated complete-case reconstruction yielded 161 patients (38 later
deaths by month 24) at month 18, and 41 (37 later deaths) at month 12. The latter
has fewer than five Standard Survival cases even before splitting and therefore
cannot support the planned five-fold comparison. Report this feasibility limit;
do not manufacture a performance estimate for that task.

## Privacy and Integration

Patient-level predictions, fold assignments and reconstructed data remain under
the ignored `01_data/interim/` directories. Only aggregate tables, plots, checks
and metadata are exported. Manuscript edits follow completed checks, preserve the
original results as reference, and use British English and the established style.

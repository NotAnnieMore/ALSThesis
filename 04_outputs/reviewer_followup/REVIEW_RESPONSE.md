# Reviewer Follow-Up: Scope and Evidence

## Scope

This revision implements textual/documentation corrections and supplementary
analyses under the original study definitions. It does not replace either primary
cohort, rerun Optuna, change the original CV schemes, change predictive encodings,
or replace selected models or operating thresholds. Results remain internal,
retrospective evaluations, not clinical validation.

The pre-revision checkpoint is commit `a3689e7` in the main repository. Revision
commits and publication status are recorded in the repositories' Git histories.

## Factual Corrections

All 25 rows in the report's red-correction table were addressed in the active
dissertation source. These include the Study II optimisation objective, DREAM
citation, selected-family descriptions, reliability-diagram interpretation,
continuity correction, population SD convention, reference threshold, coverage and
column counts, default-threshold SVM behaviour, LR coefficient description, PDP
range, height anomaly, exact category intervals, patient-grouping rationale,
SVM prediction rule, rounded significance legend, sensitivity target shortfalls,
final-implementation wording, FVC false-negative count, cohort wording, and the
Portuguese abstract's six-month result.

The four misleading bibliography keys were renamed consistently in the active
sources. Primary references were added for staging, ENCALS, PRO-ACT, XGBoost and
the resampling methods, alongside Brier and TRIPOD+AI references. Unused bibliography
entries were not cited merely to make them appear in the reference list.
All eleven newly added journal/conference DOIs were checked against Crossref and
returned the expected records. MiToS is cited by its 2015 journal issue; its first
online publication was in 2013. The NearMiss workshop contribution and Berkeley
Balanced Random Forest technical report use their primary institutional links.

The unsupported count of 23 source trials was removed rather than replaced with
the current website's trial count, which would not identify this historical extract.
The original two raster flow diagrams were retained. Their captions now specify
descriptive cross-horizon comparison, LR explanations, non-clinical review flags,
both exclusion counts and cohort overlap. Editing those labels inside the images
remains a separate graphical refinement.

## Main Suggestions

| Report item | Action and evidence |
| --- | --- |
| 1. Delayed entry | Audited the final Study II cohort using minus Onset_Delta / 30.44. PRO-ACT defines zero as source-trial onset, so the text calls this an entry-time proxy, not a verified individual enrolment date. Medians: 11.2 months in Short Survivors and 26.1 in Standard Survival. 756/1,328 Standard Survival patients are at or beyond 24 months. Left truncation is named and discussed; no entry-restricted cohort or survival redesign was fitted. |
| 2. Trial clustering | Both methods/results and the conclusion now distinguish patient separation from source-trial separation. No reliable trial-membership variable was established, and missingness patterns were not asserted to be trial identifiers. No leave-trial-out result is claimed. |
| 3. Bagging interpretation | Repeated the original 15 development folds with saved baseline parameters, comparing single models with ten-estimator BalancedBagging. All 14 original balanced-accuracy means reproduce to numerical precision. ROC-AUC improves for KNN/DT, is almost unchanged for NB, and falls for RF/SVM/MLP/LightGBM. The ten-configurations versus ten-bags protocol difference is explicit. |
| 4. Calibration | Reused validated, locally saved test probabilities for all seven selected Study II configurations. Added average precision, Brier scores, means and five-quantile-bin reliability plots. LightGBM+ROS: AP 0.704, Brier 0.1034, mean probability 0.260 versus prevalence 0.116. TreeSHAP's oversampled-model log-odds scale is explicit. No recalibration was used. |
| 5. Study I CV | Explained the limitations of five unstratified folds, prevalence variation and overlapping CV training samples. No claim of equivalence with a particular sorted/unshuffled KFold partition was adopted. Repeated stratified tuning was not substituted for the original analysis. |
| 6. Selection/baseline tables | Added Study I three-month, six-month and six-month-excluded summaries, and Study II complete-case versus incomplete-case summaries. Removed the unsupported inference that attrition alone demonstrates selection for slower progressors. |
| 7. Encoding | Disclosed numeric median imputation of race indicators. Actual full-DEV medians are one for Caucasian **and Other**, not zero for every non-Caucasian field. Other has only 22 recorded values. The onset-text default was audited: no source record used it in this extract. Riluzole has no time restriction in Study II; exact BMI/age/delay/slope bins are reported without changing fitted features. |
| 8. Search audit | Checked all 780 six-month and 480 three-month Study I trials: no zero/missing objective, all COMPLETE. Discussed boundary solutions and non-binding LightGBM parameters. Study II's stored aggregate means cannot establish the absence of individual failed folds scored as 0.5; this is now stated. The unused pruner is documented as inactive. |
| 9. Reproducibility | Captured both currently inspected runtimes with exact version inventories; documented the lack of a historical freeze and the original row-order/near-tie split caveat. Prepared pinned core requirements, complete snapshots, standalone supplementary analysis scripts and README updates for both study repositories. These do not constitute an independent clean-environment reproduction of the original experiments. |
| 10. Reporting/literature | Added primary method/staging citations and a retrospective 27-main-item TRIPOD+AI coverage map, explicitly not full compliance or a risk-of-bias certification. Added the author's confirmed absence of dedicated funding/conflicts, planning without public preregistration, and the distinction from the source trials' declarations. |

## Additional Checks

- Study I net benefit at 0.21: 0.131, pointwise 95% bootstrap interval
  [0.074, 0.196]. Paired difference from treat all: 0.016, interval
  [-0.015, 0.043]. The apparent improvement at that point is uncertain.
- Added within-stratum recall bootstrap intervals without changing cut-points.
  The mildest ALSFRS-R group's recall is 0.545 [0.231, 0.857]. Perfectly
  classified small strata can have a degenerate ordinary-bootstrap interval;
  this limitation is explicit.
- Study I scaled Brier score: 0.064. Descriptive logistic calibration intercept:
  0.249; slope: 1.166. These were not used to recalibrate or re-evaluate predictions.
- Added rank-biserial effects and conventional uncorrected Cramer's V alongside
  the original missingness statistics. Original corrected values are identified.
- Treatment-arm comparison: Yates p=0.057; uncorrected p=0.049; Fisher p=0.052.
  Interpretation no longer depends on choosing one side of 0.05.
- LGBM versus SVM sensitivity, paired on 35 Short Survivors: one detected only by
  LGBM, two only by SVM; exact McNemar p=1.000. Difference -0.029, paired bootstrap
  interval [-0.143, 0.057]. This is not evidence of equivalence.
- Recomputed the 1,967 unknown Study II outcomes from the source intermediate:
  1,801 lack onset offset; 888 have recorded death without a computable onset-to-death
  interval and 1,079 have no recorded death and no computable follow-up interval.
  These are not counts borrowed from the reference cohort.
- Recorded 25 zero coefficients out of 42 transformed LR variables. Corrected the
  coefficient value-label overlap by replotting saved coefficients, not refitting LR.
- Masked undefined precision in the original OOF threshold curve. Five fixed-parameter
  DEV fits reproduced the same F1 (0.21), F2 (0.05) and Youden (0.35) choices.
- LIME's original non-categorical perturbation of binary fields is disclosed, not
  silently replaced with new explanations or a new concordance result.
- Added glossary entries for IQR/TRIPOD+AI and a latexmk glossary build rule.
- The saved SHAP-ranking comparison gives rho=0.389328 and p=0.066321. The thesis's
  p=0.066 was retained; the report's p=0.067 was not copied over it.

## Verification and Limits

`review_validation.json` records active-source checks: no undefined/duplicate
labels, undefined citations or glossary uses, missing active graphics, or unbalanced
braces. Existing explicit page-break controls were preserved. Abstract counts,
including titles and excluding keywords, are 136 EN and 145 PT.

The new graph exports were visually inspected. No LaTeX compiler was available in
the local toolchain; a full Overleaf compilation and pagination check have not been run.
The standalone Study I and Study II audit exports were executed against the
authorised local study directories. Their numeric summaries reproduce this
revision's saved checks (relative tolerance 1e-6, absolute tolerance 1e-7).
Exported analysis functions were compared structurally with the canonical source;
the bagging script differs only in standalone paths. This packaging check does not
constitute an independent reproduction of the primary model development.
Unused duplicate-label table files remain outside the active input graph; they were
not deleted. No patient-level predictions or source data are included in this output
directory. Only aggregate results, figures, input-file hashes and version metadata
are exported.

## Deliberately Not Re-Run

Entry-restricted or delayed-entry survival models, trial-level validation, a changed
Study I CV design, wider/new Optuna searches, alternative progression target
percentiles, changed category/race/riluzole encodings, and categorical-aware LIME
refits are not presented as completed work. These would alter the original design,
training inputs, or explanations and require a separately agreed analysis plan.

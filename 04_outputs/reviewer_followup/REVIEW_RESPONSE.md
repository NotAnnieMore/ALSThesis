# Reviewer Follow-Up: Scope and Evidence

## Scope

The initial revision implemented textual/documentation corrections and aggregate
audits under the original study definitions. On 24 October 2026 the author also
approved categorical-aware LIME, repeated stratified Study I validation without
Optuna, and Study II temporal/encoding sensitivity. The original primary results
remain the reference; revised cohorts and conditional prediction tasks are reported
separately. No new Optuna search, clean-environment reproduction or test-informed
replacement of the original selected model/threshold was performed.

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
The two raster flow diagrams retain their original design and dimensions. On
24 October 2026, their internal labels were brought into agreement with the
captions: descriptive cross-horizon comparison, LR coefficients, non-clinical
review flags, both exclusion counts and cohort overlap. The overlap note no
longer sits on the selection path, and the shared exclusion note has no arrow
from the six-month cohort. Numerical cohort and split values are unchanged.

## Main Suggestions

| Report item | Action and evidence |
| --- | --- |
| 1. Delayed entry | Audited the onset-to-source-trial-reference proxy, then fitted predeclared 18-/12-month entry restrictions and conditional landmark analyses. At 18 months, entry restriction retains 424 patients; the dated complete-case landmark retains 161, with five test events. LightGBM ROC-AUC changes from reference 0.923 to 0.828 and 0.693, respectively. The 12-month landmark is infeasible for the planned validation. This tests temporal sensitivity, not an isolated causal bias effect or a delayed-entry survival model. |
| 2. Trial clustering | Both methods/results and the conclusion now distinguish patient separation from source-trial separation. No reliable trial-membership variable was established, and missingness patterns were not asserted to be trial identifiers. No leave-trial-out result is claimed. |
| 3. Bagging interpretation | Repeated the original 15 development folds with saved baseline parameters, comparing single models with ten-estimator BalancedBagging. All 14 original balanced-accuracy means reproduce to numerical precision. ROC-AUC improves for KNN/DT, is almost unchanged for NB, and falls for RF/SVM/MLP/LightGBM. The ten-configurations versus ten-bags protocol difference is explicit. |
| 4. Calibration | Reused validated, locally saved test probabilities for all seven selected Study II configurations. Added average precision, Brier scores, means and five-quantile-bin reliability plots. LightGBM+ROS: AP 0.704, Brier 0.1034, mean probability 0.260 versus prevalence 0.116. TreeSHAP's oversampled-model log-odds scale is explicit. No recalibration was used. |
| 5. Study I CV | Added three repetitions of stratified five-fold CV with saved parameters: 195 six-month and 120 three-month fits. Fold-specific targets/preprocessing remain training-only; TEST was not used. Balanced RF leads at six months (0.422), versus XGBoost 0.414. This is partition stability of selected configurations, not nested validation or a reason to treat 15 overlapping folds as independent tests. |
| 6. Selection/baseline tables | Added Study I three-month, six-month and six-month-excluded summaries, and Study II complete-case versus incomplete-case summaries. Removed the unsupported inference that attrition alone demonstrates selection for slower progressors. |
| 7. Encoding | Primary Study I race imputation remains disclosed. A DEV-only fixed-parameter sensitivity compares numeric median, explicit recorded states and omission for XGBoost/LR, using 90 fits. Mean AP changes by at most 0.0027 without replacing any primary result or LIME figure. Study II unknown onset text now remains missing; no current record used the old default. Corrected BMI boundaries change two original codes and are evaluated separately. Landmark features use only dated past records, including height and riluzole, and corrected BMI bins; missing dated treatment is no recorded use, not verified absence. The primary replication representation remains explicit. |
| 8. Search audit | Original Study I trial checks remain unchanged. Historical Study II fold failures cannot be excluded from aggregates. The revised search now logs failures and successful fold metrics, exports trial histories, and stops on errors; the inactive pruner was removed. Supplementary KNN+RUS failure is explicit rather than assigned 0.5. No historical failure-free claim or new search is made. |
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
- LIME now samples valid categorical groups and compares original-variable units
  with grouped SHAP. The original probabilities/cases reproduce. Overlap is
  5/7/6/7 variables; weighted local R2 ranges from 0.198 to 0.456. The FN surrogate
  crosses the research threshold, so categorical validity is not treated as fidelity.
- Added glossary entries for IQR/TRIPOD+AI and a latexmk glossary build rule.
- The saved SHAP-ranking comparison gives rho=0.389328 and p=0.066321. The thesis's
  p=0.066 was retained; the report's p=0.067 was not copied over it.

## Verification and Limits

`review_validation.json` records active-source checks: no undefined/duplicate
labels, undefined citations or glossary uses, missing active graphics, or unbalanced
braces. Existing explicit page-break controls were preserved. Abstract counts,
including titles and excluding keywords, are 138 EN and 147 PT after the temporal
follow-up was integrated.

The new graph exports were visually inspected. No LaTeX compiler was available in
the local toolchain; a full Overleaf compilation and pagination check have not been run.
The standalone Study I and Study II audit exports were executed against the
authorised local study directories. Their numeric summaries reproduce this
revision's saved checks (relative tolerance 1e-6, absolute tolerance 1e-7).
Exported analysis functions were compared structurally with the canonical source;
the bagging script differs only in standalone paths. This packaging check does not
constitute an independent reproduction of the primary model development.
Unused duplicate-label table files remain outside the active input graph; they were
not deleted. Patient-level probability arrays and source data are not part of the
published outputs. Case-specific fidelity CSVs remain local; the dissertation
retains its anonymised case-explanation tables and figures. Published numerical
exports contain aggregate results, input-file hashes and version metadata.

The latest supplementary code, selected parameters, synthetic tests and aggregate
results were published to the Study I and Study II repositories in commits
`d66974f` and `d89bc29`. Their standalone paths and test fixtures were checked;
the aggregate exports match the canonical results. Publication details are in
`REPOSITORY_SYNC_20261024.md`. This is not a clean-environment study reproduction.

## Deliberately Not Re-Run

Delayed-entry survival models retaining censored cases, trial-level validation,
wider/new Optuna searches, alternative progression target percentiles, replacement
of the primary Study I race encoding and clean-environment reproduction were not performed. Entry-only
restrictions and conditional landmarks do not remove all recruitment/missingness
bias or identify an isolated amount of discrimination attributable to left
truncation. Detailed results and checks are in `REVISED_ANALYSES_RESULTS.md`.

## Historical Log Search Follow-Up

On 24 October 2026, a read-only search covered the thesis working directories,
the original Study II directory, seven relevant ZIP archives and one RAR archive.
No historical Study II fold-failure ledger or persistent Optuna study was found.
The two local Step 5 result files and the backup in `CodigoMain (4).zip` have the
same SHA-256 digest. Each contains 70 unique configuration summaries reporting
50 trials, with no whole-configuration failure row. These checks do not establish
whether any individual historical fold received the silent AUC-0.5 fallback.
The existing thesis limitation therefore remains. No model was fitted and no
primary output was replaced during this search. Search scope and aggregate
evidence are recorded in `review_s2_historical_log_audit.json` and
`STUDY2_HISTORICAL_LOG_AUDIT.md`.

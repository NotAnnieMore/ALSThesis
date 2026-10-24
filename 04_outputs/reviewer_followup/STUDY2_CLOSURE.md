# Study II: Reviewer-Report Closure

Checked on 24 October 2026 against
`Thesis_Ivo_review_annotated_source/review/review_report.tex` and the active
dissertation source. The author chose to retain the original replication as a
reference and add temporal/encoding sensitivities rather than replace it.

## Scope of Closure

No additional model family or primary re-optimisation is required to implement
the selected response to the report. Its first suggestion offers early-entry
restriction OR delayed-entry survival modelling. The entry restrictions have
been fitted, with conditional landmark analyses added to address predictor
availability. This closes the agreed revision, not every source of bias.

| Report point applicable to Study II | Evidence and status |
| --- | --- |
| 1. Left truncation | Onset-to-trial-reference distribution is reported by class; exact enrolment dates were not established. Original results remain the reference. Entry limits at 18/12 months and dated landmark tasks are separately reported in Methods, Results and the supplementary appendix. No causal decomposition of the AUC change is claimed. |
| 2. Trial clustering | Patient-level separation and shared source trials are explicit. Measurement `Test_trial` and treatment `Study_Arm` are not clinical-trial IDs. Reliable membership was not reconstructed, so leave-trial-out validation remains unavailable rather than fabricated. |
| 3. BalancedBagging | Added ROC-AUC on the same 15 development folds. All original balanced-accuracy means reproduce; ranking gains are not assumed from operating-point gains. Ten bags of one configuration are distinguished from the reference's ten selected configurations, and RF/LightGBM are extensions. |
| 4. Calibration | All seven original selected models have AP, Brier score and reliability diagrams from validated saved probabilities. Reference prevalence and oversampled-model TreeSHAP log-odds are explicit. No retrospective recalibration is presented as held-out validation. |
| 6. Selection | Complete-case and excluded-participant characteristics, observation counts and class proportions are reported. Expanded-cohort imputation sensitivity is retained as descriptive, on its different split. |
| 7. Encoding | Unknown onset text now remains missing; no current source record used the former fallback. BMI-boundary correction changes two codes and is evaluated separately. Landmark BMI and dated riluzole use respect prediction time; no dated record means no recorded use, not known non-treatment. Original definitions remain explicit in the reference analysis. |
| 8. Search limits/failures | Original searches were not repeated. Historical fold failures cannot be excluded from aggregate exports, even after the archive search. Revised code logs errors and stops; new fold/trial exports improve future auditing. This uncertainty is explicit in Methods and Results limitations. |
| 9. Release reproducibility | Pinned current runtime inventories and historical-freeze limits are documented. Supplementary scripts, synthetic tests, selected parameters and aggregate results were synchronised and pushed to the dedicated public study repository in commit `d89bc29`. No clean-environment reproduction is claimed. |
| 10. Reporting/literature | Primary method/staging references and retrospective TRIPOD+AI coverage are included. Author-confirmed funding, conflicts and lack of public preregistration are distinguished from the source trials' ethics/funding. |

Report point 5 concerns Study I and is handled by its complementary repeated
stratified validation, without replacing the original selected model.

Minor Study II checks also remain covered: the 1,967 unresolved outcomes were
reconstructed independently; the LightGBM/SVM sensitivity comparison is paired;
the abstract includes the comparable BalancedBagging pattern; ROC-AUC optimisation
is described as a deliberate departure from balanced-accuracy model selection.

## Limits That Must Remain in the Text

- The primary predictors are predominantly post-diagnosis and often untimed as a
  complete vector. The reference is retrospective classification, not a model
  available at diagnosis.
- Entry-only restrictions do not impose predictor dates. Landmarking does, but
  changes the clinical prediction question and retains complete-case selection.
- The 18-month landmark has only 33 test participants and five deaths. Its ranking
  and sensitivity estimates are too imprecise to justify a preferred model.
- KNN+RUS fails for the 12-month entry cohort because k=31 exceeds 22 resampled
  training cases. The 12-month landmark has insufficient negative cases for the
  predeclared five-fold design. Neither receives a fallback performance score.
- Trial recruitment, missing outcomes, absent reliable trial IDs and unrecoverable
  historical fold logs remain limitations. None is declared resolved by rewriting.

## Final Checks

The expanded tests recalculate all 27 completed temporal result rows and 135
threshold rows from the private saved predictions. They verify test membership,
ordering, labels, confusion matrices, AP, ROC-AUC, Brier scores and threshold
classifications. Input-file hashes remain unchanged; the 432 completed fits and
infeasibility/failure records agree with the cohort flows.

The Study I runtime passes 16 focused tests; the Study II runtime passes the 15
applicable tests and skips the LIME fixture, which is not installed there. Static
LaTeX checks resolve all active labels, citations, glossary uses and graphic
paths. The author's current page-break controls and all pre-existing external
table/figure files are preserved. Compilation and visual pagination remain the
author's Overleaf check; no local compilation is claimed.

No new Study II training was performed during the closure or repository sync.
Subsequent publication is recorded in `REPOSITORY_SYNC_20261024.md`.

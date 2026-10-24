# Supplementary Repository Synchronisation

Publication on 24 October 2026 followed the author's approval to synchronise both
study repositories and commit/push the dissertation changes. The original models,
test results and thresholds were retained; no study model was retrained during
publication.

## Published Study Commits

- Study I: [`d66974f49eb8cee87710924b28ccc8232f771abb`](https://github.com/Applied-Intelligence-Hub/explainable-als-progression/commit/d66974f49eb8cee87710924b28ccc8232f771abb).
- Study II: [`d89bc2998b017f8ee97c9739a2a3025c807c3ec6`](https://github.com/Applied-Intelligence-Hub/als-survival-imbalance-ml/commit/d89bc2998b017f8ee97c9739a2a3025c807c3ec6).

The main dissertation revision and this record are included in the accompanying
main-repository commit; its identifier is available in Git history.

## Scope

Study I includes the categorical-aware LIME view and focused explanation script,
updated Step 8, repeated stratified validation, race-encoding sensitivity, four
synthetic regression tests, 21 selected parameter JSONs, updated README and
pipeline figure, and aggregate result summaries. Original local split manifests
and prediction/profile artefacts remain required for exact historical checks.

Study II includes temporal cohort construction/evaluation, corrected unknown-site
handling, fail-fast search logging, eight synthetic regression tests, selected
configurations, the aggregate 70-combination search summary, a standalone temporal
plotter, and aggregate temporal metrics/thresholds. The plotter can read the
published summaries without access to patient data or model fitting.

Only the two intended standalone path expressions differ in the Study II temporal
evaluator. Canonical Study I scripts and the other three Study II scripts match
their dissertation sources. Existing repository licence and README content outside
the update were retained; author/provenance blocks previously removed upstream
were not restored.

## Verification

- The freshly fetched remote branches matched local HEAD before editing.
- Four Study I and eight Study II standalone synthetic tests passed.
- Supplementary CLIs passed their help/import checks in their respective runtimes.
- Published JSON summaries match the canonical CSV results numerically.
- The standalone temporal plot reproduces the thesis PNG pixel for pixel.
- Selected configuration files contain hyperparameters and aggregate scores, not
  patient records. Only those files and the aggregate search CSV were explicitly
  allowed through the study repositories' output ignore rules.
- Staged-file schema checks excluded patient identifiers, outcome/probability
  arrays, source/derived patient tables and fitted-model binaries.
- Raw data, split manifests, reconstructed cohorts, individual predictions, local
  LIME weights and case-specific fidelity CSVs remain Git-ignored.

The repository preparation helper exports scoped code, tests, parameters and
aggregate results to the existing local study checkouts. It does not edit their
READMEs, rebuild the primary studies or publish data. No clean-environment
reproduction, external validation or local LaTeX compilation is claimed.

The user's loose `23_10_2026_Thesis_MSc_Ivo.pdf` is not part of the code/thesis-source
commit and remains local. Presentation and journal-paper workspaces, internal
review documents and private data remain excluded under the existing rules.

# Study I Figure Text Corrections

Updated on 24 October 2026. Only the two active PNG assets were replaced; their
filenames, pixel dimensions, cohort counts and train/test counts are unchanged.
No model training, scientific-result changes or LaTeX source edits were needed.

## Active Files

- `overleaf/images/figures/study1_pipeline.png`: 1024 x 1536 pixels.
- `overleaf/images/figures/study1_cohort_derivation.png`: 1448 x 1086 pixels.

The Methods chapter already references these exact paths. Its captions already
describe the corrected interpretation, so they did not need another edit.

## Changes

- Pipeline stage 6 says cross-horizon comparison, not validation.
- Pipeline stage 8 includes Logistic Regression coefficients.
- Pipeline stage 9 describes research reporting and non-clinical review flags.
- Cohort exclusions are shown for both horizons: 3,029 and 4,044.
- The 720-person overlap is a separate note outside the selection path.
- The shared exclusion note is not connected as an extra six-month branch.

Both files were visually inspected. PNG integrity, unchanged dimensions and the
cohort/split arithmetic were checked. No whole-thesis pagination review was
performed in this follow-up; the author confirmed that check separately.

## Editing Method and Preservation

Targeted image edits used the built-in `image_gen` tool, retaining the existing
design, icons and colours. The original PNG files were backed up outside the Git
worktree in the session's local visualisation directory. No alternate PNG/PDF
version was introduced into the LaTeX input graph.

## Final Pipeline Edit Prompt

```text
Use case: text-localization.
Asset type: existing scientific thesis pipeline diagram, portrait 2:3.
Input image 1 is the EDIT TARGET. Edit this exact diagram, not a new design.
Preserve the original layout, white background, icons, all nine numbered stages (including 3a and 3b), arrows, dashed DEV boundary, all existing colours and all numerical values. Preserve sharp publication-quality text and high resolution, ideally 2048 x 3072 or the input resolution.
Change ONLY these textual elements:
1. In stage 6, replace "Cross-horizon validation (3m vs 6m)" with "Cross-horizon comparison (3m vs 6m)".
2. In stage 8 (Explainability), replace its explanatory line with two short lines:
"SHAP global/local analysis  •  LIME local explanations"
"Logistic Regression coefficients"
Keep the two lines neatly inside the same box; do not change its title or icon.
3. In stage 9, replace its two explanatory lines with:
"Per-patient research report  •  SHAP-based feature"
"contributions  •  Descriptive review flags (non-clinical)"
Keep the stage title "Prototype Risk Report".
All other words and numbers must remain exactly as in the target. In particular stage 5 must still say "7 unweighted + 6 class-weighted configurations = 13" and "60 TPE trials" and "PR-AUC objective". No new content, no cropping, no extra border or watermark.
```

## Final Cohort Edit Prompt

```text
Use case: text-localization.
Asset type: existing scientific thesis cohort flowchart, landscape 4:3.
Input image 1 is the EDIT TARGET. Preserve its layout, existing rounded boxes, icons, serif typography, colours (purple top, teal 3-month branch, orange 6-month branch, red exclusion note), white background and all original cohort/split numbers. Produce a sharp, high-resolution corrected version of this same figure.
Make only these changes:
1. Remove the italic sentence "720 subjects eligible at both horizons" from the middle of the connecting arrows. The top N=5,436 box must branch directly into the 3-month and 6-month cohorts without an intermediate selection.
2. Add an independent italic note BELOW all four DEV/TEST boxes, centred in the bottom white margin, with NO arrow or connecting line:
"Cohort overlap (not a filter): 720 subjects eligible at both horizons"
Use the same serif typography, readable and not touching the boxes.
3. Remove the short horizontal arrow from the 6-month cohort box into the red exclusion note. The red box is a separate shared explanatory note for BOTH horizons, not a third cohort.
4. Keep that red note near its original right-side position and preserve its red warning icon. Replace its text with exactly these five lines, increasing the box height slightly if necessary and keeping all text inside it:
"Excluded per horizon:"
"3 months: N = 3,029"
"6 months: N = 4,044"
"No endpoint assessment or"
"fewer than two distinct timepoints"
The last two lines can wrap into three lines if required; all text must be comfortably readable.
Keep unchanged: top label "Subjects with valid ALSFRS-R assessment(s)" and "N = 5,436"; 3-month cohort "N = 2,407 (44.3%)"; 6-month cohort "N = 1,392 (25.6%)"; 3-month DEV "n = 1,925 (80%)" and TEST "n = 482 (20%)"; 6-month DEV "n = 1,113 (80%)" and TEST "n = 279 (20%)".
Do not add new selection steps, new numerical values, extra connecting arrows, clinical content, borders or watermarks. Preserve the rest of the original diagram.
```

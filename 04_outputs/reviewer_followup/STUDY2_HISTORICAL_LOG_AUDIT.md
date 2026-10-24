# Study II Historical Log Search

Checked on 24 October 2026. This was a read-only audit of existing files, not a
model rerun. No Optuna search, model fit or replacement of a primary result was
performed. Machine-readable evidence is in `review_s2_historical_log_audit.json`.

## Search Scope

- Searched the thesis directory tree under `G:/TESE`, including hidden and
  Git-ignored files, for log/stdout/stderr files, databases, journals and filenames
  associated with training, trials, folds or Optuna.
- Searched text, JSON and notebook files for training banners, trial/fold failure
  markers, tracebacks and persistent-storage identifiers. Git internals, virtual
  environments, dependency caches and source/patient-data directories were excluded.
- Inspected the current Study II outputs under `CodigoMain/replication/` and the
  earlier `V2Codigo/` working directory.
- Listed seven relevant ZIP archives without extracting them: five `CodigoMain`
  project backups and the two publication/Overleaf archives under `replication/`.
  Also listed `CodigoMain.rar`. Only the fourth ZIP backup contained Study II
  Step 5 result files; these were aggregate summaries, not trial/fold histories.
- Checked the available Step 5 scripts. They create in-memory Optuna studies,
  without configured persistent storage. Resampling and fit exceptions are caught
  without recording their details before the AUC-0.5 fallback is appended.

## What Was Found

The current and earlier Step 5 CSV files, and the copy in `CodigoMain (4).zip`,
have the same SHA-256 digest:

```text
14a889baef1bfb5c6d9d1bca64cd8ab8b20ab92be689050205c132630ba439ce
```

Each CSV contains:

- 70 rows and 70 unique classifier/resampling pairs;
- 50 reported trials for every configuration;
- no row with `n_trials = 0` or `best_trial < 0`, the script's whole-configuration
  failure markers;
- no mean ROC-AUC equal to zero or 0.5;
- no saved individual fold-score columns or per-fold failure flags.

The loose tuning logs found belong to Study I. Notebook content matches belong
to Study I or other data-preparation/reference projects. The Study II advisor
summary is prose, not an execution log. None supplies the missing historical
fold-level record.

## Conclusion

No historical Study II fold-failure ledger was found within this search scope.
There is no saved whole-configuration failure marker, but this does not exclude
an exception inside a fold that was silently converted to AUC 0.5. The same
aggregate output in multiple locations is a backup, not independent confirmation
that every historical fit succeeded.

The thesis already states that this absence cannot be established from the
stored means. That limitation remains accurate. This audit is not evidence that
failures occurred, nor proof that none occurred. No change to the reported metrics
or fitted models follows from this search.

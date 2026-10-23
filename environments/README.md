# Verified Runtime Snapshots

These files pin every installed distribution in the two local Python 3.12.10
environments inspected on 23 October 2026:

- `study1-verified.txt`: the Study I runtime, also used for the saved-prediction
  reviewer audits.
- `study2-verified.txt`: the Study II runtime used for the additional fixed-parameter
  BalancedBagging comparison.

The JSON companions record the Python version and package inventory. These are
current verification snapshots, not historical locks captured when every original
experiment ran. They do not establish bitwise reproducibility across operating
systems, hardware, or numerical libraries. The published study-specific repositories
must carry these snapshots alongside their installation instructions when the
revision is released; no publication or push is performed by the audit scripts.

Use a separate environment for each study:

```powershell
py -3.12 -m venv .venv-study1
.venv-study1/Scripts/python.exe -m pip install -r environments/study1-verified.txt
```

For Study II, use `study2-verified.txt` in a second environment. PRO-ACT access,
unchanged input row order, and verification against the saved split manifests remain
necessary. Installing the package inventory alone does not reconstruct those inputs.

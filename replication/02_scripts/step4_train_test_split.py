"""
Step 4 — Stratified Train/Test Split (80/20)
=============================================
Splits the processed dataset into stratified train (80%) and test (20%)
sets, preserving the Short/Non-Short class proportions.

Following Papaiz et al. (2024): 80/20 stratified split.
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '01_data', 'processed')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '03_outputs', 'step4')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

start_time = time.time()

print("=" * 70)
print("STEP 4 — STRATIFIED TRAIN/TEST SPLIT (80/20)")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD Step 3 output
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(PROCESSED_DIR, 'step3_final_dataset.csv'))
print(f"\nLoaded dataset: {len(df):,} patients × {len(df.columns)} columns")

feature_cols = [c for c in df.columns if c not in ['subject_id', 'Survival_Group']]
X = df[feature_cols]
y = df['Survival_Group']

print(f"Features: {len(feature_cols)}")
print(f"Target distribution: Short(1)={y.sum():,}, Non-Short(0)={(y == 0).sum():,}")


# ══════════════════════════════════════════════════════════════════════════
# 2. STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print(f"Splitting: {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test "
      f"(random_state={RANDOM_STATE})...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Also keep subject_ids for traceability
train_ids = df.loc[X_train.index, 'subject_id'].values
test_ids = df.loc[X_test.index, 'subject_id'].values


# ══════════════════════════════════════════════════════════════════════════
# 3. VERIFY STRATIFICATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STRATIFICATION VERIFICATION")

for name, ys in [('Full', y), ('Train', y_train), ('Test', y_test)]:
    n = len(ys)
    n_short = ys.sum()
    n_nonshort = n - n_short
    ir = n_nonshort / n_short if n_short > 0 else float('inf')
    pct_short = n_short / n * 100
    print(f"  {name:6s}: {n:>5,} patients | "
          f"Short={n_short:>4,} ({pct_short:>5.1f}%) | "
          f"Non-Short={n_nonshort:>5,} ({100-pct_short:>5.1f}%) | "
          f"IR={ir:.1f}")


# ══════════════════════════════════════════════════════════════════════════
# 4. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING OUTPUTS...")

# Save train set (features + target)
train_df = X_train.copy()
train_df['Survival_Group'] = y_train
train_df.insert(0, 'subject_id', train_ids)
train_path = os.path.join(PROCESSED_DIR, 'step4_train.csv')
train_df.to_csv(train_path, index=False)
print(f"  Train set → {train_path} ({len(train_df):,} rows)")

# Save test set (features + target)
test_df = X_test.copy()
test_df['Survival_Group'] = y_test
test_df.insert(0, 'subject_id', test_ids)
test_path = os.path.join(PROCESSED_DIR, 'step4_test.csv')
test_df.to_csv(test_path, index=False)
print(f"  Test set  → {test_path} ({len(test_df):,} rows)")

# Save numpy arrays for direct use in modeling (no subject_id)
np.savez(
    os.path.join(PROCESSED_DIR, 'step4_arrays.npz'),
    X_train=X_train.values,
    X_test=X_test.values,
    y_train=y_train.values,
    y_test=y_test.values,
    feature_names=np.array(feature_cols),
)
print(f"  Arrays    → {os.path.join(PROCESSED_DIR, 'step4_arrays.npz')}")

# Save summary
summary = {
    'total_patients': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_short': int(y_train.sum()),
    'train_nonshort': int((y_train == 0).sum()),
    'test_short': int(y_test.sum()),
    'test_nonshort': int((y_test == 0).sum()),
    'n_features': len(feature_cols),
    'random_state': RANDOM_STATE,
}
pd.Series(summary).to_csv(os.path.join(OUT_DIR, 'step4_summary.csv'))

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"Step 4 completed in {elapsed:.1f}s")
print(f"{'=' * 70}")

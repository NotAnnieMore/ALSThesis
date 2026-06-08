"""
Check best configs structure and get baseline (None) params per classifier.
"""
import json
import pandas as pd

# Best overall configs
with open('03_outputs/step5/step5_best_configs.json') as f:
    configs = json.load(f)

print("=== Best configs (overall, by AUC) ===")
for clf, c in configs.items():
    print(f"  {clf}: tech={c['imbalance_technique']}, params={c['best_params']}")

# Get baseline (None) best params from CV results
print("\n=== Baseline (None) configs from CV results ===")
df = pd.read_csv('03_outputs/step5/step5_cv_results.csv')
df['imbalance_technique'] = df['imbalance_technique'].fillna('None')
baseline = df[df['imbalance_technique'] == 'None']
for _, row in baseline.iterrows():
    print(f"  {row['classifier']}: AUC={row['mean_auc']:.4f}, params={row['best_params']}")

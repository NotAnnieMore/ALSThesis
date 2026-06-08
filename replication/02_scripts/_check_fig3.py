import pandas as pd
df = pd.read_csv('03_outputs/step5/step5_cv_results.csv')
df['imbalance_technique'] = df['imbalance_technique'].fillna('None')

for clf in ['NaiveBayes', 'RandomForest', 'MLP', 'SVM', 'KNN', 'DecisionTree', 'LightGBM']:
    sub = df[df['classifier'] == clf]
    baseline = sub[sub['imbalance_technique'] == 'None'].iloc[0]
    best = sub.loc[sub['mean_auc'].idxmax()]
    
    print(f"=== {clf} ===")
    print(f"  Baseline (None):  AUC={baseline['mean_auc']:.4f}  Recall={baseline['mean_recall']:.4f}  Spec={baseline['mean_specificity']:.4f}")
    print(f"  Best (by AUC):    AUC={best['mean_auc']:.4f}  Recall={best['mean_recall']:.4f}  Spec={best['mean_specificity']:.4f}  tech={best['imbalance_technique']}")
    same = best['imbalance_technique'] == 'None'
    print(f"  Same as baseline? {same}")
    print()

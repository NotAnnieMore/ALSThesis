"""Check SVC data and alternative FVC approaches to increase coverage"""
import pandas as pd
import numpy as np

# Load SVC
svc = pd.read_csv('01_data/raw/PROACT_SVC.csv')
print("SVC columns:", svc.columns.tolist())
print(f"SVC rows: {len(svc)}, subjects: {svc['subject_id'].nunique()}")

# Check SVC pct availability
print(f"\nSVC pct_of_Normal_Trial_1: {svc['pct_of_Normal_Trial_1'].notna().sum()} non-null")
svc_normal_col = 'Subject_Normal' if 'Subject_Normal' in svc.columns else 'subject_normal'
print(f"\nSVC {svc_normal_col}: {svc[svc_normal_col].notna().sum()} non-null")
print(f"SVC Subject_Liters_Trial_1: {svc['Subject_Liters_Trial_1'].notna().sum()} non-null")

# Compute SVC%
has_l_n = svc['Subject_Liters_Trial_1'].notna() & svc[svc_normal_col].notna()
svc['svc_pct'] = np.where(svc['pct_of_Normal_Trial_1'].notna(), svc['pct_of_Normal_Trial_1'],
    np.where(has_l_n, svc['Subject_Liters_Trial_1'] / svc[svc_normal_col] * 100, np.nan))

svc_subj = set(svc[svc['svc_pct'].notna()]['subject_id'].unique())
print(f"Subjects with SVC%: {len(svc_subj)}")

# FVC
fvc = pd.read_csv('01_data/raw/PROACT_FVC.csv')
has_l_n_f = fvc['Subject_Liters_Trial_1'].notna() & fvc['subject_normal'].notna()
fvc['fvc_pct'] = np.where(fvc['pct_of_Normal_Trial_1'].notna(), fvc['pct_of_Normal_Trial_1'],
    np.where(has_l_n_f, fvc['Subject_Liters_Trial_1'] / fvc['subject_normal'] * 100, np.nan))
fvc_subj = set(fvc[fvc['fvc_pct'].notna()]['subject_id'].unique())
print(f"Subjects with FVC%: {len(fvc_subj)}")

# Union FVC+SVC
combined_vc = fvc_subj | svc_subj
print(f"FVC or SVC: {len(combined_vc)}")
print(f"Only SVC (not FVC): {len(svc_subj - fvc_subj)}")
print(f"Only FVC (not SVC): {len(fvc_subj - svc_subj)}")

# Now check intersection with the full pipeline using FVC OR SVC
hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
vs = pd.read_csv('01_data/raw/PROACT_VITALSIGNS.csv')
alsfrs = pd.read_csv('01_data/raw/PROACT_ALSFRS.csv')

onset_subj = set(hist[hist['Onset_Delta'].notna()]['subject_id'].unique())
site_text = set(hist.dropna(subset=['Site_of_Onset'])['subject_id'].unique())
bin_cols = ['Site_of_Onset___Bulbar', 'Site_of_Onset___Limb', 'Site_of_Onset___Limb_and_Bulbar',
            'Site_of_Onset___Other', 'Site_of_Onset___Spine']
site_bin = set(hist[hist[bin_cols].notna().any(axis=1)]['subject_id'].unique())
combined_site = site_text | site_bin
age_subj = set(demo[demo['Age'].notna()]['subject_id'].unique())
sex_subj = set(demo[demo['Sex'].notna()]['subject_id'].unique())
height_subj = set(vs[vs['Height'].notna()]['subject_id'].unique())
weight_subj = set(vs[vs['Weight'].notna()]['subject_id'].unique())
alsfrs_subj = set(alsfrs['subject_id'].unique())

# With FVC only
all_fvc = fvc_subj & onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"\nAll features (FVC only): {len(all_fvc)}")

# With FVC or SVC
all_combined = combined_vc & onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"All features (FVC or SVC): {len(all_combined)}")

# Additional patients from SVC
print(f"Additional from SVC: {len(all_combined) - len(all_fvc)}")

# Let's also check: what if we DON'T require pct and just use raw liters?
fvc_raw_subj = set(fvc[fvc['Subject_Liters_Trial_1'].notna()]['subject_id'].unique())
svc_raw_subj = set(svc[svc['Subject_Liters_Trial_1'].notna()]['subject_id'].unique())
raw_vc = fvc_raw_subj | svc_raw_subj
print(f"\nRaw liters FVC: {len(fvc_raw_subj)}, SVC: {len(svc_raw_subj)}, Union: {len(raw_vc)}")

all_raw = raw_vc & onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"All features (raw liters): {len(all_raw)}")

# What about using pct_of_Normal from FVC trial 2, 3 etc?
for trial in ['pct_of_Normal_Trial_1', 'pct_of_Normal_Trial_2', 'pct_of_Normal_Trial_3']:
    if trial in fvc.columns:
        cnt = fvc[trial].notna().sum()
        subj = fvc[fvc[trial].notna()]['subject_id'].nunique()
        print(f"\nFVC {trial}: {cnt} rows, {subj} subjects")

# Let me also check what happens with Diagnosis_Delta as an alternative to Onset_Delta
diag_subj = set(hist[hist['Diagnosis_Delta'].notna()]['subject_id'].unique())
print(f"\nDiagnosis_Delta subjects: {len(diag_subj)}")
print(f"Onset OR Diagnosis: {len(onset_subj | diag_subj)}")

# Full intersection with Onset OR Diagnosis  
all_onset_or_diag = fvc_subj & (onset_subj | diag_subj) & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"All features (onset OR diagnosis delta): {len(all_onset_or_diag)}")

"""Test: using SVC as fallback for FVC, check target distribution"""
import pandas as pd
import numpy as np

fvc = pd.read_csv('01_data/raw/PROACT_FVC.csv')
svc = pd.read_csv('01_data/raw/PROACT_SVC.csv')
hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
vs = pd.read_csv('01_data/raw/PROACT_VITALSIGNS.csv')
alsfrs = pd.read_csv('01_data/raw/PROACT_ALSFRS.csv')
death = pd.read_csv('01_data/raw/PROACT_DEATHDATA.csv')

def get_at_diagnosis(df, delta_col):
    df = df.dropna(subset=[delta_col]).copy()
    df['_abs'] = df[delta_col].abs()
    idx = df.groupby('subject_id')['_abs'].idxmin()
    return df.loc[idx].drop(columns='_abs').reset_index(drop=True)

# 1. Compute FVC% (prefer pct_of_Normal, else liters/normal)
has_l_n = fvc['Subject_Liters_Trial_1'].notna() & fvc['subject_normal'].notna()
fvc['vc_pct'] = np.where(fvc['pct_of_Normal_Trial_1'].notna(), fvc['pct_of_Normal_Trial_1'],
    np.where(has_l_n, fvc['Subject_Liters_Trial_1'] / fvc['subject_normal'] * 100, np.nan))
fvc_sub = fvc[fvc['vc_pct'].notna()][['subject_id', 'Forced_Vital_Capacity_Delta', 'vc_pct']].copy()
fvc_diag = get_at_diagnosis(fvc_sub, 'Forced_Vital_Capacity_Delta')
fvc_diag = fvc_diag[['subject_id', 'vc_pct']].copy()

# 2. SVC as fallback
svc_norm = 'Subject_Normal'
has_l_n_s = svc['Subject_Liters_Trial_1'].notna() & svc[svc_norm].notna()
svc['vc_pct'] = np.where(svc['pct_of_Normal_Trial_1'].notna(), svc['pct_of_Normal_Trial_1'],
    np.where(has_l_n_s, svc['Subject_Liters_Trial_1'] / svc[svc_norm] * 100, np.nan))
svc_sub = svc[svc['vc_pct'].notna()][['subject_id', 'Slow_vital_Capacity_Delta', 'vc_pct']].copy()
svc_diag = get_at_diagnosis(svc_sub, 'Slow_vital_Capacity_Delta')
svc_diag = svc_diag[['subject_id', 'vc_pct']].copy()

# Merge: prefer FVC, fallback to SVC
vc_diag = fvc_diag.copy()
svc_only = svc_diag[~svc_diag['subject_id'].isin(fvc_diag['subject_id'])]
vc_diag = pd.concat([vc_diag, svc_only], ignore_index=True)
print(f"VC subjects (FVC+SVC fallback): {len(vc_diag)}")
vc_subj = set(vc_diag['subject_id'])

# 3. Other features
onset = hist.dropna(subset=['Onset_Delta']).groupby('subject_id')['Onset_Delta'].min().reset_index()
onset_dict = dict(zip(onset['subject_id'], onset['Onset_Delta']))
onset_subj = set(onset['subject_id'])

site_text = hist.dropna(subset=['Site_of_Onset'])[['subject_id', 'Site_of_Onset']].drop_duplicates('subject_id')
bin_cols = ['Site_of_Onset___Bulbar', 'Site_of_Onset___Limb', 'Site_of_Onset___Limb_and_Bulbar',
            'Site_of_Onset___Other', 'Site_of_Onset___Spine']
site_bin = hist[hist[bin_cols].notna().any(axis=1)][['subject_id'] + bin_cols].drop_duplicates('subject_id')
combined_site = set(site_text['subject_id']) | set(site_bin['subject_id'])

age_subj = set(demo[demo['Age'].notna()]['subject_id'])
sex_subj = set(demo[demo['Sex'].notna()]['subject_id'])
height_subj = set(vs[vs['Height'].notna()]['subject_id'])
weight_subj = set(vs[vs['Weight'].notna()]['subject_id'])
alsfrs_subj = set(alsfrs['subject_id'])

# ALL features
all_feat = vc_subj & onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"All features (with SVC fallback): {len(all_feat)}")

# Compute target
death_dedup = death.drop_duplicates('subject_id')
died_dict = dict(zip(death_dedup['subject_id'], death_dedup['Subject_Died']))
dd_dict = dict(zip(death_dedup['subject_id'], death_dedup['Death_Days']))

# Last visit
max_d = []
for df, col in [(alsfrs, 'ALSFRS_Delta'), (fvc, 'Forced_Vital_Capacity_Delta'), 
                (svc, 'Slow_vital_Capacity_Delta'), (vs, 'Vital_Signs_Delta')]:
    md = df.groupby('subject_id')[col].max().reset_index()
    md.columns = ['subject_id', 'max_d']
    max_d.append(md)
lv = pd.concat(max_d).groupby('subject_id')['max_d'].max()
lv_dict = lv.to_dict()

THRESHOLD = 24 * 30.44
short = nonshort = excluded = 0
for sid in all_feat:
    onset_d = onset_dict.get(sid, np.nan)
    died = died_dict.get(sid, None)
    dd = dd_dict.get(sid, np.nan)
    lvd = lv_dict.get(sid, np.nan)
    
    if pd.isna(onset_d):
        excluded += 1; continue
    
    if died == 'Yes' and pd.notna(dd):
        surv = dd - onset_d
        if surv <= THRESHOLD: short += 1
        else: nonshort += 1
    else:
        if pd.notna(lvd):
            fu = lvd - onset_d
            if fu > THRESHOLD: nonshort += 1
            else: excluded += 1
        else:
            excluded += 1

total = short + nonshort
print(f"\nTarget distribution (FVC+SVC fallback, {len(all_feat)} all-feature):")
print(f"  Short:     {short}")
print(f"  Non-Short: {nonshort}")
print(f"  Excluded:  {excluded}")
print(f"  Total valid: {total}")
if short > 0:
    print(f"  IR: {nonshort/short:.1f}")
    print(f"  Short%: {short/total*100:.1f}%")

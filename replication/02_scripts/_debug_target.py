"""Quick verification: how many of the 2,150 all-feature patients get a valid target?"""
import pandas as pd
import numpy as np

# Load tables
fvc = pd.read_csv('01_data/raw/PROACT_FVC.csv')
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')
vs = pd.read_csv('01_data/raw/PROACT_VITALSIGNS.csv')
alsfrs = pd.read_csv('01_data/raw/PROACT_ALSFRS.csv')
ril = pd.read_csv('01_data/raw/PROACT_RILUZOLE.csv')
death = pd.read_csv('01_data/raw/PROACT_DEATHDATA.csv')

# 1. FVC subjects (computed)
has_l_n = fvc['Subject_Liters_Trial_1'].notna() & fvc['subject_normal'].notna()
fvc['fvc_pct'] = np.where(fvc['pct_of_Normal_Trial_1'].notna(), fvc['pct_of_Normal_Trial_1'],
    np.where(has_l_n, fvc['Subject_Liters_Trial_1'] / fvc['subject_normal'] * 100, np.nan))
fvc_subj = set(fvc[fvc['fvc_pct'].notna()]['subject_id'].unique())

# 2. Onset
onset = hist.dropna(subset=['Onset_Delta']).groupby('subject_id')['Onset_Delta'].min().reset_index()
onset_subj = set(onset['subject_id'].unique())

# 3. Combined site
site_text = set(hist.dropna(subset=['Site_of_Onset'])['subject_id'].unique())
bin_cols = ['Site_of_Onset___Bulbar', 'Site_of_Onset___Limb', 'Site_of_Onset___Limb_and_Bulbar',
            'Site_of_Onset___Other', 'Site_of_Onset___Spine']
site_bin = set(hist[hist[bin_cols].notna().any(axis=1)]['subject_id'].unique())
combined_site = site_text | site_bin

# 4. Age, Sex
age_subj = set(demo[demo['Age'].notna()]['subject_id'].unique())
sex_subj = set(demo[demo['Sex'].notna()]['subject_id'].unique())

# 5. Height + Weight
height_subj = set(vs[vs['Height'].notna()]['subject_id'].unique())
weight_subj = set(vs[vs['Weight'].notna()]['subject_id'].unique())

# 6. ALSFRS
alsfrs_subj = set(alsfrs['subject_id'].unique())

# Intersection
all_feat = fvc_subj & onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"All-feature patients: {len(all_feat)}")

# Now compute target for these patients
onset_dict = dict(zip(onset['subject_id'], onset['Onset_Delta']))

# Death info
death_dedup = death.drop_duplicates('subject_id')
died_dict = dict(zip(death_dedup['subject_id'], death_dedup['Subject_Died']))
death_days_dict = dict(zip(death_dedup['subject_id'], death_dedup['Death_Days']))

# Last visit delta (max delta from ALSFRS, FVC, VS)
max_deltas = []
for df, col in [(alsfrs, 'ALSFRS_Delta'), (fvc, 'Forced_Vital_Capacity_Delta'), (vs, 'Vital_Signs_Delta')]:
    md = df.groupby('subject_id')[col].max().reset_index()
    md.columns = ['subject_id', 'max_delta']
    max_deltas.append(md)
all_max = pd.concat(max_deltas).groupby('subject_id')['max_delta'].max().reset_index()
last_visit_dict = dict(zip(all_max['subject_id'], all_max['max_delta']))

THRESHOLD = 24 * 30.44  # ~730 days

short = 0
nonshort = 0
excluded = 0
no_death = 0

for sid in all_feat:
    onset_d = onset_dict.get(sid, np.nan)
    died = died_dict.get(sid, None)
    dd = death_days_dict.get(sid, np.nan)
    lv = last_visit_dict.get(sid, np.nan)
    
    if pd.notna(onset_d):
        if died == 'Yes' and pd.notna(dd):
            survival = dd - onset_d  # onset_d is negative
            if survival <= THRESHOLD:
                short += 1
            else:
                nonshort += 1
        elif died == 'No' or died is None:
            # Use last visit
            if pd.notna(lv):
                follow_up = lv - onset_d
                if follow_up > THRESHOLD:
                    nonshort += 1
                else:
                    excluded += 1
            else:
                excluded += 1
        else:
            excluded += 1
    else:
        excluded += 1

print(f"\nTarget assignment for {len(all_feat)} all-feature patients:")
print(f"  Short:     {short}")
print(f"  Non-Short: {nonshort}")
print(f"  Excluded:  {excluded}")
total_valid = short + nonshort
print(f"  Total valid: {total_valid}")
if short > 0:
    ir = nonshort / short
    print(f"  IR: {ir:.1f}")
    print(f"  Short%: {short/total_valid*100:.1f}%")

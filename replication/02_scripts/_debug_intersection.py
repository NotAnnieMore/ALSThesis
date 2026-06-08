import pandas as pd
import numpy as np

fvc = pd.read_csv('01_data/raw/PROACT_FVC.csv')
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
death = pd.read_csv('01_data/raw/PROACT_DEATHDATA.csv')
hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')

# Check FVC pct availability more carefully
# For NaN-unit records, check if some of them DO have pct_of_Normal
nan_unit = fvc[fvc['Forced_Vital_Capacity_Units'].isna()]
print("NaN unit records with pct_of_Normal_Trial_1:")
print(f"  available: {nan_unit['pct_of_Normal_Trial_1'].notna().sum()}")
print(f"  missing: {nan_unit['pct_of_Normal_Trial_1'].isna().sum()}")

# Check pct availability across ALL records regardless of units
print(f"\nAll records with pct_of_Normal_Trial_1: {fvc['pct_of_Normal_Trial_1'].notna().sum()}")
print(f"Unique subjects with pct: {fvc[fvc['pct_of_Normal_Trial_1'].notna()]['subject_id'].nunique()}")

# Now check: maybe the NaN-unit records already have pct in subject_normal/pct columns
print(f"\nsubject_normal availability: {fvc['subject_normal'].notna().sum()}")
print(f"  unique subjects with subject_normal: {fvc[fvc['subject_normal'].notna()]['subject_id'].nunique()}")

# When pct_of_Normal_Trial_1 is available, what's the range?
pct_avail = fvc[fvc['pct_of_Normal_Trial_1'].notna()]
print(f"\npct_of_Normal_Trial_1 range: {pct_avail['pct_of_Normal_Trial_1'].min():.1f} - {pct_avail['pct_of_Normal_Trial_1'].max():.1f}")
print(f"  mean: {pct_avail['pct_of_Normal_Trial_1'].mean():.1f}")

# ==== Alternative approach: use Liters / subject_normal = pct
# How many records have Subject_Liters AND subject_normal?
has_liters_and_normal = fvc['Subject_Liters_Trial_1'].notna() & fvc['subject_normal'].notna()
print(f"\nRecords with Liters + Normal: {has_liters_and_normal.sum()}")
print(f"  subjects: {fvc[has_liters_and_normal]['subject_id'].nunique()}")

# Compute FVC% for ALL available records
fvc['fvc_pct_computed'] = np.where(
    fvc['pct_of_Normal_Trial_1'].notna(),
    fvc['pct_of_Normal_Trial_1'],
    np.where(
        has_liters_and_normal,
        fvc['Subject_Liters_Trial_1'] / fvc['subject_normal'] * 100,
        np.nan
    )
)
total_fvc = fvc['fvc_pct_computed'].notna().sum()
total_subj = fvc[fvc['fvc_pct_computed'].notna()]['subject_id'].nunique()
print(f"\nTotal records with FVC%: {total_fvc} ({total_subj} unique subjects)")

# Now let's see how this overlaps with target-eligible patients
# Need: death data (or long follow-up) + onset_delta
onset = hist.dropna(subset=['Onset_Delta']).groupby('subject_id')['Onset_Delta'].min().reset_index()
onset.rename(columns={'Onset_Delta': 'onset_delta'}, inplace=True)

print(f"\n=== Checking intersection with onset_delta ===")
fvc_subj = set(fvc[fvc['fvc_pct_computed'].notna()]['subject_id'].unique())
onset_subj = set(onset['subject_id'].unique())
age_subj = set(demo[demo['Age'].notna()]['subject_id'].unique())

print(f"FVC subjects: {len(fvc_subj)}")
print(f"Onset subjects: {len(onset_subj)}")
print(f"Age subjects: {len(age_subj)}")
print(f"FVC + Onset: {len(fvc_subj & onset_subj)}")
print(f"FVC + Onset + Age: {len(fvc_subj & onset_subj & age_subj)}")

# Now check with VS (height/weight/BMI)
vs = pd.read_csv('01_data/raw/PROACT_VITALSIGNS.csv')
height_subj = set(vs[vs['Height'].notna()]['subject_id'].unique())
weight_subj = set(vs[vs['Weight'].notna()]['subject_id'].unique())
print(f"Height subjects: {len(height_subj)}")
print(f"Weight subjects: {len(weight_subj)}")
print(f"Height + Weight: {len(height_subj & weight_subj)}")
print(f"FVC + Onset + Age + Height + Weight: {len(fvc_subj & onset_subj & age_subj & height_subj & weight_subj)}")

# Site of onset
site = hist.dropna(subset=['Site_of_Onset'])
site_subj = set(site['subject_id'].unique())
print(f"Site of Onset subjects: {len(site_subj)}")
print(f"All key features: {len(fvc_subj & onset_subj & age_subj & height_subj & weight_subj & site_subj)}")

# ALSFRS
alsfrs = pd.read_csv('01_data/raw/PROACT_ALSFRS.csv')
alsfrs_subj = set(alsfrs['subject_id'].unique())
print(f"ALSFRS subjects: {len(alsfrs_subj)}")

# Riluzole
ril = pd.read_csv('01_data/raw/PROACT_RILUZOLE.csv')
ril_subj = set(ril['subject_id'].unique())

all_intersection = fvc_subj & onset_subj & age_subj & height_subj & weight_subj & site_subj & alsfrs_subj & ril_subj
print(f"\nALL key tables intersection: {len(all_intersection)}")

# Of these, how many have death data?
death_subj = set(death[death['Subject_Died'] == 'Yes']['subject_id'].unique())
death_with_days = set(death[death['Death_Days'].notna()]['subject_id'].unique())
print(f"Deceased with Death_Days: {len(death_with_days)}")
print(f"All features + death_days: {len(all_intersection & death_with_days)}")

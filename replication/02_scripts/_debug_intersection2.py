import pandas as pd
import numpy as np

hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')

# The table has BOTH: Site_of_Onset (text column) AND Site_of_Onset___Bulbar/Limb/etc (binary columns)
# Let me extract site from EITHER source

# 1) From the text column
site_text = hist.dropna(subset=['Site_of_Onset'])[['subject_id', 'Site_of_Onset']].copy()
site_text_subj = set(site_text['subject_id'].unique())
print(f"Subjects with Site_of_Onset text: {len(site_text_subj)}")

# 2) From the binary columns
bin_cols = ['Site_of_Onset___Bulbar', 'Site_of_Onset___Limb', 
            'Site_of_Onset___Limb_and_Bulbar', 'Site_of_Onset___Other', 'Site_of_Onset___Spine']
has_any_binary = hist[bin_cols].notna().any(axis=1)
site_bin = hist[has_any_binary][['subject_id'] + bin_cols].copy()
site_bin_subj = set(site_bin['subject_id'].unique())
print(f"Subjects with binary Site columns: {len(site_bin_subj)}")
print(f"Overlap text+binary: {len(site_text_subj & site_bin_subj)}")
print(f"Union text+binary: {len(site_text_subj | site_bin_subj)}")

# Show binary column values
for col in bin_cols:
    vals = hist[col].value_counts(dropna=False).head(5)
    print(f"\n  {col}:")
    print(f"    {vals.to_dict()}")

# Now let me also check: the onset_delta
onset_subj = set(hist[hist['Onset_Delta'].notna()]['subject_id'].unique())
diag_subj = set(hist[hist['Diagnosis_Delta'].notna()]['subject_id'].unique())
print(f"\nOnset_Delta subjects: {len(onset_subj)}")
print(f"Diagnosis_Delta subjects: {len(diag_subj)}")

# Combined site (either text or binary)
combined_site_subj = site_text_subj | site_bin_subj
print(f"\nCombined site subjects: {len(combined_site_subj)}")
print(f"Combined site + Onset: {len(combined_site_subj & onset_subj)}")

# Now let me redo the full intersection with combined site
fvc = pd.read_csv('01_data/raw/PROACT_FVC.csv')
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
vs = pd.read_csv('01_data/raw/PROACT_VITALSIGNS.csv')
alsfrs = pd.read_csv('01_data/raw/PROACT_ALSFRS.csv')
ril = pd.read_csv('01_data/raw/PROACT_RILUZOLE.csv')
death = pd.read_csv('01_data/raw/PROACT_DEATHDATA.csv')

# Compute FVC% 
has_liters_and_normal = fvc['Subject_Liters_Trial_1'].notna() & fvc['subject_normal'].notna()
fvc['fvc_pct'] = np.where(
    fvc['pct_of_Normal_Trial_1'].notna(),
    fvc['pct_of_Normal_Trial_1'],
    np.where(has_liters_and_normal, fvc['Subject_Liters_Trial_1'] / fvc['subject_normal'] * 100, np.nan)
)

fvc_subj = set(fvc[fvc['fvc_pct'].notna()]['subject_id'].unique())
age_subj = set(demo[demo['Age'].notna()]['subject_id'].unique())
sex_subj = set(demo[demo['Sex'].notna()]['subject_id'].unique())
height_subj = set(vs[vs['Height'].notna()]['subject_id'].unique())
weight_subj = set(vs[vs['Weight'].notna()]['subject_id'].unique())
alsfrs_subj = set(alsfrs['subject_id'].unique())
ril_subj = set(ril['subject_id'].unique())

# Progressive intersection
print("\n=== Progressive Intersection ===")
curr = fvc_subj.copy()
print(f"FVC: {len(curr)}")
curr &= onset_subj; print(f"+Onset: {len(curr)}")
curr &= combined_site_subj; print(f"+Site: {len(curr)}")
curr &= age_subj; print(f"+Age: {len(curr)}")
curr &= sex_subj; print(f"+Sex: {len(curr)}")
curr &= height_subj; print(f"+Height: {len(curr)}")
curr &= weight_subj; print(f"+Weight: {len(curr)}")
curr &= alsfrs_subj; print(f"+ALSFRS: {len(curr)}")

# How many with death data?
died_subj = set(death[death['Subject_Died'] == 'Yes']['subject_id'].unique())
alive_subj = set(death[death['Subject_Died'] == 'No']['subject_id'].unique())
death_days_subj = set(death[death['Death_Days'].notna()]['subject_id'].unique())

print(f"\nFull intersection: {len(curr)}")
print(f"  + died with Death_Days: {len(curr & death_days_subj)}")
print(f"  + alive (Subject_Died=No): {len(curr & alive_subj)}")
print(f"  + any death record: {len(curr & (died_subj | alive_subj))}")

# Let's also try WITHOUT requiring height (BMI might be optional or computed differently)
curr2 = fvc_subj & onset_subj & combined_site_subj & age_subj & sex_subj & weight_subj & alsfrs_subj
print(f"\nWithout Height: {len(curr2)}")
print(f"  + died with Death_Days: {len(curr2 & death_days_subj)}")

# Without FVC
curr3 = onset_subj & combined_site_subj & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"\nWithout FVC: {len(curr3)}")

# Check: which feature causes biggest drop?
base = onset_subj & combined_site_subj & age_subj & sex_subj & alsfrs_subj
print(f"\nBase (Onset+Site+Age+Sex+ALSFRS): {len(base)}")
print(f"  + FVC: {len(base & fvc_subj)}")
print(f"  + Height: {len(base & height_subj)}")
print(f"  + Weight: {len(base & weight_subj)}")
print(f"  + FVC + Height + Weight: {len(base & fvc_subj & height_subj & weight_subj)}")

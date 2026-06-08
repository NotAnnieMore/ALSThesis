import pandas as pd
import numpy as np

hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')
print("ALSHISTORY columns:", hist.columns.tolist())
print(f"Total rows: {len(hist)}")
print(f"Unique subjects: {hist['subject_id'].nunique()}")

# Check onset_delta vs site_of_onset overlap
has_onset = hist['Onset_Delta'].notna()
has_site = hist['Site_of_Onset'].notna()
print(f"\nRows with Onset_Delta: {has_onset.sum()}")
print(f"Rows with Site_of_Onset: {has_site.sum()}")
print(f"Rows with BOTH: {(has_onset & has_site).sum()}")

onset_subj = set(hist[has_onset]['subject_id'].unique())
site_subj = set(hist[has_site]['subject_id'].unique())
print(f"\nSubjects with Onset_Delta: {len(onset_subj)}")
print(f"Subjects with Site_of_Onset: {len(site_subj)}")
print(f"Subjects with BOTH: {len(onset_subj & site_subj)}")

# Let me see a few rows for subjects that have site but no onset, and vice versa
both = onset_subj & site_subj
only_onset = onset_subj - site_subj
only_site = site_subj - onset_subj

print(f"\nOnly Onset: {len(only_onset)}")
print(f"Only Site: {len(only_site)}")
print(f"Both: {len(both)}")

# Show some rows for subjects that have both
if both:
    sample_subj = list(both)[:3]
    for s in sample_subj:
        rows = hist[hist['subject_id'] == s]
        print(f"\n--- Subject {s} ---")
        print(rows.to_string())

# Show distinct values of Site_of_Onset
print(f"\n\nSite_of_Onset values:")
print(hist['Site_of_Onset'].value_counts(dropna=False))

# El Escorial - maybe site_of_onset is in another table?
el = pd.read_csv('01_data/raw/PROACT_ELESCORIAL.csv')
print(f"\n\nEL ESCORIAL columns: {el.columns.tolist()}")
print(f"El Escorial rows: {len(el)}")
print(f"El Escorial subjects: {el['subject_id'].nunique()}")

# Check if Onset info is in another table
# check if there's a column that has onset region/type
for col in el.columns:
    if col != 'subject_id':
        print(f"  {col}: {el[col].notna().sum()} non-null, unique: {el[col].nunique()}")
        if el[col].nunique() < 20:
            print(f"    values: {el[col].value_counts().to_dict()}")

# Maybe site of onset is also called onset_location or limb/bulbar in a different table
# Let me check all columns across key tables
import os
for f in ['PROACT_DEMOGRAPHICS.csv', 'PROACT_TREATMENT.csv']:
    df = pd.read_csv(f'01_data/raw/{f}')
    print(f"\n{f} columns: {df.columns.tolist()}")

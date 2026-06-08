import pandas as pd

# === FVC deep investigation ===
fvc = pd.read_csv('01_data/raw/PROACT_FVC.csv')
print("=== FVC UNITS ===")
print(fvc['Forced_Vital_Capacity_Units'].value_counts(dropna=False))

# Check: when units are "Percent Predicted", Subject_Liters_Trial_1 IS the pct
pct_pred = fvc[fvc['Forced_Vital_Capacity_Units'] == 'Percent Predicted']
print(f'\nPercent Predicted rows: {len(pct_pred)}')
print(f'  subjects: {pct_pred["subject_id"].nunique()}')
print(f'  Subject_Liters_Trial_1 range: {pct_pred["Subject_Liters_Trial_1"].min():.1f} - {pct_pred["Subject_Liters_Trial_1"].max():.1f}')

liters = fvc[fvc['Forced_Vital_Capacity_Units'] == 'Liters']
print(f'\nLiters rows: {len(liters)}')
print(f'  subjects: {liters["subject_id"].nunique()}')
print(f'  pct_of_Normal_Trial_1 available: {liters["pct_of_Normal_Trial_1"].notna().sum()}')

# When units is NaN
nan_units = fvc[fvc['Forced_Vital_Capacity_Units'].isna()]
print(f'\nNaN units rows: {len(nan_units)}')
print(f'  subjects: {nan_units["subject_id"].nunique()}')
print(f'  Subject_Liters range: {nan_units["Subject_Liters_Trial_1"].min():.3f} - {nan_units["Subject_Liters_Trial_1"].max():.3f}')
print(f'  Values > 20 (likely pct): {(nan_units["Subject_Liters_Trial_1"] > 20).sum()}')
# These could be percentages when the value is > 20 (no one has 20 liters of FVC)

# === DEMOGRAPHICS: alternative age sources ===
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
print('\n=== DEMOGRAPHICS ===')
print(f'Total subjects: {demo["subject_id"].nunique()}')
print(f'Age available: {demo["Age"].notna().sum()}')
print(f'Date_of_Birth available: {demo["Date_of_Birth"].notna().sum()}')

# Check Age consistency
demo_with_age = demo[demo['Age'].notna()]
print(f'Age range: {demo_with_age["Age"].min()} - {demo_with_age["Age"].max()}')

# === ALSHISTORY: check if Disease_Duration is directly available ===
hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')
print('\n=== ALS HISTORY columns ===')
print(f'Disease_Duration: {hist["Disease_Duration"].notna().sum()} non-null')
print(f'Onset_Delta: {hist["Onset_Delta"].notna().sum()} non-null')
print(f'Diagnosis_Delta: {hist["Diagnosis_Delta"].notna().sum()} non-null')

# Perhaps disease duration can be computed differently?
# Check Subject_ALS_History_Delta
print(f'Subject_ALS_History_Delta: {hist["Subject_ALS_History_Delta"].notna().sum()} non-null')

# How many have Onset_Delta?  
onset = hist.dropna(subset=['Onset_Delta'])
print(f'\nOnset_Delta subjects: {onset["subject_id"].nunique()}')
print(f'Onset_Delta stats (days): min={onset["Onset_Delta"].min()}, max={onset["Onset_Delta"].max()}, mean={onset["Onset_Delta"].mean():.0f}')
# Onset_Delta is negative (days before trial entry = diagnosis)
# Disease duration = |onset_delta| in days

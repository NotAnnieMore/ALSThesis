import pandas as pd

full = pd.read_csv('01_data/interim/step2_full_intermediate.csv')
fv = full[full['survival_group'].notna()].copy()

mask = fv['fvc_pct'].notna() & fv['bmi'].notna() & fv['site_of_onset'].notna()
sub = fv[mask]
print(f'FVC+BMI+site: {len(sub)}')
print(f'  age_at_diagnosis: {sub["age_at_diagnosis"].notna().sum()}')
print(f'  onset_delta: {sub["onset_delta"].notna().sum()}')
print(f'  disease_duration_months: {sub["disease_duration_months"].notna().sum()}')
print(f'  age_at_onset: {sub["age_at_onset"].notna().sum()}')

print(f'\n  Has age but no onset: {(sub["age_at_diagnosis"].notna() & sub["onset_delta"].isna()).sum()}')
print(f'  Has onset but no age: {(sub["age_at_diagnosis"].isna() & sub["onset_delta"].notna()).sum()}')
print(f'  Has neither: {(sub["age_at_diagnosis"].isna() & sub["onset_delta"].isna()).sum()}')
print(f'  Has slope_q1: {sub["slope_q1_speech"].notna().sum()}')

print(f'\n--- Full dataset overlaps ---')
has_bmi = fv['bmi'].notna()
has_onset = fv['onset_delta'].notna()
has_fvc = fv['fvc_pct'].notna()
has_age = fv['age_at_diagnosis'].notna()
print(f'BMI: {has_bmi.sum()}')
print(f'onset_delta: {has_onset.sum()}')
print(f'BMI + onset: {(has_bmi & has_onset).sum()}')
print(f'FVC + onset: {(has_fvc & has_onset).sum()}')
print(f'FVC + BMI + onset: {(has_fvc & has_bmi & has_onset).sum()}')
print(f'FVC + BMI + onset + age: {(has_fvc & has_bmi & has_onset & has_age).sum()}')

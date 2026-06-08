import pandas as pd, numpy as np
death = pd.read_csv('01_data/raw/PROACT_DEATHDATA.csv')
print('Death columns:', death.columns.tolist())
print(death.head(10).to_string())

dd = death['Death_Days'].dropna()
print(f'\nDeath_Days stats:')
print(f'  min: {dd.min()}, max: {dd.max()}, mean: {dd.mean():.0f}, median: {dd.median():.0f}')
print(f'  count: {len(dd)}')
print(f'  > 730 (24mo): {(dd > 730).sum()}')
print(f'  > 365 (12mo): {(dd > 365).sum()}')
print(f'  > 0: {(dd > 0).sum()}')
print(f'  < 0: {(dd < 0).sum()}')
print(f'  = 0: {(dd == 0).sum()}')

print(f'\nSubject_Died: {death["Subject_Died"].value_counts().to_dict()}')
died_yes = death[death['Subject_Died'] == 'Yes']
died_no = death[death['Subject_Died'] == 'No']
print(f'Died=Yes with Death_Days: {died_yes["Death_Days"].notna().sum()}')
print(f'Died=No with Death_Days: {died_no["Death_Days"].notna().sum()}')

# Check: for deceased patients, how does Death_Days relate to typical values?
# If from enrollment: should be ~months (100-1000 days typically)
# If from onset: should be larger (onset_delta + enrollment to death)
print(f'\nDied=Yes Death_Days: min={died_yes["Death_Days"].min()}, max={died_yes["Death_Days"].max()}, mean={died_yes["Death_Days"].mean():.0f}')

# Cross-reference with onset_delta for some patients
hist = pd.read_csv('01_data/raw/PROACT_ALSHISTORY.csv')
onset = hist.dropna(subset=['Onset_Delta']).groupby('subject_id')['Onset_Delta'].min().reset_index()
merged = pd.merge(died_yes[['subject_id', 'Death_Days']], onset, on='subject_id')
merged['survival_from_onset'] = merged['Death_Days'] - merged['Onset_Delta']
merged['survival_from_enrollment'] = merged['Death_Days']
print(f'\nMerged (died with onset_delta): {len(merged)}')
print(f'Survival from onset: min={merged["survival_from_onset"].min():.0f}, max={merged["survival_from_onset"].max():.0f}, mean={merged["survival_from_onset"].mean():.0f}')
print(f'Survival from enrollment: min={merged["survival_from_enrollment"].min():.0f}, max={merged["survival_from_enrollment"].max():.0f}, mean={merged["survival_from_enrollment"].mean():.0f}')

# How many are Short using enrollment vs onset?
thresh = 24 * 30.44
short_onset = (merged['survival_from_onset'] <= thresh).sum()
short_enroll = (merged['survival_from_enrollment'] <= thresh).sum()
print(f'\nShort (<=24mo) from onset: {short_onset}')
print(f'Short (<=24mo) from enrollment: {short_enroll}')

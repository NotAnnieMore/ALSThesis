import pandas as pd
vs = pd.read_csv('01_data/raw/PROACT_VITALSIGNS.csv')
has_units = vs['Height_Units'].notna()
has_height = vs['Height'].notna()
print(f'Height: {has_height.sum()}, Height_Units: {has_units.sum()}')
print(f'Units only (no Height): {(has_units & ~has_height).sum()}')
print(f'Height only (no Units): {(has_height & ~has_units).sum()}')
no_units = vs[has_height & ~has_units]['Height']
print(f'Height w/o units: min={no_units.min()}, max={no_units.max()}, mean={no_units.mean():.1f}')
print(f'Height_Units: {vs["Height_Units"].value_counts().to_dict()}')

# Subjects
print(f'\nSubjects with Height: {vs[has_height]["subject_id"].nunique()}')

# Maybe the issue is: Baseline_Weight column? -> implies there might be baseline height too?
# No Baseline_Height column exists. Let me check all columns containing 'height' or 'Height' 
h_cols = [c for c in vs.columns if 'eight' in c.lower()]
print(f'\nColumns with height/weight: {h_cols}')

# Actually, maybe I should look at DEMOGRAPHICS for height
demo = pd.read_csv('01_data/raw/PROACT_DEMOGRAPHICS.csv')
h_cols_demo = [c for c in demo.columns if 'eight' in c.lower()]
print(f'Demographics height cols: {h_cols_demo}')

# Check LABS for height
labs = pd.read_csv('01_data/raw/PROACT_LABS.csv')
h_labs = labs[labs['Test_Name'].str.contains('height|stature|BMI|bmi', case=False, na=False)]
print(f'\nLabs height/BMI rows: {len(h_labs)}')
if len(h_labs) > 0:
    print(h_labs['Test_Name'].value_counts())

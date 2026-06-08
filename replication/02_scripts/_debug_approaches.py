"""Final investigation: try different FVC approaches and find the best match to ~1967"""
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

# Common features
onset = hist.dropna(subset=['Onset_Delta']).groupby('subject_id')['Onset_Delta'].min().reset_index()
onset_dict = dict(zip(onset['subject_id'], onset['Onset_Delta']))
onset_subj = set(onset['subject_id'])

site_text = set(hist.dropna(subset=['Site_of_Onset'])['subject_id'].unique())
bin_cols = ['Site_of_Onset___Bulbar', 'Site_of_Onset___Limb', 'Site_of_Onset___Limb_and_Bulbar',
            'Site_of_Onset___Other', 'Site_of_Onset___Spine']
site_bin = set(hist[hist[bin_cols].notna().any(axis=1)]['subject_id'].unique())
combined_site = site_text | site_bin

age_subj = set(demo[demo['Age'].notna()]['subject_id'])
sex_subj = set(demo[demo['Sex'].notna()]['subject_id'])
height_subj = set(vs[vs['Height'].notna()]['subject_id'])
weight_subj = set(vs[vs['Weight'].notna()]['subject_id'])
alsfrs_subj = set(alsfrs['subject_id'])

# Death + last visit for target
death_dedup = death.drop_duplicates('subject_id')
died_dict = dict(zip(death_dedup['subject_id'], death_dedup['Subject_Died']))
dd_dict = dict(zip(death_dedup['subject_id'], death_dedup['Death_Days']))

max_d = []
for df, col in [(alsfrs, 'ALSFRS_Delta'), (fvc, 'Forced_Vital_Capacity_Delta'), 
                (svc, 'Slow_vital_Capacity_Delta'), (vs, 'Vital_Signs_Delta')]:
    md = df.groupby('subject_id')[col].max().reset_index()
    md.columns = ['subject_id', 'max_d']
    max_d.append(md)
lv_dict = pd.concat(max_d).groupby('subject_id')['max_d'].max().to_dict()

THRESHOLD = 24 * 30.44

def compute_target(subjects):
    short = nonshort = excluded = 0
    for sid in subjects:
        od = onset_dict.get(sid, np.nan)
        died = died_dict.get(sid, None)
        dd = dd_dict.get(sid, np.nan)
        lvd = lv_dict.get(sid, np.nan)
        if pd.isna(od): excluded += 1; continue
        if died == 'Yes' and pd.notna(dd):
            s = dd - od
            if s <= THRESHOLD: short += 1
            else: nonshort += 1
        else:
            if pd.notna(lvd):
                f = lvd - od
                if f > THRESHOLD: nonshort += 1
                else: excluded += 1
            else: excluded += 1
    return short, nonshort, excluded

base = onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
print(f"Base (without FVC): {len(base)}")
s, ns, ex = compute_target(base)
print(f"  Target: Short={s}, Non-Short={ns}, Excluded={ex}, Valid={s+ns}")

# Approach 1: FVC pct_of_Normal only (no liters/normal fallback)
fvc_pct_only = set(fvc[fvc['pct_of_Normal_Trial_1'].notna()]['subject_id'].unique())
a1 = base & fvc_pct_only
s, ns, ex = compute_target(a1)
print(f"\nA1 (pct_of_Normal only): {len(a1)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

# Approach 2: FVC pct + liters/normal fallback (current)
has_l_n = fvc['Subject_Liters_Trial_1'].notna() & fvc['subject_normal'].notna()
fvc['fvc_pct'] = np.where(fvc['pct_of_Normal_Trial_1'].notna(), fvc['pct_of_Normal_Trial_1'],
    np.where(has_l_n, fvc['Subject_Liters_Trial_1'] / fvc['subject_normal'] * 100, np.nan))
fvc_computed = set(fvc[fvc['fvc_pct'].notna()]['subject_id'].unique())
a2 = base & fvc_computed
s, ns, ex = compute_target(a2)
print(f"A2 (FVC pct+fallback): {len(a2)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

# Approach 3: SVC + FVC combined pct
svc_norm = 'Subject_Normal'
has_l_n_s = svc['Subject_Liters_Trial_1'].notna() & svc[svc_norm].notna()
svc['svc_pct'] = np.where(svc['pct_of_Normal_Trial_1'].notna(), svc['pct_of_Normal_Trial_1'],
    np.where(has_l_n_s, svc['Subject_Liters_Trial_1'] / svc[svc_norm] * 100, np.nan))
svc_computed = set(svc[svc['svc_pct'].notna()]['subject_id'].unique())
a3 = base & (fvc_computed | svc_computed)
s, ns, ex = compute_target(a3)
print(f"A3 (FVC+SVC pct): {len(a3)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

# Approach 4: Use raw liters (FVC only) — no % needed
fvc_liters = set(fvc[fvc['Subject_Liters_Trial_1'].notna()]['subject_id'].unique())
a4 = base & fvc_liters
s, ns, ex = compute_target(a4)
print(f"A4 (FVC liters raw): {len(a4)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

# Approach 5: FVC pct + SVC pct, NO height requirement (weight only)
base_no_h = onset_subj & combined_site & age_subj & sex_subj & weight_subj & alsfrs_subj
a5 = base_no_h & fvc_computed
s, ns, ex = compute_target(a5)
print(f"A5 (FVC pct, no Height): {len(a5)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

# Approach 6: FVC pct, no Riluzole requirement
ril = pd.read_csv('01_data/raw/PROACT_RILUZOLE.csv')
ril_subj = set(ril['subject_id'].unique())
base_no_r = onset_subj & combined_site & age_subj & sex_subj & height_subj & weight_subj & alsfrs_subj
# (already = base, riluzole wasn't in base)
# Let me check if riluzole was missing from some patients
a6_with_ril = base & fvc_computed & ril_subj
a6_without_ril = base & fvc_computed
print(f"\nRiluzole check: with={len(a6_with_ril)}, without={len(a6_without_ril)}, diff={len(a6_without_ril)-len(a6_with_ril)}")

# Approach 7: Try without requiring Diagnosis_Delta for disease duration
# Use onset_delta alone (which we already use — onset_delta IS available)
# But what about patients who have last_visit but not Death_Days?
# Let's check how many "Non-Short alive" patients are classified when we use
# max delta from ALL tables (including conmeds, treatment, etc.)

# Extra deltas
conmeds = pd.read_csv('01_data/raw/PROACT_CONMEDS.csv')
treat = pd.read_csv('01_data/raw/PROACT_TREATMENT.csv')
for df, col, name in [(conmeds, 'Concomitant_Medication_Delta', 'ConMeds'),
                      (treat, 'Treatment_Group_Delta', 'Treatment')]:
    if col in df.columns:
        md = df.groupby('subject_id')[col].max().reset_index()
        md.columns = ['subject_id', 'max_d']
        max_d.append(md)
        print(f"  {name} delta max subjects: {md['subject_id'].nunique()}")

lv_dict2 = pd.concat(max_d).groupby('subject_id')['max_d'].max().to_dict()

def compute_target2(subjects):
    short = nonshort = excluded = 0
    for sid in subjects:
        od = onset_dict.get(sid, np.nan)
        died = died_dict.get(sid, None)
        dd = dd_dict.get(sid, np.nan)
        lvd = lv_dict2.get(sid, np.nan)
        if pd.isna(od): excluded += 1; continue
        if died == 'Yes' and pd.notna(dd):
            s = dd - od
            if s <= THRESHOLD: short += 1
            else: nonshort += 1
        else:
            if pd.notna(lvd):
                f = lvd - od
                if f > THRESHOLD: nonshort += 1
                else: excluded += 1
            else: excluded += 1
    return short, nonshort, excluded

print(f"\n--- With extended last-visit deltas ---")
s, ns, ex = compute_target2(a2)
print(f"A2 extended: {len(a2)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

s, ns, ex = compute_target2(a3)
print(f"A3 extended: {len(a3)} all-feat, Valid={s+ns} (S={s}, NS={ns})")

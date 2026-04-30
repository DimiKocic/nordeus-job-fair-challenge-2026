import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')
import shap
import matplotlib.pyplot as plt

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ms_train = pd.read_csv(BASE_DIR / 'member_stats_training.csv')
ms_test  = pd.read_csv(BASE_DIR / 'member_stats_test.csv')
cm_train = pd.read_csv(BASE_DIR / 'clan_matches_training.csv')
cm_test  = pd.read_csv(BASE_DIR / 'clan_matches_test.csv')

# ── 2. CLEANING ───────────────────────────────────────────────────────────────
for ms in [ms_train, ms_test]:
    ms.loc[ms['avg_stars_top_11_players'] < 0, 'avg_stars_top_11_players'] = \
        ms.loc[ms['avg_stars_top_11_players'] > 0, 'avg_stars_top_11_players'].min()

# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
def build_clan_features(ms):
    ms = ms.copy()
    ms['bonus_x_stars'] = ms['avg_training_bonus'] * ms['avg_stars_top_11_players']
    grp = ms.groupby('clan_id')

    feats = pd.DataFrame({
        'mean_training_bonus' : grp['avg_training_bonus'].mean(),
        'min_training_bonus'  : grp['avg_training_bonus'].min(),
        'mean_stars'          : grp['avg_stars_top_11_players'].mean(),
        'min_stars'           : grp['avg_stars_top_11_players'].min(),
        'mean_active_7'       : grp['days_active_last_7_days'].mean(),
        'min_active_7'        : grp['days_active_last_7_days'].min(),
        'max_days_inactive'   : grp['days_since_last_active'].max(),
        'sum_bonus_x_stars'   : grp['bonus_x_stars'].sum(),
    }).reset_index()

    # identify weakest member by training bonus
    weakest = ms.loc[ms.groupby('clan_id')['avg_training_bonus'].idxmin()][
        ['clan_id', 'user_id', 'avg_training_bonus']
    ].rename(columns={
        'user_id'            : 'weakest_user_id',
        'avg_training_bonus' : 'weakest_bonus',
    })

    # identify least active member
    least_active = ms.loc[ms.groupby('clan_id')['days_active_last_7_days'].idxmin()][
        ['clan_id', 'user_id', 'days_active_last_7_days']
    ].rename(columns={
        'user_id'                : 'least_active_user_id',
        'days_active_last_7_days': 'least_active_days',
    })

    feats = feats.merge(weakest, on='clan_id', how='left')
    feats = feats.merge(least_active, on='clan_id', how='left')

    return feats

def build_match_features(cm, clan_feats):
    string_cols  = ['weakest_user_id', 'least_active_user_id']
    numeric_cols = [c for c in clan_feats.columns
                    if c != 'clan_id' and c not in string_cols]

    df = cm.merge(clan_feats.add_prefix('c1_').rename(columns={'c1_clan_id': 'clan_1_id'}), on='clan_1_id', how='left')
    df = df.merge(clan_feats.add_prefix('c2_').rename(columns={'c2_clan_id': 'clan_2_id'}), on='clan_2_id', how='left')

    for col in numeric_cols:
        df[f'diff_{col}'] = df[f'c1_{col}'] - df[f'c2_{col}']

    return df

clan_train = build_clan_features(ms_train)
clan_test  = build_clan_features(ms_test)
train_df   = build_match_features(cm_train, clan_train)
test_df    = build_match_features(cm_test,  clan_test)

meta_cols  = ['weakest_user_id', 'least_active_user_id', 'weakest_bonus', 'least_active_days']
drop_cols  = ['clan_1_id', 'clan_2_id', 'clan_1_points', 'clan_2_points', 'clan_winner']
meta_train = [c for c in train_df.columns if any(m in c for m in meta_cols)]
meta_test  = [c for c in test_df.columns  if any(m in c for m in meta_cols)]

X      = train_df.drop(columns=drop_cols + meta_train)
X_test = test_df.drop(columns=[c for c in drop_cols + meta_test if c in test_df.columns])

# ── 4. LOAD SAVED MODEL ───────────────────────────────────────────────────────
print("=== STEP 1: Loading Saved Model ===\n")
with open(BASE_DIR / 'best_model.pkl', 'rb') as f:
    automl = pickle.load(f)
print(f"Model loaded:  {automl.best_estimator}")
print(f"Best accuracy: {1 - automl.best_loss:.4f}")

# ── 5. SHAP ───────────────────────────────────────────────────────────────────
print("\n=== STEP 2: Computing SHAP Values ===\n")
best_model  = automl.model.estimator
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# global importance
shap_importance = pd.DataFrame({
    'feature'   : X.columns,
    'importance': np.abs(sv).mean(axis=0)
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("Global Feature Importance:\n")
print(shap_importance.to_string(index=False))

# summary plot
shap.summary_plot(sv, X, show=False)
plt.tight_layout()
plt.savefig(BASE_DIR / 'shap_summary.png', bbox_inches='tight', dpi=150)
plt.close()
print("\nSaved: shap_summary.png")

# ── 6. SHAP ON TEST SET ───────────────────────────────────────────────────────
shap_test = explainer.shap_values(X_test)
if isinstance(shap_test, list):
    sv_test = shap_test[1]
else:
    sv_test = shap_test

preds = automl.predict(X_test)

# ── 7. ADVISORY REPORT ───────────────────────────────────────────────────────
def generate_report(match_idx, test_df, sv_test, preds, cm_test):
    row       = test_df.iloc[match_idx]
    clan_1_id = cm_test.iloc[match_idx]['clan_1_id']
    clan_2_id = cm_test.iloc[match_idx]['clan_2_id']
    winner    = clan_1_id if preds[match_idx] == 1 else clan_2_id
    loser     = clan_2_id if preds[match_idx] == 1 else clan_1_id

    p = 'c1_' if preds[match_idx] == 2 else 'c2_'
    o = 'c2_' if preds[match_idx] == 2 else 'c1_'

    shap_dict = dict(zip(X_test.columns, sv_test[match_idx]))

    print(f"\n{'='*60}")
    print(f"  CLAN ADVISORY REPORT — Match {match_idx + 1}")
    print(f"{'='*60}")
    print(f"  {clan_1_id}  vs  {clan_2_id}")
    print(f"  Predicted Winner:     {winner}")
    print(f"  Needs Improvement:    {loser}")

    # training bonus
    loser_bonus     = row[f'{p}mean_training_bonus']
    winner_bonus    = row[f'{o}mean_training_bonus']
    weakest_user    = row[f'{p}weakest_user_id']
    weakest_val     = row[f'{p}weakest_bonus']

    print(f"\n  📊 TRAINING BONUS")
    print(f"  {loser} average:  {loser_bonus:.2f}")
    print(f"  {winner} average: {winner_bonus:.2f}")
    if loser_bonus < winner_bonus:
        print(f"  ❌ You are {winner_bonus - loser_bonus:.2f} points behind in training")
        print(f"  → Weakest member: {weakest_user} (bonus: {weakest_val:.2f})")
        print(f"  → Upgrade Attack, Defense, Possession & Condition bonuses")
    else:
        print(f"  ✅ Your training bonus is higher than your opponent")

    # activity
    loser_activity    = row[f'{p}mean_active_7']
    winner_activity   = row[f'{o}mean_active_7']
    least_active_user = row[f'{p}least_active_user_id']
    least_active_val  = row[f'{p}least_active_days']

    print(f"\n  📊 ACTIVITY (last 7 days)")
    print(f"  {loser} average:  {loser_activity:.1f}/7 days")
    print(f"  {winner} average: {winner_activity:.1f}/7 days")
    if loser_activity < winner_activity:
        print(f"  ❌ Your clan is less active than your opponent")
        print(f"  → Least active member: {least_active_user} ({least_active_val:.0f} days active)")
        print(f"  → Encourage all managers to log in daily before the match")
    else:
        print(f"  ✅ Your clan activity is higher than your opponent")

    # squad quality
    loser_stars     = row[f'{p}mean_stars']
    winner_stars    = row[f'{o}mean_stars']
    loser_min_stars = row[f'{p}min_stars']

    print(f"\n  📊 SQUAD QUALITY")
    print(f"  {loser} average stars:  {loser_stars:.2f}")
    print(f"  {winner} average stars: {winner_stars:.2f}")
    if loser_stars < winner_stars:
        print(f"  ❌ Squad quality gap of {winner_stars - loser_stars:.2f} stars")
        print(f"  → Your weakest player: {loser_min_stars:.2f} stars")
        print(f"  → Focus on upgrading your lowest ranked players first")
    else:
        print(f"  ✅ Your squad quality is competitive")

    # biggest shap factor
    top_feature = max(shap_dict, key=lambda k: abs(shap_dict[k]))
    top_value   = shap_dict[top_feature]
    print(f"\n  🔑 BIGGEST FACTOR IN THIS PREDICTION")
    print(f"  Feature: {top_feature}")
    print(f"  Impact:  {top_value:+.4f} ({'favors ' + winner if top_value > 0 else 'favors ' + loser})")
    print(f"{'='*60}\n")

    return {
        'training_gap' : loser_bonus < winner_bonus,
        'activity_gap' : loser_activity < winner_activity,
        'quality_gap'  : loser_stars < winner_stars,
    }

# ── 8. GENERATE REPORTS ───────────────────────────────────────────────────────
print("\n=== STEP 3: Advisory Reports (10 examples) ===")

# show 5 predicted losses and 5 predicted wins for variety
losses = [i for i, p in enumerate(preds) if p == 2][:5]
wins   = [i for i, p in enumerate(preds) if p == 1][:5]
examples = losses + wins

all_gaps = []
for i in examples:
    gaps = generate_report(i, test_df, sv_test, preds, cm_test)
    all_gaps.append(gaps)

# ── 9. POPULATION SUMMARY ────────────────────────────────────────────────────
print("\n=== STEP 4: Population-Level Summary (all test matches) ===\n")

summary = []
for i in range(len(preds)):
    row    = test_df.iloc[i]
    p      = 'c1_' if preds[i] == 2 else 'c2_'
    o      = 'c2_' if preds[i] == 2 else 'c1_'
    summary.append({
        'training_gap': row[f'{p}mean_training_bonus'] < row[f'{o}mean_training_bonus'],
        'activity_gap': row[f'{p}mean_active_7']       < row[f'{o}mean_active_7'],
        'quality_gap' : row[f'{p}mean_stars']           < row[f'{o}mean_stars'],
    })

summary_df = pd.DataFrame(summary)
total      = len(summary_df)

print(f"  Total test matches: {total}")
print(f"\n  Most common reasons predicted losers are losing:\n")
print(f"  1. Training bonus gap  — {summary_df['training_gap'].sum()} matches ({summary_df['training_gap'].mean()*100:.1f}%)")
print(f"  2. Activity gap        — {summary_df['activity_gap'].sum()} matches ({summary_df['activity_gap'].mean()*100:.1f}%)")
print(f"  3. Squad quality gap   — {summary_df['quality_gap'].sum()} matches ({summary_df['quality_gap'].mean()*100:.1f}%)")
print(f"\n  Clans losing on ALL THREE dimensions: {(summary_df.all(axis=1)).sum()} ({(summary_df.all(axis=1)).mean()*100:.1f}%)")
print(f"  Clans losing on training bonus ONLY:  {((summary_df['training_gap']) & (~summary_df['activity_gap']) & (~summary_df['quality_gap'])).sum()}")
print(f"  Clans losing on activity ONLY:        {((~summary_df['training_gap']) & (summary_df['activity_gap']) & (~summary_df['quality_gap'])).sum()}")
print(f"  Clans losing on quality ONLY:         {((~summary_df['training_gap']) & (~summary_df['activity_gap']) & (summary_df['quality_gap'])).sum()}")
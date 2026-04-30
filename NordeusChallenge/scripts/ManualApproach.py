import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')
from flaml import AutoML

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
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
    return feats

def build_match_features(cm, clan_feats):
    feat_cols = [c for c in clan_feats.columns if c != 'clan_id']
    df = cm.merge(clan_feats.add_prefix('c1_').rename(columns={'c1_clan_id': 'clan_1_id'}), on='clan_1_id', how='left')
    df = df.merge(clan_feats.add_prefix('c2_').rename(columns={'c2_clan_id': 'clan_2_id'}), on='clan_2_id', how='left')
    for col in feat_cols:
        df[f'diff_{col}'] = df[f'c1_{col}'] - df[f'c2_{col}']
    return df

print("=== STEP 1: Feature Engineering ===\n")
clan_train = build_clan_features(ms_train)
clan_test  = build_clan_features(ms_test)
train_df   = build_match_features(cm_train, clan_train)
test_df    = build_match_features(cm_test,  clan_test)

drop_cols = ['clan_1_id', 'clan_2_id', 'clan_1_points', 'clan_2_points', 'clan_winner']
X         = train_df.drop(columns=drop_cols)
y         = train_df['clan_winner']

print(f"Feature matrix: {X.shape}")
print(f"Features: {list(X.columns)}\n")

# ── 4. FLAML ──────────────────────────────────────────────────────────────────
print("=== STEP 2: FLAML AutoML ===\n")
automl = AutoML()
automl.fit(
    X, y,
    task           = 'classification',
    time_budget    = 600,
    metric         = 'accuracy',
    seed           = 42,
    verbose        = 1,
    estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree', 'svc'],
)

# ── 5. VALIDATION ─────────────────────────────────────────────────────────────
print("\n=== STEP 3: Validation ===\n")
print(f"Best model:    {automl.best_estimator}")
print(f"Best accuracy: {1 - automl.best_loss:.4f}")
print(f"Best config:   {automl.best_config}")

# ── 6. SAVE MODEL ─────────────────────────────────────────────────────────────
print("\n=== STEP 4: Saving Model ===\n")
with open(BASE_DIR / 'best_model.pkl', 'wb') as f:
    pickle.dump(automl, f)
print("Model saved: best_model.pkl")

# ── 7. PREDICTIONS ────────────────────────────────────────────────────────────
print("\n=== STEP 5: Generating Predictions ===\n")
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
preds  = automl.predict(X_test)

submission = pd.DataFrame({
    'clan_1_id'            : cm_test['clan_1_id'],
    'clan_2_id'            : cm_test['clan_2_id'],
    'predicted_clan_winner': preds
})

submission.to_csv(BASE_DIR / 'predictions_manual.csv', index=False)
print(f"Predictions saved. Shape: {submission.shape}")
print(submission.head())
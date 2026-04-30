import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from flaml import AutoML

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ms_train = pd.read_csv(BASE_DIR / 'member_stats_training.csv')
ms_test  = pd.read_csv(BASE_DIR / 'member_stats_test.csv')
cm_train = pd.read_csv(BASE_DIR / 'clan_matches_training.csv')
cm_test  = pd.read_csv(BASE_DIR / 'clan_matches_test.csv')

# ── 2. CLEANING ───────────────────────────────────────────────────────────────
ms_train.loc[ms_train['avg_stars_top_11_players'] < 0, 'avg_stars_top_11_players'] = \
    ms_train.loc[ms_train['avg_stars_top_11_players'] > 0, 'avg_stars_top_11_players'].min()

# ── 3. SIMPLE MEAN AGGREGATION ────────────────────────────────────────────────
MEMBER_FEATURES = [
    'days_active_last_28_days',
    'days_active_last_7_days',
    'days_since_last_active',
    'clan_multiplier',
    'avg_stars_top_11_players',
    'avg_stars_top_3_players',
    'avg_training_bonus',
]

def aggregate_clan(ms):
    return ms.groupby('clan_id')[MEMBER_FEATURES].mean().reset_index()

def build_match_features(cm, clan_feats):
    df = cm.merge(clan_feats.add_prefix('c1_').rename(columns={'c1_clan_id': 'clan_1_id'}), on='clan_1_id', how='left')
    df = df.merge(clan_feats.add_prefix('c2_').rename(columns={'c2_clan_id': 'clan_2_id'}), on='clan_2_id', how='left')
    for col in MEMBER_FEATURES:
        df[f'diff_{col}'] = df[f'c1_{col}'] - df[f'c2_{col}']
    return df

clan_train = aggregate_clan(ms_train)
clan_test  = aggregate_clan(ms_test)
train_df   = build_match_features(cm_train, clan_train)
test_df    = build_match_features(cm_test,  clan_test)

drop_cols = ['clan_1_id', 'clan_2_id', 'clan_1_points', 'clan_2_points', 'clan_winner']
X         = train_df.drop(columns=drop_cols)
y         = train_df['clan_winner']

# ── 4. RANDOM FOREST FEATURE IMPORTANCE ──────────────────────────────────────
print("=== STEP 1: Random Forest Feature Importance ===\n")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importance = pd.DataFrame({
    'feature'   : X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print(importance.to_string(index=False))

# ── 5. SELECT TOP FEATURES ────────────────────────────────────────────────────
threshold    = importance['importance'].mean()
top_features = importance[importance['importance'] >= threshold]['feature'].tolist()

print(f"\nThreshold (mean importance): {threshold:.4f}")
print(f"Selected {len(top_features)} features out of {len(X.columns)}")
print(f"\nSelected features:\n{top_features}")

X_selected      = X[top_features]
X_test_selected = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])[top_features]

# ── 6. FLAML ──────────────────────────────────────────────────────────────────
print("\n=== STEP 2: FLAML AutoML ===\n")
automl = AutoML()
automl.fit(
    X_selected, y,
    task           = 'classification',
    time_budget    = 600,
    metric         = 'accuracy',
    seed           = 42,
    verbose        = 1,
    estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree'],
)

# ── 7. VALIDATION ─────────────────────────────────────────────────────────────
print("\n=== STEP 3: Validation ===\n")
print(f"Best model:    {automl.best_estimator}")
print(f"Best accuracy: {1 - automl.best_loss:.4f}")
print(f"Best config:   {automl.best_config}")

# ── 8. PREDICTIONS ────────────────────────────────────────────────────────────
print("\n=== STEP 4: Generating Predictions ===\n")
preds = automl.predict(X_test_selected)

submission = pd.DataFrame({
    'clan_1_id'            : cm_test['clan_1_id'],
    'clan_2_id'            : cm_test['clan_2_id'],
    'predicted_clan_winner': preds
})

submission.to_csv(BASE_DIR / 'predictions_auto.csv', index=False)
print(f"Predictions saved. Shape: {submission.shape}")
print(submission.head())
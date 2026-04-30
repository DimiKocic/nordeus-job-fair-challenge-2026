import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ms_train = pd.read_csv(BASE_DIR / 'member_stats_training.csv')
ms_test  = pd.read_csv(BASE_DIR / 'member_stats_test.csv')
cm_train = pd.read_csv(BASE_DIR / 'clan_matches_training.csv')
cm_test  = pd.read_csv(BASE_DIR / 'clan_matches_test.csv')

# ── 2. DECLARE FEATURES AND TARGET ───────────────────────────────────────────
MEMBER_FEATURES = [
    'days_active_last_28_days',
    'days_active_last_7_days',
    'days_since_last_active',
    'clan_multiplier',
    'avg_stars_top_11_players',
    'avg_stars_top_3_players',
    'avg_training_bonus',
]

MEMBER_ID_COLS = ['clan_id', 'user_id']
MATCH_ID_COLS  = ['clan_1_id', 'clan_2_id']
TARGET         = 'clan_winner'

# columns to skip for IQR (discrete/categorical by nature)
IQR_SKIP_COLS  = ['clan_multiplier']

# ── 3. CLEANING RULES ────────────────────────────────────────────────────────
class DataCleaner:

    def __init__(self, rules):
        self.rules = rules
        self.report = {}

    def check_missing(self, df, name):
        missing = df.isnull().sum().sum()
        self.report[name]['missing_values'] = missing
        if missing > 0:
            df = df.dropna()
            print(f"  [{name}] missing values found: {missing} — rows dropped")
        else:
            print(f"  [{name}] missing values: 0 — nothing to do")
        return df

    def check_duplicates(self, df, name):
        dups = df.duplicated().sum()
        self.report[name]['duplicates'] = dups
        if dups > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"  [{name}] duplicates found: {dups} — rows dropped")
        else:
            print(f"  [{name}] duplicates: 0 — nothing to do")
        return df

    def fix_negative_outliers_iqr(self, df, name):
        num_cols = [c for c in df.select_dtypes(include='number').columns
                    if c not in IQR_SKIP_COLS]
        total_fixed = 0
        for col in num_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                # calculate IQR lower bound
                Q1  = df[col].quantile(0.25)
                Q3  = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR

                # if lower bound is still negative use minimum positive value
                if lower_bound < 0:
                    lower_bound = df.loc[df[col] > 0, col].min()

                df.loc[df[col] < 0, col] = lower_bound
                total_fixed += neg_count
                print(f"  [{name}] {col}: {neg_count} negative value(s) replaced with {lower_bound:.4f}")

        if total_fixed == 0:
            print(f"  [{name}] negative outliers: 0 — nothing to do")

        self.report[name]['negative_outliers_fixed'] = total_fixed
        return df

    def clean(self, df, name):
        self.report[name] = {}
        print(f"\n--- Cleaning: {name} (shape: {df.shape}) ---")
        for rule in self.rules:
            if rule == 'missing_values':
                df = self.check_missing(df, name)
            elif rule == 'duplicates':
                df = self.check_duplicates(df, name)
            elif rule == 'fix_negative_outliers_iqr':
                df = self.fix_negative_outliers_iqr(df, name)
        print(f"  [{name}] final shape: {df.shape}")
        return df

    def summary(self):
        print("\n=== CLEANING SUMMARY ===")
        for dataset, findings in self.report.items():
            print(f"\n  {dataset}:")
            for rule, count in findings.items():
                print(f"    {rule}: {count}")


# apply cleaner to all four datasets
cleaner = DataCleaner(rules=['missing_values', 'duplicates', 'fix_negative_outliers_iqr'])

ms_train = cleaner.clean(ms_train, 'ms_train')
ms_test  = cleaner.clean(ms_test,  'ms_test')
cm_train = cleaner.clean(cm_train, 'cm_train')
cm_test  = cleaner.clean(cm_test,  'cm_test')

cleaner.summary()

# Clan Victory Predictor
### Nordeus Job Fair 2026 — Data Science Challenge

Predicting which association wins a weekend tournament match in **Top Eleven** using member statistics, and providing actionable coaching advice via a SHAP-powered advisory assistant.

---

## Results

| Approach | Best Model | Accuracy |
|---|---|---|
| Automated (`auto_approach.py`) | XGBoost | 58.00% |
| Manual (`ManualApproach.py`) | XGBoost | 58.99% |

Manual feature engineering outperformed automated selection — domain knowledge of the game mechanics (clan multipliers, training bonuses, positional matchups) added predictive value.

---

## Repository Structure

```
NordeusChallenge/
│
├── README.md
├── predictions_manual.csv          ← final submission file
│
├── scripts/
│   ├── CleaningPipeline.py
│   ├── auto_approach.py
│   ├── ManualApproach.py
│   ├── ShapAnalysis.py
│   └── GenerateReport.py
│
└── report/
    ├── clan_report.html            ← open in browser for full advisory report
    └── shap_summary.png
```

---

## How to Run

**Requirements:**
```
pip install pandas numpy scikit-learn flaml[automl] shap lightgbm xgboost matplotlib
```

Place the four CSV data files in the root directory, then run in order:

```
python scripts/ManualApproach.py     # trains model, saves predictions
python scripts/ShapAnalysis.py       # SHAP analysis and advisory reports
python scripts/GenerateReport.py     # generates clan_report.html
```

---

## Bonus Task — Advisory Assistant

The advisory assistant uses SHAP values to give each association plain-English coaching advice — identifying whether the whole association needs to improve or just one specific manager, and what to focus on (training bonuses, activity, squad quality).

Open `report/clan_report.html` in any browser to see the full interactive report.

---

## Data

Datasets not included due to file size. Place these in the root directory before running:
`member_stats_training.csv`, `member_stats_test.csv`, `clan_matches_training.csv`, `clan_matches_test.csv`

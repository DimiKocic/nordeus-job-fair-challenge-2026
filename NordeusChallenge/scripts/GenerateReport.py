import pandas as pd
import numpy as np
import warnings
import pickle
import base64
from pathlib import Path
warnings.filterwarnings('ignore')
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ms_train = pd.read_csv(BASE_DIR / 'member_stats_training.csv')
ms_test  = pd.read_csv(BASE_DIR / 'member_stats_test.csv')
cm_train = pd.read_csv(BASE_DIR / 'clan_matches_training.csv')
cm_test  = pd.read_csv(BASE_DIR / 'clan_matches_test.csv')

for ms in [ms_train, ms_test]:
    ms.loc[ms['avg_stars_top_11_players'] < 0, 'avg_stars_top_11_players'] = \
        ms.loc[ms['avg_stars_top_11_players'] > 0, 'avg_stars_top_11_players'].min()

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
    weakest = ms.loc[ms.groupby('clan_id')['avg_training_bonus'].idxmin()][
        ['clan_id','user_id','avg_training_bonus']
    ].rename(columns={'user_id':'weakest_user_id','avg_training_bonus':'weakest_bonus'})
    least_active = ms.loc[ms.groupby('clan_id')['days_active_last_7_days'].idxmin()][
        ['clan_id','user_id','days_active_last_7_days']
    ].rename(columns={'user_id':'least_active_user_id','days_active_last_7_days':'least_active_days'})
    feats = feats.merge(weakest, on='clan_id', how='left')
    feats = feats.merge(least_active, on='clan_id', how='left')
    return feats

def build_match_features(cm, clan_feats):
    string_cols  = ['weakest_user_id','least_active_user_id']
    numeric_cols = [c for c in clan_feats.columns if c != 'clan_id' and c not in string_cols]
    df = cm.merge(clan_feats.add_prefix('c1_').rename(columns={'c1_clan_id':'clan_1_id'}), on='clan_1_id', how='left')
    df = df.merge(clan_feats.add_prefix('c2_').rename(columns={'c2_clan_id':'clan_2_id'}), on='clan_2_id', how='left')
    for col in numeric_cols:
        df[f'diff_{col}'] = df[f'c1_{col}'] - df[f'c2_{col}']
    return df

clan_train = build_clan_features(ms_train)
clan_test  = build_clan_features(ms_test)
train_df   = build_match_features(cm_train, clan_train)
test_df    = build_match_features(cm_test,  clan_test)

meta_cols  = ['weakest_user_id','least_active_user_id','weakest_bonus','least_active_days']
drop_cols  = ['clan_1_id','clan_2_id','clan_1_points','clan_2_points','clan_winner']
meta_train = [c for c in train_df.columns if any(m in c for m in meta_cols)]
meta_test  = [c for c in test_df.columns  if any(m in c for m in meta_cols)]
X          = train_df.drop(columns=drop_cols + meta_train)
X_test     = test_df.drop(columns=[c for c in drop_cols + meta_test if c in test_df.columns])

# ── 2. LOAD MODEL ─────────────────────────────────────────────────────────────
print("Loading model...")
with open(BASE_DIR / 'best_model.pkl', 'rb') as f:
    automl = pickle.load(f)

best_model  = automl.model.estimator
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)
sv          = shap_values[1] if isinstance(shap_values, list) else shap_values

shap_importance = pd.DataFrame({
    'feature'   : X.columns,
    'importance': np.abs(sv).mean(axis=0)
}).sort_values('importance', ascending=False).reset_index(drop=True)

# ── 3. SHAP PLOT ──────────────────────────────────────────────────────────────
print("Generating SHAP plot...")
shap.summary_plot(sv, X, show=False, plot_size=(10,6))
plt.tight_layout()
plt.savefig('/tmp/shap_plot.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()
with open('/tmp/shap_plot.png','rb') as f:
    shap_img_b64 = base64.b64encode(f.read()).decode()

# ── 4. SHAP ON TEST ───────────────────────────────────────────────────────────
shap_test = explainer.shap_values(X_test)
sv_test   = shap_test[1] if isinstance(shap_test, list) else shap_test
preds     = automl.predict(X_test)

# ── 5. POPULATION SUMMARY ────────────────────────────────────────────────────
print("Computing population summary...")
summary = []
for i in range(len(preds)):
    row = test_df.iloc[i]
    p   = 'c1_' if preds[i] == 2 else 'c2_'
    o   = 'c2_' if preds[i] == 2 else 'c1_'
    summary.append({
        'training_gap': row[f'{p}mean_training_bonus'] < row[f'{o}mean_training_bonus'],
        'activity_gap': row[f'{p}mean_active_7']       < row[f'{o}mean_active_7'],
        'quality_gap' : row[f'{p}mean_stars']           < row[f'{o}mean_stars'],
    })
summary_df   = pd.DataFrame(summary)
total        = len(summary_df)
training_pct = summary_df['training_gap'].mean() * 100
activity_pct = summary_df['activity_gap'].mean() * 100
quality_pct  = summary_df['quality_gap'].mean()  * 100
all_three    = summary_df.all(axis=1).mean()      * 100
only_bonus   = ((summary_df['training_gap']) & (~summary_df['activity_gap']) & (~summary_df['quality_gap'])).sum()
only_activity= ((~summary_df['training_gap']) & (summary_df['activity_gap']) & (~summary_df['quality_gap'])).sum()
only_quality = ((~summary_df['training_gap']) & (~summary_df['activity_gap']) & (summary_df['quality_gap'])).sum()

# ── 6. NATURAL LANGUAGE ADVISORY ─────────────────────────────────────────────
FEATURE_LABELS = {
    'diff_min_training_bonus' : 'one manager has critically low training bonuses',
    'diff_mean_training_bonus': 'overall association training level is too low',
    'diff_sum_bonus_x_stars'  : 'combined quality and preparation is insufficient',
    'diff_mean_stars'         : 'overall squad quality is lower than opponent',
    'diff_min_stars'          : 'weakest player matchup is a major disadvantage',
    'diff_max_days_inactive'  : 'at least one manager has been inactive too long',
    'diff_mean_active_7'      : 'association engagement this week is too low',
    'diff_min_active_7'       : 'one manager has barely been active this week',
}

def get_top_reason(shap_dict):
    top = max(shap_dict, key=lambda k: abs(shap_dict[k]))
    return FEATURE_LABELS.get(top, top.replace('_', ' '))

def training_advice_html(loser, winner, loser_bonus, winner_bonus, weakest_user, weakest_val, loser_min_bonus):
    gap = winner_bonus - loser_bonus
    if loser_bonus >= winner_bonus:
        return f"""
        <div class="metric-block positive">
            <div class="metric-title">⚡ Training Preparation</div>
            <div class="metric-text">Your association is well prepared — your average training bonus of
            <strong>{loser_bonus:.1f}</strong> is higher than your opponent's <strong>{winner_bonus:.1f}</strong>.
            Keep up the training routine before the match.</div>
        </div>"""

    if weakest_val == 0:
        member_msg = f"""One of your managers, <strong>{weakest_user}</strong>, has done <strong>zero training</strong>
        this season. This is your biggest vulnerability — an untrained manager will lose their matchup and gift free points
        to the opponent. Prioritise getting this manager to upgrade their Attack, Defense, Possession and Condition bonuses immediately."""
    elif weakest_val < loser_bonus * 0.5:
        member_msg = f"""One manager in particular, <strong>{weakest_user}</strong>, is significantly under-trained
        (bonus: <strong>{weakest_val:.1f}</strong>) compared to your association average of {loser_bonus:.1f}.
        This imbalance is costing your whole association — focus on bringing this manager up to the group level."""
    else:
        member_msg = f"""The entire association needs to invest more in training bonuses. Your weakest member
        <strong>{weakest_user}</strong> has a bonus of <strong>{weakest_val:.1f}</strong> — everyone should
        work on upgrading Attack, Defense, Possession and Condition before the match."""

    return f"""
    <div class="metric-block negative">
        <div class="metric-title">⚡ Training Preparation — Action Required</div>
        <div class="metric-text">Your association's average training bonus is <strong>{loser_bonus:.1f}</strong>
        compared to your opponent's <strong>{winner_bonus:.1f}</strong> — a gap of <strong>{gap:.1f} points</strong>.
        This is the area where you can improve most before the match.</div>
        <div class="metric-advice">{member_msg}</div>
    </div>"""

def activity_advice_html(loser, winner, loser_act, winner_act, least_user, least_val, loser_min_act):
    if loser_act >= winner_act:
        return f"""
        <div class="metric-block positive">
            <div class="metric-title">🏃 Association Engagement</div>
            <div class="metric-text">Your association is more engaged than your opponent — averaging
            <strong>{loser_act:.1f} out of 7 days</strong> active this week vs their <strong>{winner_act:.1f}</strong>.
            Active managers make better decisions during matches.</div>
        </div>"""

    if least_val == 0:
        member_msg = f"""<strong>{least_user}</strong> has not logged in at all this week. An absent manager
        cannot prepare their team, adjust tactics, or make the most of their players.
        Reach out and make sure all 6 managers are active before the tournament begins."""
    else:
        member_msg = f"""Your least active manager, <strong>{least_user}</strong>, has only been active
        <strong>{least_val:.0f} out of 7 days</strong> this week. Encourage every member to log in daily —
        even small improvements in engagement have a measurable impact on match outcomes."""

    return f"""
    <div class="metric-block negative">
        <div class="metric-title">🏃 Association Engagement — Action Required</div>
        <div class="metric-text">Your association has averaged <strong>{loser_act:.1f} active days</strong>
        this week vs your opponent's <strong>{winner_act:.1f}</strong>. More engaged associations
        consistently outperform less active ones.</div>
        <div class="metric-advice">{member_msg}</div>
    </div>"""

def quality_advice_html(loser, winner, loser_stars, winner_stars, loser_min_stars):
    gap = winner_stars - loser_stars
    if loser_stars >= winner_stars:
        return f"""
        <div class="metric-block positive">
            <div class="metric-title">⭐ Squad Quality</div>
            <div class="metric-text">Your squad quality is competitive — your association averages
            <strong>{loser_stars:.2f} stars</strong> vs your opponent's <strong>{winner_stars:.2f}</strong>.
            Quality matchups are in your favour.</div>
        </div>"""

    if gap > 1.5:
        urgency = f"""This is a significant quality gap of <strong>{gap:.2f} stars</strong>. Squad quality
        takes time and investment to build — this is a long-term priority for your association."""
    elif gap > 0.5:
        urgency = f"""There is a noticeable gap of <strong>{gap:.2f} stars</strong>. Focus on upgrading
        your weakest players first — your lowest ranked manager currently has only
        <strong>{loser_min_stars:.2f} stars</strong> which puts them at a disadvantage in their matchup."""
    else:
        urgency = f"""The quality gap is small (<strong>{gap:.2f} stars</strong>). A focused investment
        in your weakest player (currently <strong>{loser_min_stars:.2f} stars</strong>) could close
        this gap quickly."""

    return f"""
    <div class="metric-block negative">
        <div class="metric-title">⭐ Squad Quality — Improvement Needed</div>
        <div class="metric-text">Your association averages <strong>{loser_stars:.2f} stars</strong>
        while your opponent fields <strong>{winner_stars:.2f} stars</strong>. {urgency}</div>
    </div>"""

def match_report_html(match_idx):
    row       = test_df.iloc[match_idx]
    clan_1_id = cm_test.iloc[match_idx]['clan_1_id']
    clan_2_id = cm_test.iloc[match_idx]['clan_2_id']
    winner    = clan_1_id if preds[match_idx] == 1 else clan_2_id
    loser     = clan_2_id if preds[match_idx] == 1 else clan_1_id
    p         = 'c1_' if preds[match_idx] == 2 else 'c2_'
    o         = 'c2_' if preds[match_idx] == 2 else 'c1_'
    shap_dict = dict(zip(X_test.columns, sv_test[match_idx]))
    top_reason = get_top_reason(shap_dict)

    training_html = training_advice_html(
        loser, winner,
        row[f'{p}mean_training_bonus'], row[f'{o}mean_training_bonus'],
        row[f'{p}weakest_user_id'], row[f'{p}weakest_bonus'],
        row[f'{p}min_training_bonus']
    )
    activity_html = activity_advice_html(
        loser, winner,
        row[f'{p}mean_active_7'], row[f'{o}mean_active_7'],
        row[f'{p}least_active_user_id'], row[f'{p}least_active_days'],
        row[f'{p}min_active_7']
    )
    quality_html = quality_advice_html(
        loser, winner,
        row[f'{p}mean_stars'], row[f'{o}mean_stars'],
        row[f'{p}min_stars']
    )

    return f"""
    <div class="match-card">
        <div class="match-header">
            <div class="match-title">Match {match_idx + 1}</div>
            <div class="match-clans">
                <span class="clan-tag {'winner-tag' if preds[match_idx]==1 else 'loser-tag'}">{clan_1_id}</span>
                <span class="vs">vs</span>
                <span class="clan-tag {'winner-tag' if preds[match_idx]==2 else 'loser-tag'}">{clan_2_id}</span>
            </div>
            <div class="prediction-badge">🏆 Predicted Winner: <strong>{winner}</strong></div>
        </div>
        <div class="advice-header">
            Association Advisory for <strong>{loser}</strong> —
            The primary reason you are predicted to lose is that <em>{top_reason}</em>.
        </div>
        <div class="metrics">
            {training_html}
            {activity_html}
            {quality_html}
        </div>
    </div>"""

# importance table
importance_rows = ''
for _, r in shap_importance.head(8).iterrows():
    bar_width = int(r['importance'] / shap_importance['importance'].max() * 100)
    label = FEATURE_LABELS.get(r['feature'], r['feature'].replace('_',' '))
    importance_rows += f"""
    <tr>
        <td>{label}</td>
        <td>{r['importance']:.4f}</td>
        <td><div class="bar" style="width:{bar_width}%"></div></td>
    </tr>"""

losses       = [i for i, p in enumerate(preds) if p == 2][:5]
wins         = [i for i, p in enumerate(preds) if p == 1][:5]
matches_html = ''.join([match_report_html(i) for i in losses + wins])

# ── 7. BUILD HTML ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clan Victory Predictor — Nordeus Job Fair 2026</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0a0e1a;
    --surface:  #111827;
    --surface2: #1a2235;
    --accent:   #f97316;
    --accent2:  #3b82f6;
    --win:      #22c55e;
    --lose:     #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --border:   #1e293b;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'DM Mono',monospace; font-size:14px; line-height:1.7; }}

  .hero {{
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 60px 40px 50px;
    position: relative; overflow: hidden;
  }}
  .hero::before {{
    content:''; position:absolute; top:-100px; right:-100px;
    width:400px; height:400px;
    background:radial-gradient(circle, rgba(249,115,22,0.08) 0%, transparent 70%);
    border-radius:50%;
  }}
  .hero-label {{ font-family:'Syne',sans-serif; font-size:11px; letter-spacing:4px; text-transform:uppercase; color:var(--accent); margin-bottom:16px; }}
  .hero h1 {{
    font-family:'Syne',sans-serif; font-size:clamp(32px,5vw,56px); font-weight:800; line-height:1.1; margin-bottom:16px;
    background:linear-gradient(135deg,#fff 0%,var(--accent) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }}
  .hero-sub {{ color:var(--muted); max-width:600px; font-size:15px; }}

  .container {{ max-width:1100px; margin:0 auto; padding:40px; }}
  .section {{ margin-bottom:60px; }}
  .section-title {{
    font-family:'Syne',sans-serif; font-size:11px; letter-spacing:4px; text-transform:uppercase;
    color:var(--accent); margin-bottom:24px; display:flex; align-items:center; gap:12px;
  }}
  .section-title::after {{ content:''; flex:1; height:1px; background:var(--border); }}

  .stats-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin-bottom:40px; }}
  .stat-card {{
    background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px;
    position:relative; overflow:hidden;
  }}
  .stat-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--accent); }}
  .stat-value {{ font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:var(--accent); line-height:1; margin-bottom:8px; }}
  .stat-label {{ color:var(--muted); font-size:12px; }}

  .table-wrap {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ background:var(--surface2); padding:12px 16px; text-align:left; font-family:'Syne',sans-serif; font-size:11px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); }}
  td {{ padding:12px 16px; border-bottom:1px solid var(--border); font-size:13px; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:var(--surface2); }}
  .bar {{ height:6px; background:var(--accent); border-radius:3px; min-width:4px; }}

  .plot-wrap {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
  .plot-wrap img {{ width:100%; display:block; }}

  .insights-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; }}
  .insight-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; }}
  .insight-pct {{ font-family:'Syne',sans-serif; font-size:42px; font-weight:800; line-height:1; margin-bottom:8px; }}
  .insight-label {{ color:var(--muted); font-size:13px; }}
  .pct-orange {{ color:var(--accent); }}
  .pct-blue   {{ color:var(--accent2); }}
  .pct-green  {{ color:var(--win); }}
  .pct-red    {{ color:var(--lose); }}

  .match-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; margin-bottom:24px; overflow:hidden; }}
  .match-header {{
    background:var(--surface2); padding:20px 24px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:20px; flex-wrap:wrap;
  }}
  .match-title {{ font-family:'Syne',sans-serif; font-weight:700; font-size:12px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); }}
  .match-clans {{ display:flex; align-items:center; gap:10px; flex:1; }}
  .clan-tag {{ padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500; }}
  .winner-tag {{ background:rgba(34,197,94,0.15); color:var(--win); border:1px solid rgba(34,197,94,0.3); }}
  .loser-tag  {{ background:rgba(239,68,68,0.1); color:var(--lose); border:1px solid rgba(239,68,68,0.2); }}
  .vs {{ color:var(--muted); font-size:12px; }}
  .prediction-badge {{ font-size:13px; color:var(--win); }}

  .advice-header {{
    padding:16px 24px; font-size:14px; color:var(--text);
    border-bottom:1px solid var(--border);
    background:rgba(249,115,22,0.05);
    border-left:3px solid var(--accent);
  }}

  .metrics {{ padding:20px 24px; display:flex; flex-direction:column; gap:16px; }}

  .metric-block {{ border-radius:10px; padding:18px 20px; }}
  .metric-block.positive {{
    background:rgba(34,197,94,0.07);
    border:1px solid rgba(34,197,94,0.2);
  }}
  .metric-block.negative {{
    background:rgba(239,68,68,0.06);
    border:1px solid rgba(239,68,68,0.2);
  }}
  .metric-title {{ font-family:'Syne',sans-serif; font-weight:700; font-size:13px; margin-bottom:10px; }}
  .metric-block.positive .metric-title {{ color:var(--win); }}
  .metric-block.negative .metric-title {{ color:var(--lose); }}
  .metric-text {{ font-size:13px; color:var(--text); margin-bottom:10px; line-height:1.7; }}
  .metric-advice {{
    font-size:13px; color:var(--text);
    background:rgba(249,115,22,0.08);
    border-left:2px solid var(--accent);
    padding:10px 14px; border-radius:0 8px 8px 0;
    line-height:1.7;
  }}

  .footer {{ border-top:1px solid var(--border); padding:30px 40px; text-align:center; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-label">Nordeus · Job Fair 2026 · Data Science Challenge</div>
  <h1>Clan Victory<br>Predictor</h1>
  <p class="hero-sub">Predicting association tournament outcomes in Top Eleven and providing actionable coaching advice to help associations improve their chances of winning.</p>
</div>

<div class="container">

  <div class="section">
    <div class="section-title">Model Performance</div>
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value">58.99%</div><div class="stat-label">Validation Accuracy</div></div>
      <div class="stat-card"><div class="stat-value">XGB</div><div class="stat-label">Best Model (XGBoost)</div></div>
      <div class="stat-card"><div class="stat-value">24</div><div class="stat-label">Features Used</div></div>
      <div class="stat-card"><div class="stat-value">24,288</div><div class="stat-label">Training Matches</div></div>
      <div class="stat-card"><div class="stat-value">23,646</div><div class="stat-label">Test Predictions</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">What Determines Who Wins</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Factor</th><th>Importance Score</th><th>Relative Impact</th></tr></thead>
        <tbody>{importance_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Feature Impact Distribution</div>
    <div class="plot-wrap"><img src="data:image/png;base64,{shap_img_b64}" alt="SHAP Summary Plot"></div>
  </div>

  <div class="section">
    <div class="section-title">Why Associations Lose — {total:,} Matches Analysed</div>
    <div class="insights-grid">
      <div class="insight-card">
        <div class="insight-pct pct-orange">{training_pct:.1f}%</div>
        <div class="insight-label">of losing associations were out-trained by their opponent — the single most common reason for defeat</div>
      </div>
      <div class="insight-card">
        <div class="insight-pct pct-blue">{quality_pct:.1f}%</div>
        <div class="insight-label">of losing associations had lower overall squad quality than their opponent</div>
      </div>
      <div class="insight-card">
        <div class="insight-pct pct-green">{activity_pct:.1f}%</div>
        <div class="insight-label">of losing associations were less active in the 7 days leading up to the match</div>
      </div>
      <div class="insight-card">
        <div class="insight-pct pct-red">{all_three:.1f}%</div>
        <div class="insight-label">of losing associations were behind on all three dimensions simultaneously — training, activity and quality</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Association Advisory Reports — 10 Example Matches</div>
    {matches_html}
  </div>

</div>

<div class="footer">Nordeus Job Fair 2026 · Data Science Challenge · Clan Victory Predictor</div>
</body>
</html>"""

output_path = BASE_DIR / 'clan_report.html'
with open(output_path, 'w') as f:
    f.write(html)
print(f"Report saved: {output_path}")
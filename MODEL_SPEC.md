# Model Specification: FPL Points Prediction

## The Problem

**Predict the `total_points` a player will score in the NEXT gameweek (GW+1).**

This is a **regression** problem. Each row in our dataset represents one player in one gameweek, and the target is the points they scored in that gameweek. The features are all computed from data BEFORE that gameweek (lagged).

---

## Data Flow

```
Raw GW CSVs (per season)          Fixtures + Teams CSVs
        |                                  |
        v                                  v
  Merge all GWs into             Extract FDR, home/away,
  one DataFrame per season       team strength
        |                                  |
        +----------------+----------------+
                         |
                         v
              Feature Engineering
              (rolling averages, lagged features)
                         |
                         v
              Final Dataset: one row = one player-gameweek
              Features = past data only | Target = total_points
                         |
              +----------+----------+
              |                     |
              v                     v
     Linear Regression         XGBoost Regressor
       (Baseline)               (Advanced)
              |                     |
              v                     v
         Predictions (MAE, RMSE)
              |
              v
     Compare to Paper 2 benchmarks
```

---

## Seasons & Split

| Purpose | Seasons | Notes |
|---------|---------|-------|
| **Training** | 2022-23, 2023-24 | These have xG/xA columns. ~38 GW x ~600 players each |
| **Testing** | 2023-24 (last 10 GWs) OR full 2023-24 | Hold-out evaluation |
| **Validation** | Cross-validation on training set | For hyperparameter tuning |

> **Why not 2021-22?** It lacks `expected_goals`, `expected_assists`, `expected_goals_conceded`, and `starts` columns. We CAN include it if we only use features available in all seasons, but it's simpler to start with 2022-23+ which have the richest feature set.

> **Alternative split (matching CLAUDE.md):** Train on 2021-22 + 2022-23, Test on 2023-24. This requires dropping xG/xA features or imputing them for 2021-22.

---

## Target Variable

```
y = total_points  (integer, typically 0-15, can be negative)
```

This is the FPL points the player earned in that specific gameweek. Predicting this directly (not decomposed) is the simpler approach, matching Paper 1 (OpenFPL).

---

## Input Features

### Tier 1: Core Features (Baseline Model - Linear Regression)

These use only basic FPL API data available in all seasons:

| Feature | How to Compute | Source Column |
|---------|---------------|--------------|
| `form_last_3` | Mean of `total_points` over previous 3 GWs | total_points |
| `form_last_5` | Mean of `total_points` over previous 5 GWs | total_points |
| `minutes_last_3` | Mean of `minutes` over previous 3 GWs | minutes |
| `was_home` | 1 if playing at home, 0 if away (for the TARGET GW) | was_home |
| `opponent_difficulty` | FDR rating of the upcoming opponent (from fixtures) | team_h_difficulty / team_a_difficulty |
| `ict_index_last_3` | Mean of `ict_index` over previous 3 GWs | ict_index |
| `position` | One-hot encoded: GK, DEF, MID, FWD | position |
| `value` | Player's price (divided by 10) | value |

**Total: ~10 features** (including one-hot encoded position)

### Tier 2: Extended Features (XGBoost Model)

All Tier 1 features PLUS:

| Feature | How to Compute | Source Column |
|---------|---------------|--------------|
| `goals_last_5` | Mean of `goals_scored` over previous 5 GWs | goals_scored |
| `assists_last_5` | Mean of `assists` over previous 5 GWs | assists |
| `clean_sheets_last_5` | Mean of `clean_sheets` over previous 5 GWs | clean_sheets |
| `bps_last_5` | Mean of `bps` over previous 5 GWs | bps |
| `influence_last_5` | Mean of `influence` over previous 5 GWs | influence |
| `creativity_last_5` | Mean of `creativity` over previous 5 GWs | creativity |
| `threat_last_5` | Mean of `threat` over previous 5 GWs | threat |
| `xG_last_5` | Mean of `expected_goals` over previous 5 GWs | expected_goals |
| `xA_last_5` | Mean of `expected_assists` over previous 5 GWs | expected_assists |
| `xGC_last_5` | Mean of `expected_goals_conceded` over previous 5 GWs | expected_goals_conceded |
| `saves_last_5` | Mean of `saves` over previous 5 GWs | saves |
| `yellow_cards_last_5` | Mean of `yellow_cards` over previous 5 GWs | yellow_cards |
| `bonus_last_5` | Mean of `bonus` over previous 5 GWs | bonus |
| `selected_pct` | Ownership percentage (log-scaled) | selected |
| `team_strength` | Team overall strength rating | teams.csv: strength |
| `opponent_strength` | Opponent overall strength rating | teams.csv: strength |
| `cumulative_points_season` | Total points earned so far this season | computed |
| `games_played_season` | GWs played (minutes > 0) so far this season | computed |

**Total: ~28 features**

---

## Data Leakage Prevention (CRITICAL)

For predicting GW N points, we can ONLY use data from GW 1 to GW N-1:

```
Predicting GW 10 points for Player X:

  ALLOWED: GW 1-9 stats for Player X (and all other players)
  ALLOWED: Fixture info for GW 10 (opponent, home/away, FDR)
  NOT ALLOWED: Any GW 10 stats (that's what we're predicting)
  NOT ALLOWED: Any future data (GW 11+)
```

Rolling averages must be computed with a **shift of 1 GW** to avoid including the target GW.

---

## Models

### Model 1: Linear Regression (Baseline)
- **Library:** scikit-learn `LinearRegression`
- **Features:** Tier 1 only (~10 features)
- **Preprocessing:** StandardScaler on numeric features
- **No hyperparameter tuning needed**
- **Purpose:** Establish a simple baseline, meet professor's requirement

### Model 2: XGBoost Regressor (Advanced)
- **Library:** xgboost `XGBRegressor`
- **Features:** Tier 1 + Tier 2 (~28 features)
- **Preprocessing:** No scaling needed (tree-based)
- **Hyperparameter tuning:** Grid search or random search with CV
- **Key hyperparameters to tune:**
  - `n_estimators`: [100, 300, 500]
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `subsample`: [0.7, 0.85, 1.0]
  - `colsample_bytree`: [0.7, 0.85, 1.0]
- **Purpose:** Show improvement over baseline, compare XGBoost vs Linear

---

## Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MAE** (Primary) | mean(\|y - y_hat\|) | Main metric for comparison, matches CLAUDE.md |
| **RMSE** (Secondary) | sqrt(mean((y - y_hat)^2)) | Penalizes large errors more, matches Paper 1 |
| **R^2** (Informational) | 1 - SS_res/SS_tot | Context only; expect 0.01-0.10 (normal for football) |

### Evaluation Breakdown
- **Overall:** MAE/RMSE across all players and positions
- **By position:** Separate MAE for GK, DEF, MID, FWD
- **By return category** (matching Paper 1):
  - Zeros: players who got 0 points (didn't play)
  - Blanks: players who got 1-2 points
  - Tickers: players who got 3-4 points
  - Haulers: players who got 5+ points

---

## Benchmark Comparison

### vs Paper 2 (Uppsala Thesis - Primary Comparison)
Paper 2 achieved with **Linear Regression:**
- **1,293 cumulative FPL points** over 21 GWs (top 12% of managers)
- Their approach: decomposed predictions + LP optimization

Paper 2 achieved with **XGBoost:**
- **1,256 cumulative FPL points** over 21 GWs

We can compare:
1. **Direct MAE/RMSE** on our test set vs their reported component metrics.
2. **Simulated cumulative points** by picking a team each GW based on our predictions.

### vs Paper 1 (OpenFPL - Stretch Goal)
Paper 1 reported (1 GW ahead, overall):
- Baseline (Last 5 avg): RMSE ~0.79-5.61 depending on category
- OpenFPL: RMSE ~0.82-5.14 depending on category

We can compare our RMSE by return category against these numbers.

### Our Target (from CLAUDE.md):
- **MAE < 2.0** overall average across all positions

---

## What Success Looks Like

### Minimum (Project Pass):
1. Working pipeline: raw data -> features -> predictions
2. Linear Regression baseline with reported MAE/RMSE
3. XGBoost model with reported MAE/RMSE
4. Show XGBoost vs Linear Regression comparison
5. MAE < 2.0 overall
6. Compare results against Paper 2 benchmarks

### Good (Strong Project):
7. Position-specific models (separate model per GK/DEF/MID/FWD)
8. Feature importance analysis (which features matter most)
9. Simulated FPL season (pick teams, measure cumulative points)
10. Comparison against both papers

### Excellent (Beyond Requirements):
11. SHAP explainability on XGBoost
12. Squad optimization layer
13. Multiple rolling windows (3 and 5 matches)
14. Additional seasons / Understat data integration

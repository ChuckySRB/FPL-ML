# Paper 2: Enhancing Fantasy Premier League Strategies through Machine Learning and Large Language Models

**Authors:** Emil Notelid & Theo Ostlund
**Institution:** Uppsala University (Master's Thesis, 30 credits)
**Published:** May 2025
**Supervisor:** David Sumpter (co-founder of Twelve Football)

---

## 1. Paper Overview

This is a Master's thesis that builds a **full-stack AI assistant for FPL** combining:
1. ML prediction models (Linear Regression + XGBoost)
2. Squad optimization (Linear Programming)
3. LLM-powered explainable recommendations (ChatGPT)
4. A web-based UI for interacting with predictions

The key differentiator from other papers: it doesn't just predict - it **explains why** each prediction was made using SHAP values translated into natural language by an LLM.

### Key Findings
- Linear regression models achieved **1,293 points** over GW 1-21 of 2024/25 (top 12% of all FPL managers).
- XGBoost achieved **1,256 points** over the same period.
- Linear regression beat XGBoost in cumulative FPL points despite XGBoost having lower MAE.
- Users with the AI assistant scored significantly higher on interpretability, trust, and adoption metrics.

---

## 2. Prediction Approach (DIFFERENT from Paper 1)

### Decomposed Point Prediction
Instead of predicting `total_points` directly, this paper **decomposes FPL points into individual scoring components** and predicts each separately:

| Model Name | Position | What It Predicts |
|-----------|----------|-----------------|
| `xStarted` | DEF, MID, FWD | Binary: will the player start? (Logistic Regression) |
| `xGoals` | DEF, MID, FWD | Expected number of goals scored |
| `xAssists` | DEF, MID, FWD | Expected number of assists |
| `xBps` | DEF, MID, FWD | Expected Bonus Points System score |
| `xConcededGoals` | GK, DEF, MID | Expected goals conceded (for clean sheet calc) |
| `xSaves` | GK | Expected number of saves |
| `xYellow` | GK, DEF, MID, FWD | Probability of yellow card |

Then total predicted points = sum of individual component predictions converted to FPL points.

### Why This Matters
- Each component has its own predictive model with its own features.
- This allows **position-specific feature selection** (e.g., shot creation metrics for xGoals of forwards, tackle/interception stats for xConcededGoals of defenders).
- It enables **explainability**: you can tell a user "Player X is predicted 6 points because we expect 0.4 goals, 0.2 assists, high clean sheet probability, and 2 BPS."

---

## 3. Data Sources

### 3.1 FPL API Data (Public)
Standard gameweek-level player data from the vaastav/Fantasy-Premier-League GitHub repo:
- `merged_gw.csv` per season
- Variables: total_points, minutes, goals_scored, assists, clean_sheets, bonus, bps, influence, creativity, threat, ict_index, xP, xG, xA, saves, yellow_cards, red_cards, was_home, opponent_team, etc.

### 3.2 Match-Event Data (Licensed/External)
Two additional datasets providing much richer per-90-minute metrics:

**Player-Match Dataset (60+ per-90 metrics):**
- Attacking: xG per 90, xGOT per 90, Shots per 90, Key passes per 90, Touches in box per 90, Box entries per 90
- Passing: Creative passes per 90, Playmaking passes per 90, Passes (xT) per 90
- Dribbling: Dribbles (xT) per 90, Dribbles success %, Ball progression (xT) per 90
- Defending: Defensive actions per 90, Interceptions per 90, True tackles won per 90
- Aerials: Aerials won per 90, Headed plays per 90
- Physical: Under pressure retention per 90, Ball recoveries per 90

**Team-Match Dataset:**
- np Shots, np Goals, np xG, xG, Shot conversion
- xT (Expected Threat), Penalty area touches, Box entries
- PPDA, Opp. PPDA, Defensive intensity
- Win/Loss/Draw probability %, Ball possession %
- Territorial dominance, Ball-in-play minutes

### 3.3 Seasons Used
- **Training:** 2020/21, 2021/22, 2022/23, 2023/24 (4 seasons)
- **Testing:** 2024/25 (GW 1-21)

---

## 4. Feature Engineering

### 4.1 Rolling Averages
- **Last 5 matches** rolling average for most variables (not multiple horizons like Paper 1).
- Example: `goals_last5`, `minutes_last5`, `saves_last5`, `assists_last5`, `clean_sheets_last5`

### 4.2 Season-Level Aggregates
- `goals_per90_season`, `assists_per90_season`, `threat_per90_season`, `creativity_per90_season`
- Cumulative stats: `cumulative_goals`, `cumulative_assists`, `cumulative_threat`, `cumulative_creativity`

### 4.3 Team Context Features
- `team_strength` - derived team quality metric
- `opponent_strength` - opponent quality
- `relative_team_strength` - ratio/comparison
- `was_home` - home/away indicator

### 4.4 Engineered Composite Features (The "Secret Sauce")

**Player-based engineered metrics:**
| Feature | Description |
|---------|-----------|
| `progressive_threat_efficiency` | Per-90 xT from ball progression, runs, and dribbles, normalized by carries |
| `finishing_efficiency` | Gap between actual goals per-90 and xG per-90 |
| `chance_quality` | Quality of scoring opportunities involving the player |
| `assist_efficiency` | Efficiency in converting passes to assists |
| `creative_passing_threat` | Threat generated by creative passes |
| `ball_retention_under_pressure` | Ability to keep the ball under pressure |
| `aerial_dominance` | Effectiveness in aerial duels |
| `defensive_duel_effectiveness` | Effectiveness in defensive duels |

**Team-based engineered metrics:**
| Feature | Description |
|---------|-----------|
| `attacking_efficiency` | Product of box-entry-to-shot rate and shot-conversion rate |
| `pressing_intensity_diff` | Difference between opponent PPDA and own PPDA |
| `attack_directness` | How direct the team's attacking play is |
| `defensive_compactness` | Team compactness in defense |
| `high_press_effectiveness` | Effectiveness when pressing high |
| `game_control` | Overall match control metric |
| `game_control_diff` | Game control vs opponent |

### 4.5 Feature Selection Process
1. **Correlation filtering:** Remove features with correlation > threshold (position-specific) to prevent multicollinearity.
2. **Forward stepwise AIC:** Iteratively add features only if they lower Akaike Information Criterion.
3. Result: Each model per position has a **different, curated feature subset**.

---

## 5. Models & Training

### 5.1 Logistic Regression (xStarted)
- Binary classifier: will the player start?
- Separate model per position (DEF, MID, FWD).
- Results: ~87% accuracy for DEF, ~85% for MID, ~91% for FWD.
- This is used as a **gate** - if predicted not to start, other predictions become irrelevant.

### 5.2 Linear Regression (Main Models)
- Separate model per scoring component per position.
- Missing-value imputation + feature normalization pipeline.
- Feature selection via correlation + forward AIC.
- **These produced the best cumulative FPL simulation results (1,293 points).**

### 5.3 XGBoost (Comparison Model)
- Same feature candidate pool as linear regression.
- Hyperparameter tuning via **grid search with cross-validation**.
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`.
- SHAP values extracted for explainability.
- **Lower MAE but higher variance; produced 1,256 cumulative points.**

### 5.4 Key XGBoost Results (xGoals model)

| Model | Position | MAE | RMSE | R^2 |
|-------|----------|-----|------|-----|
| Linear Reg | FWD | 0.3739 | 0.4954 | 0.0691 |
| Linear Reg | MID | 0.1805 | 0.3374 | 0.0745 |
| Linear Reg | DEF | 0.0645 | 0.1830 | 0.0018 |
| XGBoost | FWD | 0.2502 | 0.5002 | 0.0511 |
| XGBoost | MID | 0.1182 | 0.3438 | 0.0609 |
| XGBoost | DEF | 0.0341 | 0.1847 | 0.0023 |

**Key insight:** R^2 values are very low (0.001-0.07) for all models. This is normal for football prediction - the sport is inherently noisy and unpredictable.

---

## 6. Squad Optimization

### Linear Programming approach:
- **Objective:** Maximize sum of predicted points for starting 11.
- **Constraints:** Budget (100M), max 3 players per club, position limits (2 GK, 5 DEF, 5 MID, 3 FWD), formation rules.
- **Transfers:** Greedy one-gameweek-ahead strategy (no multi-week planning).
- Automatic substitutions when a starter doesn't play.
- Solved using CBC (open-source LP solver).

---

## 7. LLM Integration & Explainability

### For Linear Models:
- Multiply model coefficients by feature values per observation.
- Rank features by impact on prediction.
- Pass rankings to ChatGPT to generate natural language explanations.

### For XGBoost:
- Extract SHAP values per prediction.
- Group positive and negative SHAP features.
- Pass to LLM for natural language generation.

### User Study Results (Scale 1-7):
| Theme | AI Assistant Group | Raw Data Group |
|-------|-------------------|---------------|
| Interpretability | 6.11 | 2.80 |
| Trust | 5.67 | 3.20 |
| Efficiency | 6.17 | 4.00 |
| Usability | 6.50 | 2.40 |
| Adoption | 6.50 | 3.40 |

---

## 8. Delimitations (What They Didn't Model)
- Penalty saves, penalty misses (too rare/noisy)
- Red cards, own goals (too rare)
- Goals scored by goalkeepers (too rare)
- FPL chip strategies (Wildcard, Free Hit, Bench Boost, Triple Captain)
- Multi-gameweek transfer planning

---

## 9. Data Preparation Guide (What We Need to Replicate)

### Step 1: Get FPL API Data
- Download `merged_gw.csv` from vaastav/Fantasy-Premier-League for each season.
- Key fields: total_points, minutes, goals_scored, assists, clean_sheets, bonus, bps, influence, creativity, threat, ict_index, expected_goals, expected_assists, was_home, opponent_team, value.

### Step 2: Create Rolling Features (Last 5 Matches)
- `goals_last5`, `assists_last5`, `minutes_last5`, `saves_last5`, `clean_sheets_last5`
- `points_last5`, `ict_last5`, `threat_last5`, `creativity_last5`
- All must be **lagged** (computed from past data only).

### Step 3: Create Season Aggregates
- `goals_per90_season`, `cumulative_goals`, `cumulative_threat`, etc.

### Step 4: Add Team Context
- `team_strength`, `opponent_strength`, `relative_team_strength`
- `was_home`
- Can derive team strength from goals scored/conceded ratios.

### Step 5: Build Separate Models Per Component
- For each of: goals, assists, BPS, conceded goals, saves, yellow cards.
- For each position: DEF, MID, FWD (and GK where applicable).

### Step 6: Feature Selection
- Correlation matrix to remove multicollinear features.
- Forward AIC selection for linear models.

### Step 7: Combine Predictions Into Total Points
- Convert each component prediction to FPL point values using the scoring rules.
- Sum them up.

---

## 10. Relevance to Our Project

### What We Can Directly Use
- **Same data source:** vaastav/Fantasy-Premier-League GitHub repo.
- **Same models:** Linear Regression (our baseline) + XGBoost (our advanced model).
- **Same seasons:** Training on 2021-22, 2022-23 (plus additional seasons); Testing on 2023-24.
- **Rolling averages over last 5 matches** - similar to our `form_last_3` but with a 5-match window.

### Key Insights for Our Project
1. **Decomposing total_points into components may not be necessary** for our minimum requirements - direct prediction is simpler and used by Paper 1.
2. **Linear regression can beat XGBoost in cumulative simulation** even if XGBoost has lower MAE - this is important context for our baseline vs. advanced comparison.
3. **R^2 will be low** (~0.01-0.07) and that's expected for football data. Don't be discouraged.
4. **xStarted model** is valuable - knowing if a player will play is the #1 predictor of points.
5. The **match-event per-90 data** they used is richer than what's in the public FPL dataset, so our results will likely be more modest without it.

### Target MAE Benchmark
- Their xGoals MAE: 0.06-0.37 depending on position (but this is just one component).
- Overall simulated performance: top 12% of FPL managers.
- Our target MAE < 2.0 for total_points is achievable with the public FPL data alone.

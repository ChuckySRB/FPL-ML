# Paper 1: OpenFPL - An Open-Source Forecasting Method Rivaling State-of-the-Art FPL Services

**Author:** Daniel Groos (Groos Analytics)
**Published:** July 2025 (arXiv:2508.09992v1)
**Code:** https://github.com/daniegr/OpenFPL

---

## 1. Paper Overview

OpenFPL is an open-source FPL forecasting method that uses **only publicly available data** (FPL API + Understat API) to predict player FPL points. It achieves accuracy comparable to the leading commercial service (FPL Review Massive Data Model) and **surpasses it for high-return players** (>2 points).

### Key Contribution
- First fully open-source, public-data-only method that rivals paid commercial FPL forecasting services.
- Position-specific ensemble models (one per FPL position: GK, DEF, MID, FWD, AM).
- Prospective evaluation on out-of-sample 2024-25 data at 1, 2, and 3 gameweek horizons.

---

## 2. Prediction Target

- **Target variable:** Normalized FPL points in the upcoming match (normalized to [0, 1] range).
- **Prediction type:** Regression (direct point prediction).
- **Forecast horizons:** 1, 2, and 3 gameweeks ahead.
- **Output:** Median forecasted FPL points from ensemble of 50 individual models per position.

---

## 3. Data Sources

| Source | What It Provides |
|--------|-----------------|
| **FPL API** | Player-level: position, availability, minutes played, goals, assists, saves, FPL points, ICT (Influence/Creativity/Threat), BPS (Bonus Points System). Team-level: fixture opponent, home/away status. |
| **Understat API** | Player-level: xG, xA, shots, xGChain, xGBuildup, key passes. Team-level: Deep, Deep allowed, PPDA (Passes Per Defensive Action), xGA. |
| **FPL Historical Dataset** | GitHub repo by Vaastav Anand: https://github.com/vaastav/Fantasy-Premier-League |

### Seasons Used
- **Development (Training):** 2020-21, 2021-22, 2022-23, 2023-24 (4 seasons)
- **Evaluation (Test):** 2024-25 (GW 32-38, prospective)

### Cross-Validation Strategy
- 5-fold CV based on **team splits** (not random player splits).
- Each fold contains ~16 team-seasons (teams appearing in all 4 seasons grouped together).
- Teams split between upper and lower table halves per fold.

---

## 4. Feature Engineering (Critical Details)

### 4.1 Feature Categories

Features are organized into 4 groups with **multiple time horizons**:

| Feature Group | Symbol | Description |
|--------------|--------|-------------|
| **Player history** | X_p | Player-specific stats averaged over different windows |
| **Team history** | X_t | Team-level stats averaged over different windows |
| **Opponent history** | X_o | Opponent team stats averaged over different windows |
| **Match status** | X_s | Current availability, league rank, opponent rank |

### 4.2 Rolling Window Horizons

Each historical feature is computed at **6 different time horizons**:
- **Last 1 match**
- **Last 3 matches**
- **Last 5 matches**
- **Last 10 matches**
- **Last 38 matches** (full season)

This means each raw stat becomes 5 features (one per window), massively expanding the feature set.

### 4.3 Position-Specific Feature Sets

Different feature sets for different positions:

| Position | # Features | Key Differences |
|----------|-----------|-----------------|
| **GK** | 196 | Includes saves, penalties saved; excludes goals scored, assists |
| **DEF/MID/FWD** | 206 | Full offensive and defensive stats |
| **AM (Assistant Manager)** | 122 | Simplified: no detailed player-level offensive stats |

### 4.4 Player-Level Features (X_p) - from FPL API
- FPL points, Relevant FPL points (home/away venue specific)
- Minutes played
- Influence, Creativity, Threat (ICT index components)
- Goals scored, Penalties missed, Assists
- Goals conceded, Own goals, Saves, Penalties saved
- Yellow cards, Red cards
- BPS (Bonus Points System), FPL bonus points

### 4.5 Player-Level Features (X_p) - from Understat API
- Shots, xG, xGChain, xGBuildup, Key passes, xA

### 4.6 Team-Level Features (X_t) - from FPL API
- Goals scored, Goals conceded
- League rank, Opponent league rank

### 4.7 Team-Level Features (X_t) - from Understat API
- xG, Deep allowed, PPDA allowed (attacking & defending)
- xGA, Deep, PPDA (attacking & defending)

### 4.8 Opponent Features (X_o)
- Same as team features but for the upcoming opponent

### 4.9 Match Status Features (X_s)
- Player availability (0%, 25%, 50%, 75%, 100%)
- Team league rank (current)
- Opponent league rank (current)

---

## 5. Models & Training

### 5.1 Model Architecture
- **Ensemble of XGBoost + Random Forest** per position.
- K-Best Search (K=10) for automatic hyperparameter tuning.
- Ensemble = top K models from K-Best Search across all 5 CV folds = **50 models per position**.
- Final prediction = **median** of 50 model predictions.

### 5.2 Hyperparameter Search Space

**Random Forest:**
| Parameter | Values |
|-----------|--------|
| n_estimators | 200, 400, 800 |
| max_depth | 10, 20, None |
| min_samples_split | 2, 5 |
| min_samples_leaf | 1, 2, 5 |
| max_features | "sqrt", 0.2, 1.0 |
| bootstrap | True, False |

**XGBoost:**
| Parameter | Values |
|-----------|--------|
| n_estimators | 300, 600, 1200 |
| max_depth | 3, 5, 7 |
| learning_rate | 0.01, 0.05, 0.1 |
| subsample | 0.5, 0.75, 1.0 |
| colsample_bytree | 0.5, 0.75, 1.0 |
| min_child_weight | 1.0, 5.0 |
| gamma | 0.0, 0.1 |
| reg_lambda | 1.0, 5.0 |

### 5.3 Preprocessing
- Feature normalization: **MinMaxScaler** (to [0, 1])
- Target normalization: MinMaxScaler
- Sample weighting: **KBinsDiscretizer** (ordinal encoding) with position-specific bin counts (2, 3, 4, 3, 5 for GK, DEF, MID, FWD, AM) + `compute_sample_weight` with `class_weight="balanced"`, clipped at 95th percentile, rescaled to unit mean.
- XGBoost: `early_stopping_rounds=30`, `eval_metric="rmse"`
- All models: `random_state=42`, `n_jobs=-1`

---

## 6. Evaluation Metrics & Results

### 6.1 Metrics
- **Primary:** RMSE
- **Secondary:** MAE

### 6.2 Return Categories (for stratified evaluation)
| Category | Definition |
|----------|-----------|
| **Zeros** | Players who did not play (0 FPL points) |
| **Blanks** | Played but earned max 2 FPL points |
| **Tickers** | Returned 3 or 4 FPL points |
| **Haulers** | Achieved 5+ FPL points |

### 6.3 Key Results (1 GW ahead, RMSE with MAE in parentheses)

| Category | Last 5 Baseline | FPL Review | OpenFPL |
|----------|----------------|------------|---------|
| **Zeros** | 0.791 (0.270) | **0.689** (0.237) | 0.818 (0.427) |
| **Blanks** | 1.400 (0.652) | **1.189** (0.597) | 1.291 (0.749) |
| **Tickers** | 2.136 (1.645) | 1.594 (1.227) | **1.517** (1.127) |
| **Haulers** | 5.613 (4.709) | 5.172 (4.381) | **5.142** (4.317) |

**Key insight:** OpenFPL is best for high-return players (Tickers & Haulers), while FPL Review is better for low-return predictions (Zeros & Blanks).

### 6.4 Results by Position (1 GW, RMSE)

| Position | Last 5 | FPL Review | OpenFPL |
|----------|--------|------------|---------|
| GK | 0.672 | 0.512 | 0.616 |
| DEF | 0.753 | 0.740 | 0.812 |
| MID | 0.831 | 0.686 | 0.902 |
| FWD | 0.877 | 0.709 | 0.719 |
| AM | N/A | N/A | 6.192 |

---

## 7. Data Preparation Guide (What We Need to Replicate)

### Step 1: Download Historical Data
- Get `merged_gw.csv` for each season from the vaastav/Fantasy-Premier-League GitHub repo.
- Get Understat data (xG, xA, shots, etc.) per player per match.

### Step 2: Create Player-Gameweek Rows
- Each row = one player in one gameweek.
- Include the target: `total_points` for the NEXT gameweek.

### Step 3: Compute Rolling Averages
For each feature, compute mean over:
- Last 1, 3, 5, 10, and 38 matches (all lagged - no future data leakage).

### Step 4: Add Team & Opponent Context
- Team goals scored/conceded (rolling), league rank.
- Understat team stats: xG, xGA, Deep, PPDA.
- Same for opponent.

### Step 5: Add Match Status
- Player availability percentage.
- Current league position of team and opponent.

### Step 6: Position-Specific Feature Sets
- Build separate feature matrices for GK, DEF/MID/FWD, and AM.

### Step 7: Normalize & Weight
- MinMaxScaler on features and target.
- Sample weighting using discretized target distribution.

---

## 8. Relevance to Our Project

### What We Can Directly Use
- **Same data source:** vaastav/Fantasy-Premier-League GitHub repo.
- **Same models:** XGBoost (we already plan this) + Random Forest.
- **Rolling window approach** aligns with our `form_last_3` and `ict_index_rolling` plans.
- **Position-specific models** - we should consider this.

### What We Should Adapt
- Our CLAUDE.md specifies simpler features (form_last_3, minutes_last_3, opponent_difficulty, was_home, ict_index_rolling) - this paper shows the value of many more features at multiple horizons.
- We can use their rolling window horizons (1, 3, 5) as inspiration but don't need all 6.
- We use MAE as primary metric (they use RMSE as primary, MAE as secondary).

### Target MAE Benchmark
- OpenFPL overall MAE at 1 GW: ~0.427-1.127 depending on category.
- Our target: MAE < 2.0 overall - this should be achievable.

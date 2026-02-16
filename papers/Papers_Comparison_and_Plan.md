# Comparison of Papers & Action Plan for Our Project

---

## 1. Side-by-Side Comparison

| Aspect | Paper 1: OpenFPL | Paper 2: Enhancing FPL |
|--------|-----------------|----------------------|
| **Type** | Research paper (arXiv) | Master's thesis (Uppsala) |
| **Published** | July 2025 | May 2025 |
| **Primary Goal** | Beat commercial FPL forecasting services | Build explainable AI assistant for FPL |
| **Prediction Target** | Total FPL points (direct) | Decomposed: goals, assists, BPS, etc. |
| **Models Used** | XGBoost + Random Forest ensemble | Linear Regression + XGBoost |
| **Data Sources** | FPL API + Understat API (public only) | FPL API + Licensed match-event data |
| **Training Seasons** | 2020-21 to 2023-24 | 2020-21 to 2023-24 |
| **Test Season** | 2024-25 (GW 32-38) | 2024-25 (GW 1-21) |
| **Rolling Windows** | 1, 3, 5, 10, 38 matches | 5 matches (single window) |
| **Position-Specific** | Yes (5 groups: GK, DEF, MID, FWD, AM) | Yes (per component per position) |
| **Feature Count** | 122-206 per position | Large (60+ per-90 metrics + engineered) |
| **Hyperparameter Tuning** | K-Best Search (custom) | Grid Search with CV |
| **Evaluation Metric** | RMSE (primary), MAE (secondary) | MAE, RMSE, R^2 |
| **Best Result** | Rivals FPL Review (commercial) | 1,293 points (top 12% of managers) |
| **Open Source** | Yes (GitHub with trained models) | No |
| **Explainability** | No | Yes (SHAP + LLM) |
| **Squad Optimization** | No (forecasting only) | Yes (LP-based) |

---

## 2. Key Differences in Approach

### Prediction Strategy
- **Paper 1** predicts `total_points` **directly** using a single ensemble model per position.
- **Paper 2** **decomposes** total_points into components (goals, assists, BPS, clean sheets, saves, yellow cards) and predicts each separately, then sums.

**For our project:** Direct prediction is simpler and more aligned with our CLAUDE.md spec. We should start with direct `total_points` prediction.

### Feature Complexity
- **Paper 1** uses 6 different time horizons per feature (1, 3, 5, 10, 38 matches) creating 196-206 features.
- **Paper 2** uses 1 time horizon (last 5 matches) but much richer per-90 metrics from licensed data.

**For our project:** We should use 2-3 time horizons (last 3 and last 5 or 10) from the public FPL data. Our CLAUDE.md already specifies `form_last_3` and `minutes_last_3` which is a good start.

### Model Choice
- **Paper 1** uses **ensemble** of XGBoost + Random Forest (50 models per position).
- **Paper 2** uses individual Linear Regression and XGBoost models, finding that **Linear Regression beats XGBoost in cumulative simulation** despite having higher MAE.

**For our project:** Our CLAUDE.md specifies Linear Regression (baseline) + XGBoost (advanced), which matches Paper 2's approach perfectly.

---

## 3. What Both Papers Agree On

1. **Position-specific models are important.** Both papers train separate models per position.
2. **Rolling averages are essential.** Both use historical rolling stats (not just latest GW data).
3. **Public FPL data from vaastav GitHub repo is sufficient** for building competitive models.
4. **XGBoost doesn't always beat simpler models** in practice (Paper 1 uses it in an ensemble, Paper 2 shows linear regression wins in simulation).
5. **Football is inherently unpredictable.** Low R^2 values (~0.01-0.07) are normal and expected.
6. **Minutes played / starting probability is the strongest predictor.** Both papers emphasize this.
7. **Understat/advanced stats (xG, xA) add value** but the FPL API alone provides a solid foundation.

---

## 4. Benchmark Metrics for Our Project

### From Paper 1 (OpenFPL) - Overall RMSE at 1 GW ahead:
| Method | Zeros | Blanks | Tickers | Haulers |
|--------|-------|--------|---------|---------|
| Last 5 baseline | 0.791 | 1.400 | 2.136 | 5.613 |
| OpenFPL | 0.818 | 1.291 | 1.517 | 5.142 |

### From Paper 2 - Component-level MAE (xGoals):
| Position | Linear Reg MAE | XGBoost MAE |
|----------|---------------|-------------|
| FWD | 0.3739 | 0.2502 |
| MID | 0.1805 | 0.1182 |
| DEF | 0.0645 | 0.0341 |

### From Paper 2 - Simulation Results:
| Model | Cumulative Points (GW 1-21) | FPL Percentile |
|-------|---------------------------|----------------|
| Linear Regression | 1,293 | Top 12% |
| XGBoost | 1,256 | Top ~14% |
| Baseline (cumulative points heuristic) | 1,132 | Lower |

### Our Target (from CLAUDE.md):
- **MAE < 2.0** overall across all positions.
- This is achievable - even a simple "Last 5 average" baseline gets reasonable results.

---

## 5. Action Plan for Our Project

### Phase 1: Data Preparation (Priority)
1. **Download data** from vaastav/Fantasy-Premier-League for seasons 2021-22, 2022-23 (training) and 2023-24 (testing).
2. **Create player-gameweek rows** from `merged_gw.csv` or individual GW files.
3. **Engineer lagged features** (all computed from PAST data only):
   - `form_last_3`: Average total_points over last 3 GWs
   - `form_last_5`: Average total_points over last 5 GWs (add this based on papers)
   - `minutes_last_3`: Average minutes over last 3 GWs
   - `minutes_last_5`: Average minutes over last 5 GWs
   - `ict_index_rolling`: Rolling average of ICT index (last 3 or 5)
   - `goals_last_5`, `assists_last_5`, `clean_sheets_last_5`
   - `was_home`: Boolean for next match
   - `opponent_difficulty`: FDR for upcoming match (from fixtures)
4. **Add team context:**
   - Team strength (derived from goals scored/conceded)
   - Opponent strength
   - League position

### Phase 2: Baseline Model (Linear Regression)
1. Train a **single Linear Regression** model on all players.
2. Use simple features only (as per CLAUDE.md): form_last_3, minutes_last_3, opponent_difficulty, was_home, ict_index_rolling.
3. Evaluate with MAE and RMSE.
4. Then try **position-specific Linear Regression** (one per position: GK, DEF, MID, FWD).

### Phase 3: Advanced Model (XGBoost)
1. Expand feature set: add multiple rolling windows (3 and 5), more stats.
2. Train XGBoost with hyperparameter tuning (grid search or random search).
3. Try position-specific XGBoost models.
4. Compare with baseline.

### Phase 4: Evaluation & Comparison
1. Report MAE and RMSE on 2023-24 test set.
2. Compare our results with the benchmarks from both papers.
3. Potentially run a simulation (pick teams based on predictions) to measure practical value.

### Phase 5 (Optional): Enhancements
- Add Understat data (xG, xA) if time permits.
- Add SHAP explanations (inspired by Paper 2).
- Build a simple squad optimizer.

---

## 6. Critical Lessons from Both Papers

### DO:
- Use **lagged features only** (no data leakage - this is emphasized in both papers and our CLAUDE.md).
- Train **position-specific models** for better results.
- Use **rolling averages** over multiple windows.
- Include **team context** (strength, opponent difficulty, home/away).
- Use **sample weighting** to balance high-scoring vs. low-scoring predictions (Paper 1).
- Expect **low R^2** (~0.01-0.07) - this is normal for football data.

### DON'T:
- Don't expect XGBoost to automatically beat Linear Regression in practical simulation.
- Don't use only the latest gameweek stats - rolling averages are much more stable.
- Don't ignore non-playing players - predicting zeros is important (Paper 1 shows this).
- Don't try to predict rare events like own goals, penalty saves directly.
- Don't over-engineer with too many features before getting a working baseline.

---

## 7. Quick Reference: Feature Priority List

Based on both papers, here are the most impactful features ranked by importance:

### Tier 1 (Must Have):
1. **Minutes played** (rolling average) - strongest signal for whether a player will score points
2. **Recent form** (points in last 3-5 GWs) - captures current performance level
3. **Was home** (home/away indicator) - consistent advantage for home players
4. **Opponent strength / FDR** - fixture difficulty matters

### Tier 2 (Should Have):
5. **ICT Index components** (Influence, Creativity, Threat) - FPL's own advanced metrics
6. **Player availability** percentage - injury/suspension risk
7. **Team strength** relative to opponent
8. **BPS (Bonus Points System)** rolling average
9. **Goals/assists rolling** averages

### Tier 3 (Nice to Have):
10. **xG, xA** (from Understat or FPL API's expected_goals/expected_assists)
11. **Clean sheets** rolling average (for DEF/GK)
12. **Saves** rolling average (for GK)
13. **League rank** of team and opponent
14. **Transfers in/out** (ownership momentum)

### Tier 4 (Advanced - If Time Permits):
15. **PPDA** and pressing metrics (from Understat)
16. **Multiple rolling windows** (1, 3, 5, 10 matches)
17. **xGChain, xGBuildup** (Understat)
18. **Engineered interaction features** (finishing efficiency, pressing intensity, etc.)

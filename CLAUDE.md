# Project Description

This project is a part of Master Studies on University of Belgrade, Mashine Learning Course. The goal of the project
is to use Mashine Learning techniques on some public dataset and have some Research Paper as a reference on the scores
to compere the results with.

## The Dataset chosen

I choose to train a model on the Premier League Fantasy data and try to potentialy use the AI built in the FPL as tool.
I downloaded 2 papers to compare the results of the training with. We can choose one to compare with.

## Your Goal

Prepare all the necesery project struture, API calls, data pulling and preparation, evaluation pipelines and so on,
so that I can then try my mashine learining techinques on and get results.
Minimum we want for this project is to train the MLs on one comparable metric from paper and have it for the needs of
project to get points.
After getting does minimal requirements we could further train the model and add more features so it usable as a companion
tool for the fpl.


# Project Specification: FPL Points Prediction

## The Objective
Predict the `total_points` a player will score in the **NEXT** Gameweek (GW+1).
This is a regression problem.

## The Data Source
- **Repository:** https://github.com/vaastav/Fantasy-Premier-League
- **Seasons to use:**
  - Training: 2021-22, 2022-23
  - Testing: 2023-24
  - 2024-25 last season
  - 2025-26 current season for final applicaiton
  

## Methodology & Models (Professors Requirements)
1.  **Baseline Model:** Linear Regression.
    - Simple features only (e.g., average points, cost, fixture difficulty).
2.  **Advanced Model:** XGBoost Regressor.
    - Complex features (rolling averages, interaction terms).
3.  **Metrics:** MAE (Primary), RMSE (Secondary).

## Critical Feature Engineering (The "Secret Sauce")
The agent must generate a dataset where each row is a Player-Gameweek combination.
**IMPORTANT:** To avoid Data Leakage, all features must be "Lagged".
- If we are predicting points for GW 10, we can ONLY use data from GW 1 to GW 9.
- **Features to engineer:**
  - `form_last_3`: Average points in last 3 games.
  - `minutes_last_3`: Average minutes played (crucial for rotation risk).
  - `opponent_difficulty`: FDR (Fixture Difficulty Rating) for the *upcoming* match.
  - `was_home`: Boolean (1 if playing home next).
  - `ict_index_rolling`: Rolling average of influence, creativity, threat.

## Success Criteria (Benchmark)
Compare the MAE
Target MAE: < 2.0 (overall average across all positions).


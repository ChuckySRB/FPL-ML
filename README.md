# Fantasy Premier League ML Assistant

This repository is a leakage-safe, partial reproduction of the OpenFPL player-performance study. It predicts FPL points per player-fixture, evaluates the models on a held-out season, and provides a local dashboard and legal-squad optimizer.

## Final result

Training uses 2020/21-2023/24; 2024/25 is never used for fitting. The primary prospective test is GW32-38 (5,533 eligible player-fixtures). Unweighted XGBoost is best with **MAE 0.944, RMSE 1.842, and R² 0.315**. Random Forest (RMSE 1.846) and Linear Regression (1.856) are close. Large hauls remain the main failure mode.

This is a partial paper reproduction: public FPL inputs are used, but the full Understat and availability feature set from OpenFPL is unavailable.

## Repository layout

- `src/preprocessing/` — season-safe loading and lagged feature engineering.
- `src/evaluation/` — metrics, return categories, and stable-club validation.
- `src/optimization.py` — 15-player FPL squad optimization.
- `scripts/` — canonical dataset, training, figure, and deliverable commands.
- `notebooks/01...04` — EDA through final error analysis; `05_app.ipynb` is the older exploratory UI notebook.
- `outputs/results/` and `outputs/figures/` — generated evaluation artifacts.
- `reports/` and `presentation/` — final Markdown, PDF, and PowerPoint deliverables.
- `app.py` — local Streamlit analysis tool.

## Reproduce the project

Create an environment and install dependencies:

```powershell
python -m venv .fpl
.fpl\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the canonical workflow from the repository root:

```powershell
python scripts/build_fixed_dataset.py
python scripts/run_academic_pipeline.py
python scripts/create_eda_figures.py
python scripts/create_deliverables.py
python -m unittest discover -s tests -v
```

Generated CSV/model artifacts are intentionally ignored by Git. The data source is [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League); rerun the scripts after raw data changes.

## Use the dashboard

```powershell
streamlit run app.py
```

The five tabs cover model results, player projections, largest errors, methodology, and squad optimization. Choose one or more gameweeks and a prediction model in the sidebar. The optimizer maximizes projected points subject to a £100m default budget, 2 GK/5 DEF/5 MID/3 FWD, and no more than three players per club.

The data-source selector supports both the historical 2024/25 test and CSVs created by `predict_next_5_gameweeks.py`. Current exports are enriched with local player prices and club metadata before optimization. The included current export is from 2025/26; collect the new season and generate a fresh CSV before making 2026/27 decisions.

## Final deliverables

- `reports/final_report.pdf` — complete Serbian-language project report.
- `presentation/final_presentation.pptx` — six-slide, five-minute presentation with speaker notes.
- `notebooks/04_evaluation_and_error_analysis.ipynb` — executed final evaluation notebook.
- `progess-datum.md` — detailed audit trail, decisions, limitations, and checklist.

## Methodological safeguards

All rolling features use prior information only (`shift(1)`), reset at season boundaries, and share the same pre-gameweek history for double-gameweek fixtures. Fixture context remains fixture-specific. Imputation is learned on training data only. The legacy `GKP` label is normalized to `GK`; test-only `AM` rows are reported but excluded from comparisons that lack training support.

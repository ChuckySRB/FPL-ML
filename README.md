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

The default **Weekly Assistant** workspace selects a season and GW, loads an
existing package, and only refreshes/recalculates when **Run predictions** is
pressed. Its five tabs cover the current GW, the direct next-five-GW average,
player comparison, risk flags, and the current-squad/AI handoff. The last tab
downloads the report, completed strategy prompt, system role, CSV, and JSON.

Use **Model Research** in the sidebar to reopen the historical academic
dashboard. Its optimizer remains available for leakage-safe test projections;
the Weekly Assistant also includes an explicitly labelled wildcard squad
optimizer for either forecast horizon.

## Final deliverables

- `reports/final_report.pdf` — complete Serbian-language project report.
- `presentation/final_presentation.pptx` — six-slide, five-minute presentation with speaker notes.
- `notebooks/04_evaluation_and_error_analysis.ipynb` — executed final evaluation notebook.
- `progess-datum.md` — detailed audit trail, decisions, limitations, and checklist.

## Methodological safeguards

All rolling features use prior information only (`shift(1)`), reset at season boundaries, and share the same pre-gameweek history for double-gameweek fixtures. Fixture context remains fixture-specific. Imputation is learned on training data only. The legacy `GKP` label is normalized to `GK`; test-only `AM` rows are reported but excluded from comparisons that lack training support.
## Production weekly assistant

The new assistant core trains two independent models: one fixture and a direct average over the next five gameweeks. It also retrofits pre-2025/26 targets with an explicitly estimated defensive-contribution bonus, while avoiding double-counting official 2025/26 points.

```powershell
python scripts/train_fpl_assistant.py
python scripts/generate_gw_report.py --season 2025-26 --gw 34
```

The weekly command writes a Markdown report, structured JSON, player/fixture CSVs, and a ready-to-fill chat prompt under `outputs/assistant/<season>/gw<n>/`. See `FPL_ASSISTANT.md` for target definitions, validation results, limitations, and interface behavior.

For a new season, use the same command with `--gw 1`. GW1 is routed through
dedicated cross-season preseason models, GW2–GW5 progressively blend preseason
and current evidence, and GW6+ uses the mature in-season models. Official
`ep_next` is shown as a sanity reference, not treated as ground truth.

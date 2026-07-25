# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives under `src/`: `data_collection/` retrieves FPL API and historical data, `preprocessing/` loads schemas and engineers leakage-safe features, `models/` contains neural-network implementations and training helpers, and `evaluation/` provides cross-validation and tracking. Project-wide paths, seasons, features, and metrics are defined in `configs/config.py`.

Use `scripts/` for runnable collection and verification utilities. Exploration and model experiments belong in `notebooks/`. Keep downloaded data in `data/raw/` or `data/processed/`, trained artifacts in `models/`, and generated figures, logs, predictions, or metrics under `outputs/`. Reference papers and accompanying analyses live in `papers/`.

## Setup, Test, and Development Commands

Create a local environment and install dependencies:

```powershell
python -m venv .fpl
.\.fpl\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run `python scripts/collect_data.py` for the interactive data-collection workflow. Use `python scripts/test_data_collection.py --test loading` for an offline-oriented loader check, or `--test historical`, `--test current`, and `--test all` for API-backed checks. Launch experiments with `jupyter notebook`; run the prediction workflow with `python predict_next_5_gameweeks.py`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, descriptive docstrings, and imports grouped as standard library, third-party, then local modules. Use `snake_case` for files, functions, variables, and feature columns; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for configuration constants. Preserve the project’s player-gameweek tabular convention. All predictors for gameweek `GW+1` must be derived only from data available through `GW`.

No formatter or linter is configured, so keep changes consistent with nearby code and avoid notebook-only logic that should be reusable from `src/`.

## Testing Guidelines

Current tests are executable scripts rather than a formal test framework. Add focused checks near `scripts/test_data_collection.py`, name test functions `test_<behavior>`, and cover success and missing-data paths. State when a test needs network access or writes datasets. For model changes, report MAE (primary) and RMSE (secondary), using chronological or season-based splits to prevent leakage.

## Commit & Pull Request Guidelines

History favors short, imperative or title-style summaries such as `Latest GW Predictions`; keep each commit focused and describe the affected behavior. Pull requests should explain the data/model change, commands run, seasons and split used, and metric impact. Link relevant issues and include plots or screenshots when predictions or notebook visuals change. Do not commit generated CSV/JSON/Parquet data, model binaries, virtual environments, or cache files; the existing `.gitignore` excludes them.

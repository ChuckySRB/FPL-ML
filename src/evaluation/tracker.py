"""
Experiment tracker for FPL model runs.

Usage:
    tracker = ExperimentTracker()
    tracker.log(
        name='XGBoost v1',
        y_true=y_test,
        y_pred=xgb_preds,
        positions=test_positions,
        config={
            'model': 'XGBoost',
            'features': 'Tier 2',
            'n_features': 29,
            'train_seasons': '2022-23',
            'test_season': '2023-24',
            'params': {'n_estimators': 300, 'max_depth': 5, ...}
        }
    )
    tracker.summary()          # print table of all runs
    df = tracker.load_runs()   # load as DataFrame
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import RESULTS_DIR


RUNS_FILE = RESULTS_DIR / 'experiment_runs.jsonl'

CATEGORIES = ['Zeros', 'Blanks', 'Tickers', 'Haulers']
POSITIONS  = ['GK', 'DEF', 'MID', 'FWD']


def _get_category(pts: float) -> str:
    if pts == 0:    return 'Zeros'
    elif pts <= 2:  return 'Blanks'
    elif pts <= 4:  return 'Tickers'
    else:           return 'Haulers'


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     positions: np.ndarray) -> dict:
    """Compute full metrics: overall, by category, by position."""
    metrics = {
        'overall_rmse': _rmse(y_true, y_pred),
        'overall_mae':  _mae(y_true, y_pred),
        'n_test':       int(len(y_true)),
    }

    cats = np.array([_get_category(p) for p in y_true])
    for cat in CATEGORIES:
        mask = cats == cat
        if mask.sum() > 0:
            metrics[f'{cat}_rmse'] = _rmse(y_true[mask], y_pred[mask])
            metrics[f'{cat}_mae']  = _mae(y_true[mask], y_pred[mask])
            metrics[f'{cat}_n']    = int(mask.sum())
        else:
            metrics[f'{cat}_rmse'] = None
            metrics[f'{cat}_mae']  = None
            metrics[f'{cat}_n']    = 0

    for pos in POSITIONS:
        mask = positions == pos
        if mask.sum() > 0:
            metrics[f'{pos}_rmse'] = _rmse(y_true[mask], y_pred[mask])
            metrics[f'{pos}_mae']  = _mae(y_true[mask], y_pred[mask])
            metrics[f'{pos}_n']    = int(mask.sum())
        else:
            metrics[f'{pos}_rmse'] = None
            metrics[f'{pos}_mae']  = None
            metrics[f'{pos}_n']    = 0

    return metrics


class ExperimentTracker:
    """Log model runs to a file and compare them."""

    def __init__(self, runs_file: Path = RUNS_FILE):
        self.runs_file = Path(runs_file)
        self.runs_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log(self,
            name: str,
            y_true,
            y_pred,
            positions,
            config: dict = None) -> dict:
        """Log a model run.

        Args:
            name:       Human-readable run name, e.g. 'XGBoost v1'
            y_true:     Actual points (array-like)
            y_pred:     Predicted points (array-like)
            positions:  Position labels aligned with y_true (array-like of GK/DEF/MID/FWD)
            config:     Dict with any info you want to save:
                          model, features, n_features, params,
                          train_seasons, test_season, notes, ...

        Returns:
            The run record as a dict.
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        positions = np.array(positions, dtype=str)

        run = {
            'run_id':    _next_run_id(self.runs_file),
            'name':      name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Merge in user config
        if config:
            # Flatten params dict into the record for easy tabular display
            params = config.pop('params', {})
            run.update(config)
            if params:
                run['params'] = params

        # Compute and merge metrics
        run.update(_compute_metrics(y_true, y_pred, positions))

        # Append to file
        with open(self.runs_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(run) + '\n')

        print(f"  Logged run #{run['run_id']:03d} '{name}'  "
              f"RMSE={run['overall_rmse']:.4f}  MAE={run['overall_mae']:.4f}")

        return run

    def load_runs(self) -> pd.DataFrame:
        """Load all runs as a DataFrame."""
        if not self.runs_file.exists():
            return pd.DataFrame()

        records = []
        with open(self.runs_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return pd.DataFrame()

        df = pd.json_normalize(records)   # flattens nested dicts like params.*
        return df

    def clear(self):
        """Delete all saved runs (start fresh)."""
        if self.runs_file.exists():
            self.runs_file.unlink()
        print("All runs cleared.")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self, metric: str = 'rmse'):
        """Print a comparison table of all runs."""
        df = self.load_runs()
        if df.empty:
            print("No runs logged yet.")
            return

        m = metric.lower()
        cols_cat = [f'{c}_{m}' for c in CATEGORIES if f'{c}_{m}' in df.columns]
        cols_pos = [f'{p}_{m}' for p in POSITIONS  if f'{p}_{m}' in df.columns]

        display_cols = (['run_id', 'name', 'timestamp',
                         f'overall_{m}'] + cols_cat + cols_pos)
        display_cols = [c for c in display_cols if c in df.columns]

        print(f"\n{'='*100}")
        print(f"  EXPERIMENT RUNS — {metric.upper()}")
        print(f"{'='*100}")
        print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    def comparison_table(self, metric: str = 'rmse') -> pd.DataFrame:
        """Return a clean pivoted comparison table (runs × categories/positions)."""
        df = self.load_runs()
        if df.empty:
            return df

        m = metric.lower()
        cols = (['run_id', 'name'] +
                [f'{c}_{m}' for c in CATEGORIES if f'{c}_{m}' in df.columns] +
                [f'{p}_{m}' for p in POSITIONS  if f'{p}_{m}' in df.columns] +
                [f'overall_{m}'])
        cols = [c for c in cols if c in df.columns]
        return df[cols].set_index(['run_id', 'name'])

    def best_run(self, metric: str = 'overall_mae') -> pd.Series:
        """Return the run with the best (lowest) value of a metric."""
        df = self.load_runs()
        if df.empty:
            return None
        idx = df[metric].idxmin()
        return df.loc[idx]


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------

def _next_run_id(runs_file: Path) -> int:
    """Return next sequential run id."""
    if not runs_file.exists():
        return 1
    count = sum(1 for line in open(runs_file, encoding='utf-8') if line.strip())
    return count + 1


def load_runs(runs_file: Path = RUNS_FILE) -> pd.DataFrame:
    """Module-level shortcut: load all runs as a DataFrame."""
    return ExperimentTracker(runs_file).load_runs()

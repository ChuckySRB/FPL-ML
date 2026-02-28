"""
Team-stratified cross-validation for FPL prediction.

Implements the cross-validation strategy from Paper 1 (OpenFPL, Groos 2025):

  "5-fold CV based on team splits (not random player splits).
   Each fold contains ~16 team-seasons. Teams split between upper
   and lower table halves per fold."

Why team-based splits?
  - Player rows from the same team are highly correlated (same fixture,
    same opponent difficulty, shared team form).
  - A random row split would leak this correlation into the validation set,
    giving overly optimistic CV scores.
  - Splitting by team ensures the validation set contains teams unseen
    during training for that fold, giving a more realistic estimate.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────
# SPLITTER
# ─────────────────────────────────────────────────────────────

class TeamStratifiedKFold:
    """
    K-fold cross-validation split by team membership.

    Teams are ranked by average total points (proxy for league strength)
    and divided into upper/lower halves. Teams from each half are
    distributed round-robin across folds, so every fold has a balanced
    mix of strong and weak teams.

    All gameweeks for a given team land in the same fold — they are
    never split across train and validation within the same fold.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5, matching the paper).
    random_state : int
        Controls the shuffle within upper/lower halves before fold assignment.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        team_col: str = 'team',
        points_col: str = 'total_points',
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, val_indices) for each fold.

        Parameters
        ----------
        df : DataFrame
            The full dataset, must contain team_col and points_col.
        team_col : str
            Column identifying a player's team.
        points_col : str
            Column used to rank teams by average quality.

        Yields
        ------
        train_idx, val_idx : np.ndarray of integer positions
        """
        rng = np.random.RandomState(self.random_state)

        # Rank teams by mean points — best team first
        team_avg = (
            df.groupby(team_col)[points_col]
            .mean()
            .sort_values(ascending=False)
        )
        teams_ranked = team_avg.index.values
        n_teams = len(teams_ranked)

        # Upper half = stronger teams, lower half = weaker teams
        upper = teams_ranked[: n_teams // 2].copy()
        lower = teams_ranked[n_teams // 2 :].copy()
        rng.shuffle(upper)
        rng.shuffle(lower)

        # Round-robin assignment within each half
        fold_of = {}
        for i, t in enumerate(upper):
            fold_of[t] = i % self.n_splits
        for i, t in enumerate(lower):
            fold_of[t] = i % self.n_splits

        team_fold = df[team_col].map(fold_of).values
        all_idx = np.arange(len(df))

        for fold_id in range(self.n_splits):
            val_mask = team_fold == fold_id
            train_idx = all_idx[~val_mask]
            val_idx   = all_idx[val_mask]
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits

    def fold_team_table(
        self,
        df: pd.DataFrame,
        team_col: str = 'team',
        team_name_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a summary DataFrame showing which teams are in each fold.

        Useful for sanity-checking that each fold has a balanced mix of
        strong and weak teams.

        Parameters
        ----------
        team_name_col : str, optional
            If provided, map team IDs to names using this column from df.
        """
        rng = np.random.RandomState(self.random_state)

        team_avg = (
            df.groupby(team_col)['total_points']
            .mean()
            .sort_values(ascending=False)
        )
        teams_ranked = team_avg.index.values
        n_teams = len(teams_ranked)

        upper = teams_ranked[: n_teams // 2].copy()
        lower = teams_ranked[n_teams // 2 :].copy()
        rng.shuffle(upper)
        rng.shuffle(lower)

        rows = []
        for i, t in enumerate(upper):
            rows.append({'team': t, 'half': 'upper', 'fold': i % self.n_splits,
                         'avg_pts': float(team_avg[t])})
        for i, t in enumerate(lower):
            rows.append({'team': t, 'half': 'lower', 'fold': i % self.n_splits,
                         'avg_pts': float(team_avg[t])})

        result = pd.DataFrame(rows).sort_values(['fold', 'half']).reset_index(drop=True)

        if team_name_col and team_name_col in df.columns:
            name_map = df.drop_duplicates(team_col).set_index(team_col)[team_name_col]
            result['team_name'] = result['team'].map(name_map)

        return result


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION RUNNER
# ─────────────────────────────────────────────────────────────

_CATEGORIES = {
    'Zeros':   lambda y: y == 0,
    'Blanks':  lambda y: (y >= 1) & (y <= 2),
    'Tickers': lambda y: (y >= 3) & (y <= 4),
    'Haulers': lambda y: y >= 5,
}

_POSITIONS = ['GK', 'DEF', 'MID', 'FWD']


def cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    full_df: pd.DataFrame,
    cv: TeamStratifiedKFold,
    positions_col: str = 'position_label',
    team_col: str = 'team',
    sample_weight_fn=None,
    verbose: bool = True,
) -> Dict:
    """
    Run team-stratified cross-validation and return out-of-fold predictions.

    The model is fitted on the training fold and used to predict the
    validation fold. This is repeated for all folds. The concatenated
    OOF predictions cover every row in the dataset exactly once.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must implement fit(X, y) and predict(X). If sample_weight_fn is
        provided, fit is called as fit(X, y, sample_weight=...).
    X : DataFrame
        Feature matrix, aligned with full_df (same row order).
    y : Series
        Target (total_points), aligned with full_df.
    full_df : DataFrame
        Full DataFrame with at minimum 'team' and 'total_points' columns.
        Used for the team-based splitting logic.
    cv : TeamStratifiedKFold
        The splitter instance.
    positions_col : str
        Column in full_df with player position labels (GK/DEF/MID/FWD).
    team_col : str
        Column in full_df with team IDs.
    sample_weight_fn : callable, optional
        If provided, called as sample_weight_fn(y_train_fold) and the
        result is passed to model.fit as sample_weight=.
        Example: lambda y: hauler_weights(y, power=1.0)
    verbose : bool
        Print per-fold metrics.

    Returns
    -------
    dict with:
        'oof_preds'    : np.ndarray — OOF predictions (len == len(X))
        'fold_metrics' : list of dicts — per-fold MAE, RMSE, n_val
        'overall_mae'  : float
        'overall_rmse' : float
        'fold_df'      : pd.DataFrame — fold_metrics as a table
        'category_mae' : dict — MAE per return category (Zeros/Blanks/Tickers/Haulers)
        'position_mae' : dict — MAE per position
    """
    X_np = X.values if hasattr(X, 'values') else np.asarray(X)
    y_np = y.values if hasattr(y, 'values') else np.asarray(y)

    oof_preds = np.full(len(X_np), np.nan)
    fold_metrics: List[Dict] = []

    for fold_id, (train_idx, val_idx) in enumerate(
        cv.split(full_df, team_col=team_col)
    ):
        X_tr, X_val = X_np[train_idx], X_np[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]

        # Optional sample weights (e.g. hauler weighting)
        fit_kwargs = {}
        if sample_weight_fn is not None:
            fit_kwargs['sample_weight'] = sample_weight_fn(y_tr)

        model.fit(X_tr, y_tr, **fit_kwargs)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        mae  = mean_absolute_error(y_val, preds)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

        fold_metrics.append({
            'fold': fold_id + 1,
            'n_val': int(len(val_idx)),
            'mae':  round(mae, 4),
            'rmse': round(rmse, 4),
        })

        if verbose:
            print(f"  Fold {fold_id + 1}/{cv.n_splits}  "
                  f"n_val={len(val_idx):,}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ── OOF aggregate ────────────────────────────────────────────────────
    valid = ~np.isnan(oof_preds)
    overall_mae  = float(mean_absolute_error(y_np[valid], oof_preds[valid]))
    overall_rmse = float(np.sqrt(mean_squared_error(y_np[valid], oof_preds[valid])))

    if verbose:
        print(f"\n  OOF overall  MAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}")

    # ── Category breakdown ───────────────────────────────────────────────
    category_mae = {}
    for cat, mask_fn in _CATEGORIES.items():
        mask = mask_fn(y_np) & valid
        if mask.sum() > 0:
            category_mae[cat] = float(mean_absolute_error(y_np[mask], oof_preds[mask]))

    # ── Position breakdown ───────────────────────────────────────────────
    position_mae = {}
    if positions_col in full_df.columns:
        pos_arr = full_df[positions_col].values
        for pos in _POSITIONS:
            mask = (pos_arr == pos) & valid
            if mask.sum() > 0:
                position_mae[pos] = float(
                    mean_absolute_error(y_np[mask], oof_preds[mask])
                )

    if verbose:
        print("\n  MAE by return category:")
        for cat, mae in category_mae.items():
            print(f"    {cat:<10} {mae:.4f}")
        print("\n  MAE by position:")
        for pos, mae in position_mae.items():
            print(f"    {pos:<6} {mae:.4f}")

    return {
        'oof_preds':    oof_preds,
        'fold_metrics': fold_metrics,
        'overall_mae':  overall_mae,
        'overall_rmse': overall_rmse,
        'fold_df':      pd.DataFrame(fold_metrics),
        'category_mae': category_mae,
        'position_mae': position_mae,
    }

'''Train, evaluate, and persist the completed academic FPL experiment.'''

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.baselines import Tier0FormModel  # noqa: E402
from src.evaluation import (  # noqa: E402
    TeamStratifiedKFold,
    balanced_return_weights,
    cross_validate,
)
from src.preprocessing import TIER0_FEATURES, TIER1_FEATURES  # noqa: E402

DATA_DIR = ROOT / 'data' / 'processed' / 'tier2_fixed_2020-21_to_2024_25'
POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
PROSPECTIVE_GWS = list(range(32, 39))
RANDOM_STATE = 42

PAPER_CATEGORY_RMSE = {
    'Last-5 (paper)': {
        'Zeros': 0.791, 'Blanks': 1.400,
        'Tickers': 2.136, 'Haulers': 5.613,
    },
    'OpenFPL': {
        'Zeros': 0.818, 'Blanks': 1.291,
        'Tickers': 1.517, 'Haulers': 5.142,
    },
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def category_masks(y: np.ndarray, minutes: np.ndarray) -> dict:
    zeros = (minutes == 0) & (y == 0)
    return {
        'Zeros': zeros,
        'Blanks': (~zeros) & (y <= 2),
        'Tickers': (y >= 3) & (y <= 4),
        'Haulers': y >= 5,
    }


def metric_record(model: str, scope: str, group_type: str,
                  group: str, y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        'model': model,
        'scope': scope,
        'group_type': group_type,
        'group': group,
        'n': int(len(y)),
        'mae': float(mean_absolute_error(y, pred)),
        'rmse': rmse(y, pred),
        'r2': float(r2_score(y, pred)) if len(y) > 1 else np.nan,
    }


def evaluate_predictions(predictions: dict, y: np.ndarray,
                         metadata: pd.DataFrame) -> pd.DataFrame:
    records = []
    eligible = metadata['position_label'].isin(POSITIONS).to_numpy()
    scopes = {
        'full_test': eligible,
        'prospective_gw32_38': (
            eligible & metadata['round'].isin(PROSPECTIVE_GWS).to_numpy()),
    }
    for scope, scope_mask in scopes.items():
        y_scope = y[scope_mask]
        minutes = metadata.loc[scope_mask, 'minutes'].to_numpy()
        positions = metadata.loc[scope_mask, 'position_label'].to_numpy()
        rounds = metadata.loc[scope_mask, 'round'].to_numpy()
        for model_name, prediction in predictions.items():
            pred_scope = prediction[scope_mask]
            records.append(metric_record(
                model_name, scope, 'overall', 'All', y_scope, pred_scope))
            for category, local_mask in category_masks(
                    y_scope, minutes).items():
                if local_mask.any():
                    records.append(metric_record(
                        model_name, scope, 'category', category,
                        y_scope[local_mask], pred_scope[local_mask]))
            for position in POSITIONS:
                local_mask = positions == position
                if local_mask.any():
                    records.append(metric_record(
                        model_name, scope, 'position', position,
                        y_scope[local_mask], pred_scope[local_mask]))
            for gameweek in sorted(np.unique(rounds)):
                local_mask = rounds == gameweek
                records.append(metric_record(
                    model_name, scope, 'gameweek', f'GW{int(gameweek)}',
                    y_scope[local_mask], pred_scope[local_mask]))
    return pd.DataFrame(records)


def train_position_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                            train_meta: pd.DataFrame,
                            X_test: pd.DataFrame,
                            test_meta: pd.DataFrame,
                            fallback: np.ndarray):
    '''Train a compact RF/XGB ensemble for each supported position.'''
    predictions = fallback.copy()
    fitted = {}
    for position in POSITIONS:
        train_mask = train_meta['position_label'].eq(position).to_numpy()
        test_mask = test_meta['position_label'].eq(position).to_numpy()
        X_pos = X_train.loc[train_mask]
        y_pos = y_train.loc[train_mask]
        weights = balanced_return_weights(y_pos.to_numpy(), power=0.75)
        models = []
        position_predictions = []

        for index, seed in enumerate([42, 52, 62]):
            rf = RandomForestRegressor(
                n_estimators=220,
                max_depth=[10, 14, None][index],
                min_samples_leaf=[2, 3, 2][index],
                max_features=['sqrt', 0.7, 1.0][index],
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_pos, y_pos, sample_weight=weights)
            models.append(rf)
            position_predictions.append(rf.predict(X_test.loc[test_mask]))

        for index, seed in enumerate([72, 82, 92]):
            xgb = XGBRegressor(
                n_estimators=260,
                max_depth=[3, 4, 5][index],
                learning_rate=[0.04, 0.035, 0.03][index],
                subsample=0.85,
                colsample_bytree=[0.75, 0.85, 0.95][index],
                min_child_weight=[1, 3, 5][index],
                reg_lambda=2.0,
                random_state=seed,
                n_jobs=4,
                objective='reg:squarederror',
            )
            xgb.fit(X_pos, y_pos, sample_weight=weights)
            models.append(xgb)
            position_predictions.append(xgb.predict(X_test.loc[test_mask]))

        predictions[test_mask] = np.median(
            np.vstack(position_predictions), axis=0)
        fitted[position] = models
        print(f'  {position}: {len(X_pos):,} rows, {len(models)} models')
    return predictions, fitted


def save_figures(metrics: pd.DataFrame, prediction_frame: pd.DataFrame,
                 best_model: str, feature_importance: pd.Series) -> None:
    sns.set_theme(style='whitegrid')
    prospective = metrics.query(
        "scope == 'prospective_gw32_38' and group_type == 'overall'")
    ordered = prospective.sort_values('rmse')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=ordered, x='rmse', y='model', ax=axes[0],
                color='#4c78a8')
    sns.barplot(data=ordered.sort_values('mae'), x='mae', y='model',
                ax=axes[1], color='#f58518')
    axes[0].set_title('Prospective RMSE (GW32-38)')
    axes[1].set_title('Prospective MAE (GW32-38)')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'model_comparison_fixed.png', dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv')['total_points']
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv')['total_points']
    train_meta = pd.read_csv(DATA_DIR / 'train_full.csv')
    test_meta = pd.read_csv(DATA_DIR / 'test_full.csv')
    print(f'Data: train={X_train.shape}, test={X_test.shape}')

    tier0_cols = [column for column in TIER0_FEATURES
                  if column in X_train.columns]
    tier1_cols = [column for column in TIER1_FEATURES
                  if column in X_train.columns]
    predictions = {
        'Zero baseline': np.zeros(len(X_test)),
        'Mean baseline': np.full(len(X_test), y_train.mean()),
        'Last-5 baseline': X_test['form_last_5'].to_numpy(),
    }
    tier0 = Tier0FormModel().fit(X_train[tier0_cols], y_train)
    predictions['Tier-0 Form+FDR'] = tier0.predict(X_test[tier0_cols])

    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(X_train[tier1_cols], y_train)
    predictions['Linear Regression'] = linear.predict(X_test[tier1_cols])

    print('Training pooled Random Forest...')
    weights = balanced_return_weights(y_train.to_numpy(), power=0.75)
    random_forest = RandomForestRegressor(
        n_estimators=350, max_depth=14, min_samples_leaf=3,
        max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    predictions['Random Forest'] = random_forest.predict(X_test)

    print('Training pooled XGBoost...')
    xgboost = XGBRegressor(
        n_estimators=450, max_depth=5, learning_rate=0.035,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        reg_alpha=0.05, reg_lambda=2.0, random_state=RANDOM_STATE,
        n_jobs=6, objective='reg:squarederror')
    xgboost.fit(X_train, y_train)
    predictions['XGBoost'] = xgboost.predict(X_test)

    print('Training sample-weighted XGBoost...')
    weighted_xgboost = XGBRegressor(
        n_estimators=450, max_depth=5, learning_rate=0.035,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        reg_alpha=0.05, reg_lambda=2.0, random_state=RANDOM_STATE,
        n_jobs=6, objective='reg:squarederror')
    weighted_xgboost.fit(X_train, y_train, sample_weight=weights)
    predictions['XGBoost weighted'] = weighted_xgboost.predict(X_test)

    print('Training position-specific RF/XGB ensembles...')
    ensemble_prediction, position_models = train_position_ensemble(
        X_train, y_train, train_meta, X_test, test_meta,
        predictions['XGBoost'])
    predictions['Position RF+XGB ensemble'] = ensemble_prediction

    print('Evaluating held-out and prospective scopes...')
    metrics = evaluate_predictions(
        predictions, y_test.to_numpy(), test_meta)
    metrics.to_csv(RESULTS_DIR / 'model_metrics_fixed.csv', index=False)
    prospective_overall = metrics.query(
        "scope == 'prospective_gw32_38' and group_type == 'overall'")
    best_model = prospective_overall.sort_values('rmse').iloc[0]['model']
    print(prospective_overall.sort_values('rmse')[
        ['model', 'mae', 'rmse', 'r2']].to_string(index=False))
    print(f'Best prospective RMSE: {best_model}')

    prediction_frame = test_meta[[
        'season', 'round', 'fixture', 'element', 'name', 'position_label',
        'team_name', 'opponent_team', 'was_home', 'minutes', 'price',
        'opponent_difficulty',
    ]].copy()
    prediction_frame['actual_points'] = y_test.to_numpy()
    prediction_frame['evaluation_eligible'] = prediction_frame[
        'position_label'].isin(POSITIONS)
    for model_name, values in predictions.items():
        safe_name = model_name.lower().replace(' ', '_').replace('+', 'plus')
        prediction_frame[f'pred_{safe_name}'] = values
    prediction_frame['best_model'] = best_model
    prediction_frame['best_prediction'] = predictions[best_model]
    prediction_frame['error'] = (
        prediction_frame['best_prediction']
        - prediction_frame['actual_points'])
    prediction_frame['absolute_error'] = prediction_frame['error'].abs()
    prediction_frame.to_csv(
        RESULTS_DIR / 'predictions_2024_25_fixed.csv', index=False)
    prediction_frame[
        prediction_frame['round'].isin(PROSPECTIVE_GWS)
        & prediction_frame['evaluation_eligible']
    ].sort_values('absolute_error', ascending=False).head(100).to_csv(
        RESULTS_DIR / 'largest_errors_fixed.csv', index=False)

    feature_importance = pd.Series(
        xgboost.feature_importances_, index=X_train.columns,
        name='importance').sort_values(ascending=False)
    feature_importance.to_csv(
        RESULTS_DIR / 'feature_importance_fixed.csv')
    save_figures(metrics, prediction_frame, best_model, feature_importance)

    category = metrics.query(
        "scope == 'prospective_gw32_38' and group_type == 'category' "
        "and model == @best_model")[['group', 'rmse']]
    comparison_rows = [
        {'method': best_model, 'category': row.group, 'rmse': row.rmse}
        for row in category.itertuples()
    ]
    for method, values in PAPER_CATEGORY_RMSE.items():
        comparison_rows.extend({
            'method': method, 'category': category_name, 'rmse': value,
        } for category_name, value in values.items())
    comparison = pd.DataFrame(comparison_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=comparison, x='category', y='rmse', hue='method',
                order=['Zeros', 'Blanks', 'Tickers', 'Haulers'], ax=ax)
    ax.set_title('Category RMSE: our best model vs OpenFPL paper')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'category_rmse_fixed.png', dpi=160)
    plt.close(fig)

    position = metrics.query(
        "scope == 'prospective_gw32_38' and group_type == 'position' "
        "and model == @best_model")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=position, x='group', y='rmse', ax=ax,
                order=POSITIONS, color='#54a24b')
    ax.set_title(f'RMSE by position: {best_model}')
    ax.set_xlabel('Position')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'position_rmse_fixed.png', dpi=160)
    plt.close(fig)

    by_gw = metrics.query(
        "scope == 'prospective_gw32_38' and group_type == 'gameweek' "
        "and model == @best_model").copy()
    by_gw['gw'] = by_gw['group'].str.replace('GW', '').astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=by_gw.sort_values('gw'), x='gw', y='rmse',
                 marker='o', ax=ax)
    ax.set_title(f'Prospective RMSE by gameweek: {best_model}')
    ax.set_xlabel('Gameweek')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'gw_rmse_fixed.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    feature_importance.sort_values().tail(15).plot.barh(
        ax=ax, color='#e45756')
    ax.set_title('Pooled XGBoost feature importance (top 15)')
    ax.set_xlabel('Gain importance')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'feature_importance_fixed.png', dpi=160)
    plt.close(fig)

    prospective_rows = prediction_frame[
        prediction_frame['round'].isin(PROSPECTIVE_GWS)
        & prediction_frame['evaluation_eligible']]
    fig, ax = plt.subplots(figsize=(7, 6))
    sample = prospective_rows.sample(
        min(3000, len(prospective_rows)), random_state=RANDOM_STATE)
    sns.scatterplot(data=sample, x='actual_points', y='best_prediction',
                    alpha=0.25, s=18, ax=ax)
    limit = max(sample['actual_points'].max(),
                sample['best_prediction'].max())
    ax.plot([-4, limit], [-4, limit], '--', color='black', linewidth=1)
    ax.set_title(f'Actual vs predicted: {best_model}')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'actual_vs_predicted_fixed.png', dpi=160)
    plt.close(fig)

    print('Running five-fold stable-club CV for pooled XGBoost...')
    cv_model = XGBRegressor(
        n_estimators=180, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE, n_jobs=6,
        objective='reg:squarederror')
    cv_result = cross_validate(
        cv_model, X_train, y_train, train_meta,
        TeamStratifiedKFold(n_splits=5, random_state=RANDOM_STATE),
        team_col='team_name',
        sample_weight_fn=lambda y: balanced_return_weights(y, power=0.75),
        verbose=True)
    cv_result['fold_df'].to_csv(
        RESULTS_DIR / 'cv_fold_metrics_fixed.csv', index=False)

    paper_rows = []
    best_categories = metrics.query(
        "scope == 'prospective_gw32_38' and group_type == 'category' "
        "and model == @best_model")
    for row in best_categories.itertuples():
        paper_rows.append({
            'method': best_model, 'category': row.group,
            'rmse': row.rmse, 'mae': row.mae,
        })
    for method, values in PAPER_CATEGORY_RMSE.items():
        paper_rows.extend({
            'method': method, 'category': category_name,
            'rmse': value, 'mae': np.nan,
        } for category_name, value in values.items())
    pd.DataFrame(paper_rows).to_csv(
        RESULTS_DIR / 'paper_comparison_fixed.csv', index=False)

    joblib.dump({
        'feature_names': list(X_train.columns),
        'tier1_features': tier1_cols,
        'tier0': tier0,
        'linear': linear,
        'random_forest': random_forest,
        'xgboost': xgboost,
        'weighted_xgboost': weighted_xgboost,
        'position_ensemble': position_models,
        'positions': POSITIONS,
        'best_model': best_model,
    }, MODELS_DIR / 'academic_models_fixed.joblib')

    summary = {
        'best_model': best_model,
        'prospective_gameweeks': PROSPECTIVE_GWS,
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'prospective_rows': int((
            test_meta['round'].isin(PROSPECTIVE_GWS)
            & test_meta['position_label'].isin(POSITIONS)).sum()),
        'excluded_test_positions': sorted(
            set(test_meta['position_label']) - set(POSITIONS)),
        'cv_overall_mae': cv_result['overall_mae'],
        'cv_overall_rmse': cv_result['overall_rmse'],
        'models_per_position': 6,
        'methodological_scope': (
            'Partial OpenFPL reproduction using public FPL features; '
            'Understat and AM training data unavailable.'
        ),
        'prospective_metrics': prospective_overall.set_index('model')[[
            'mae', 'rmse', 'r2', 'n']].to_dict(orient='index'),
    }
    (RESULTS_DIR / 'model_summary_fixed.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

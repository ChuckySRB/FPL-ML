# FPL Feature Engineering & Preprocessing Guide

Complete guide for creating ML-ready features and preprocessing FPL data for model training.

## Quick Start

```python
from src.preprocessing import FPLDataLoader, FPLFeatureEngineer, FPLPreprocessor

# 1. Load data
loader = FPLDataLoader()
gameweeks_df = loader.load_gameweeks('2023-24')
teams_df = loader.load_teams('2023-24')
players_df = loader.load_players('2023-24')

# 2. Engineer features
engineer = FPLFeatureEngineer()
features_df = engineer.create_all_features(gameweeks_df, teams_df, players_df)

# 3. Preprocess for training
preprocessor = FPLPreprocessor(scaler_type='standard')
data = preprocessor.prepare_for_training(features_df, target_col='total_points')

X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
```

## Feature Categories

### 1. Rolling Features
Moving averages that capture recent performance trends.

**Created for**: points, minutes, goals, assists, bonus
**Windows**: 3, 5, 10 games

```python
# Example features:
- total_points_rolling_3  # Average points last 3 games
- total_points_rolling_5  # Average points last 5 games
- goals_scored_rolling_std_3  # Volatility in goals
```

**Use case**: Identify players in good form vs. average performers

### 2. Lag Features
Previous gameweek values for temporal patterns.

**Lags**: 1, 2, 3 gameweeks back

```python
# Example features:
- total_points_lag_1  # Points in previous gameweek
- minutes_lag_2  # Minutes 2 gameweeks ago
```

**Use case**: Capture immediate past performance

### 3. Form Features
Advanced metrics for player form and consistency.

```python
# Created features:
- form_weighted  # Weighted average (recent games weighted more)
- consistency_5  # Inverse of std deviation (1 / (1 + std))
- form_trend  # Linear trend coefficient (improving/declining)
- points_per_90  # Points per 90 minutes
- games_played  # Cumulative games count
```

**Use case**: Identify consistent performers vs. volatile players

### 4. Attacking Features
Offensive performance metrics.

```python
# Created features:
- goal_involvement  # Goals + assists
- goal_involvement_rolling_3  # Recent goal involvement
- xg_rolling_3  # Expected goals (3-game avg)
- goals_vs_xg  # Over/underperformance vs expected
- xa_rolling_3  # Expected assists
- creativity_rolling_5  # FPL creativity metric
- threat_rolling_5  # FPL threat metric
```

**Use case**: Identify attacking threats, compare xG vs actual

### 5. Defensive Features
Defensive performance for DEF/GK.

```python
# Created features:
- clean_sheets_rolling_3  # Recent clean sheet rate
- goals_conceded_rolling_5  # Conceded goals average
- xgc_rolling_5  # Expected goals conceded
- saves_rolling_5  # Saves (for goalkeepers)
```

**Use case**: Predict clean sheets, evaluate defensive strength

### 6. Opponent Features
Difficulty based on opponent strength.

```python
# Created features:
- opponent_strength  # Opponent overall strength
- opponent_difficulty  # 100 - opponent strength
- opponent_strength_attack_home/away
- opponent_strength_defence_home/away
```

**Use case**: Adjust predictions based on fixture difficulty

### 7. Home/Away Features
Location-specific performance.

```python
# Created features:
- avg_points_home  # Average points at home
- avg_points_away  # Average points away
- was_home  # Boolean indicator
```

**Use case**: Account for home advantage

### 8. Value Features
Economic metrics and selection trends.

```python
# Created features:
- points_per_cost  # Points / (cost/10)
- value_score  # Points per million
- net_transfers  # Transfers in - transfers out
- transfer_momentum  # Rolling transfer trend
```

**Use case**: Identify value picks, popular players

## Feature Engineering Functions

### Create Specific Feature Types

```python
engineer = FPLFeatureEngineer()

# Individual feature types
df = engineer.create_rolling_features(df, columns=['total_points', 'minutes'])
df = engineer.create_lag_features(df, columns=['total_points'])
df = engineer.create_form_features(df)
df = engineer.create_attacking_features(df)
df = engineer.create_defensive_features(df)
df = engineer.create_opponent_features(df, teams_df)
df = engineer.create_home_away_features(df)
df = engineer.create_value_features(df, players_df)

# Or create all at once
df = engineer.create_all_features(df, teams_df, players_df)
```

### Custom Rolling Windows

```python
engineer = FPLFeatureEngineer()
engineer.rolling_windows = [2, 4, 8]  # Custom windows

df = engineer.create_rolling_features(df, columns=['total_points'])
# Creates: total_points_rolling_2, total_points_rolling_4, total_points_rolling_8
```

## Preprocessing Pipeline

### 1. Handle Missing Values

```python
preprocessor = FPLPreprocessor()

# Different strategies
df = preprocessor.handle_missing_values(df, strategy='median')  # Default
df = preprocessor.handle_missing_values(df, strategy='mean')
df = preprocessor.handle_missing_values(df, strategy='constant', fill_value=0)
```

### 2. Encode Categorical Variables

```python
# Automatic detection and encoding
df = preprocessor.encode_categorical(df)

# Specify columns
df = preprocessor.encode_categorical(df, categorical_cols=['position_label', 'team'])
```

### 3. Scale Features

```python
# Different scalers
preprocessor = FPLPreprocessor(scaler_type='standard')  # StandardScaler
preprocessor = FPLPreprocessor(scaler_type='robust')    # RobustScaler
preprocessor = FPLPreprocessor(scaler_type='minmax')    # MinMaxScaler

# Scale
X_train_scaled = preprocessor.scale_features(X_train, fit=True)
X_test_scaled = preprocessor.scale_features(X_test, fit=False)
```

### 4. Feature Selection

```python
# Automatic feature/target selection
X, y = preprocessor.select_features(
    df,
    target_col='total_points',
    exclude_cols=['element', 'round', 'fixture']
)
```

## Data Splitting Strategies

### Random Split (Not Recommended for Time Series)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Temporal Split (Recommended)

```python
# Use recent gameweeks for testing
train_df, test_df = preprocessor.create_temporal_split(
    df,
    date_col='round',
    test_rounds=5  # Last 5 gameweeks
)
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Train and evaluate model
```

## Complete Pipeline

```python
from src.preprocessing import (
    FPLDataLoader,
    FPLFeatureEngineer,
    FPLPreprocessor,
    prepare_training_data
)

# 1. Load data
loader = FPLDataLoader()
gw_df = loader.load_gameweeks('2023-24')
teams_df = loader.load_teams('2023-24')
players_df = loader.load_players('2023-24')

# 2. Feature engineering
engineer = FPLFeatureEngineer()
features_df = engineer.create_all_features(gw_df, teams_df, players_df)

# 3. Prepare training data (remove incomplete records)
ready_df = prepare_training_data(
    features_df,
    target_col='total_points',
    drop_first_n_games=5  # Need history for rolling features
)

# 4. Full preprocessing
preprocessor = FPLPreprocessor(scaler_type='standard')
data = preprocessor.prepare_for_training(
    ready_df,
    target_col='total_points',
    temporal_split=True,
    test_rounds=5,
    scale=True,
    handle_missing=True
)

# 5. Extract datasets
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# 6. Save for later use
preprocessor.save_pipeline('models/preprocessor.pkl')
```

## Position-Specific Modeling

```python
from src.preprocessing import create_position_specific_data

# Filter by position
forwards_df = create_position_specific_data(features_df, position='FWD')
defenders_df = create_position_specific_data(features_df, position='DEF')

# Train position-specific models
preprocessor_fwd = FPLPreprocessor()
data_fwd = preprocessor_fwd.prepare_for_training(forwards_df)
```

## Feature Analysis

### Get Feature Groups

```python
from src.preprocessing import get_feature_columns_by_type

feature_groups = get_feature_columns_by_type(X_train)

# Access specific groups
rolling_features = feature_groups['rolling']
lag_features = feature_groups['lag']
form_features = feature_groups['form']
```

### Feature Importance

```python
import pandas as pd

# Correlation with target
train_with_target = X_train.copy()
train_with_target['target'] = y_train

correlations = train_with_target.corr()['target'].drop('target')
top_features = correlations.abs().sort_values(ascending=False).head(20)

print("Top 20 features:")
print(top_features)
```

## Saving and Loading

### Save Processed Data

```python
from configs.config import PROCESSED_DATA_DIR

output_dir = PROCESSED_DATA_DIR / '2023-24'
output_dir.mkdir(parents=True, exist_ok=True)

# Save datasets
X_train.to_csv(output_dir / 'X_train.csv', index=False)
X_test.to_csv(output_dir / 'X_test.csv', index=False)
y_train.to_csv(output_dir / 'y_train.csv', index=False)
y_test.to_csv(output_dir / 'y_test.csv', index=False)

# Save preprocessor
preprocessor.save_pipeline(output_dir / 'preprocessor.pkl')
```

### Load Processed Data

```python
import pandas as pd

X_train = pd.read_csv('data/processed/2023-24/X_train.csv')
X_test = pd.read_csv('data/processed/2023-24/X_test.csv')
y_train = pd.read_csv('data/processed/2023-24/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/2023-24/y_test.csv').squeeze()

# Load preprocessor
preprocessor = FPLPreprocessor()
preprocessor.load_pipeline('data/processed/2023-24/preprocessor.pkl')
```

## Best Practices

### 1. Sort Before Feature Engineering
```python
# ALWAYS sort by player and gameweek first
df = df.sort_values(['element', 'round'])
```

### 2. Drop First N Games
```python
# Remove games with incomplete history
df = prepare_training_data(df, drop_first_n_games=5)
```

### 3. Use Temporal Splits
```python
# For time series, use temporal validation
data = preprocessor.prepare_for_training(df, temporal_split=True)
```

### 4. Scale After Split
```python
# Fit scaler on train, transform both train and test
X_train_scaled = preprocessor.scale_features(X_train, fit=True)
X_test_scaled = preprocessor.scale_features(X_test, fit=False)
```

### 5. Handle Missing Values Carefully
```python
# Use median for robustness
df = preprocessor.handle_missing_values(df, strategy='median')
```

## Common Use Cases

### Predicting Next Gameweek Points

```python
# Use all historical data, predict next week
df = engineer.create_all_features(gameweeks_df, teams_df, players_df)
# Select current gameweek data as "test"
latest_gw = df['round'].max()
X_latest = df[df['round'] == latest_gw]
# Use model to predict next gameweek
```

### Position-Specific Predictions

```python
# Train separate models for each position
for position in ['GK', 'DEF', 'MID', 'FWD']:
    pos_df = create_position_specific_data(features_df, position)
    data = preprocessor.prepare_for_training(pos_df)
    # Train position-specific model
```

### Multi-Season Training

```python
# Load multiple seasons
seasons_df = loader.load_multi_season_gameweeks(['2021-22', '2022-23', '2023-24'])
# Engineer features
features_df = engineer.create_all_features(seasons_df, teams_df, players_df)
# Prepare for training
data = preprocessor.prepare_for_training(features_df)
```

## Troubleshooting

### Issue: Too many missing values
**Solution**: Increase `drop_first_n_games` or use more aggressive imputation

### Issue: Features not being created
**Solution**: Check if required columns exist in dataframe

### Issue: Poor model performance
**Solution**: Try different feature groups, check for data leakage

### Issue: Temporal validation failing
**Solution**: Ensure 'round' column exists and data is sorted

## Next Steps

After feature engineering and preprocessing:
1. Train baseline models (Linear Regression, Random Forest)
2. Try advanced models (XGBoost, LightGBM, CatBoost)
3. Hyperparameter tuning with Optuna
4. Feature selection and importance analysis
5. Model evaluation and comparison with research papers
6. Deploy for predictions

See [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) for a complete walkthrough.

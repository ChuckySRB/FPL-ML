"""
Configuration file for FPL Machine Learning Project
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

# Model directories
MODELS_DIR = BASE_DIR / 'models'

# Output directories
OUTPUTS_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
LOGS_DIR = OUTPUTS_DIR / 'logs'
RESULTS_DIR = OUTPUTS_DIR / 'results'

# FPL API URLs
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FPL_ELEMENT_SUMMARY_URL = f"{FPL_BASE_URL}/element-summary/"
FPL_ENTRY_URL = f"{FPL_BASE_URL}/entry/"
FPL_FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"

# Historical data source
HISTORICAL_DATA_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Seasons to download
SEASONS = [
    '2016-17',
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
    '2025-26'
]

# Player positions
POSITIONS = {
    1: 'GK',
    2: 'DEF',
    3: 'MID',
    4: 'FWD'
}

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature engineering
ROLLING_WINDOWS = [3, 5, 10]  # Windows for rolling averages
LAG_FEATURES = [1, 2, 3]  # Number of gameweeks to lag

# Training parameters
CROSS_VALIDATION_FOLDS = 5
HYPERPARAMETER_TRIALS = 50  # For Optuna

# Metrics to track (aligned with research papers)
METRICS = [
    'mae',  # Mean Absolute Error
    'rmse',  # Root Mean Squared Error
    'r2',  # R-squared
    'mape',  # Mean Absolute Percentage Error
]

# Target variable
TARGET = 'total_points'  # What we're trying to predict

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  MODELS_DIR, FIGURES_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

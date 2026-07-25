"""
Preprocessing pipeline for FPL ML models.

Handles:
- Multi-season data loading + feature engineering
- Train/test split (by season or temporal)
- Feature selection (Tier 0 through Tier 3)
- Scaling for Linear Regression
- Saving processed datasets
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.preprocessing.data_loader import FPLDataLoader
from src.preprocessing.feature_engineering import (
    FPLFeatureEngineer, TIER0_FEATURES, TIER1_FEATURES, TIER2_FEATURES,
    TIER3_FEATURES,
    prepare_training_data
)


class FPLPreprocessor:
    """End-to-end preprocessing pipeline for FPL prediction."""

    def __init__(self):
        self.loader = FPLDataLoader()
        self.engineer = FPLFeatureEngineer()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None

    def build_dataset(self,
                      train_seasons: List[str],
                      test_season: str,
                      tier: int = 2,
                      min_gw: int = 6) -> Dict:
        """Build full train/test dataset from raw data.

        Args:
            train_seasons: Seasons to use for training (e.g., ['2022-23'])
            test_season: Season to use for testing (e.g., '2023-24')
            tier: Feature tier (0=heuristic, 1=baseline, 2=extended,
                3=advanced)
            min_gw: Minimum GW to include per season

        Returns:
            Dict with X_train, X_test, y_train, y_test, and metadata
        """
        print("=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)

        # --- Load and engineer TRAINING data ---
        print(f"\n1. Loading training data: {train_seasons}")
        train_df = self._load_and_engineer(train_seasons, tier)
        train_df = prepare_training_data(train_df, min_gw=min_gw)
        print(f"   Training records: {len(train_df):,}")

        # --- Load and engineer TEST data ---
        print(f"\n2. Loading test data: {test_season}")
        test_df = self._load_and_engineer([test_season], tier)
        test_df = prepare_training_data(test_df, min_gw=min_gw)
        print(f"   Test records: {len(test_df):,}")

        # --- Select features ---
        feature_lists = {
            0: TIER0_FEATURES,
            1: TIER1_FEATURES,
            2: TIER2_FEATURES,
            3: TIER3_FEATURES,
        }
        if tier not in feature_lists:
            raise ValueError('tier must be 0, 1, 2, or 3')
        feature_list = feature_lists[tier]
        available_features = [
            feature for feature in feature_list
            if feature in train_df.columns and feature in test_df.columns
        ]
        missing_features = [
            feature for feature in feature_list
            if feature not in available_features
        ]

        if missing_features:
            print(f"\n   Warning: Missing features: {missing_features}")

        print(f"\n3. Features selected ({len(available_features)}):")
        for f in available_features:
            print(f"   - {f}")

        self.feature_names = available_features

        # --- Extract X and y ---
        X_train = train_df[available_features].copy()
        y_train = train_df['total_points'].copy()
        X_test = test_df[available_features].copy()
        y_test = test_df['total_points'].copy()

        # --- Handle missing values ---
        print(f"\n4. Handling missing values...")
        n_missing_train = X_train.isnull().sum().sum()
        n_missing_test = X_test.isnull().sum().sum()
        print(f"   Train missing: {n_missing_train}, Test missing: {n_missing_test}")

        X_train = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=available_features, index=X_train.index
        )
        X_test = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=available_features, index=X_test.index
        )

        # --- Summary ---
        print(f"\n{'=' * 60}")
        print("DATASET READY")
        print(f"{'=' * 60}")
        print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"  Test:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
        print(f"  Target (train): mean={y_train.mean():.2f}, std={y_train.std():.2f}")
        print(f"  Target (test):  mean={y_test.mean():.2f}, std={y_test.std():.2f}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': available_features,
            'train_df': train_df,
            'test_df': test_df,
        }

    def scale_for_linear(self, X_train: pd.DataFrame,
                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features for Linear Regression (StandardScaler).

        XGBoost does NOT need scaling - only call this for Linear Regression.
        """
        cols = X_train.columns
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=cols, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=cols, index=X_test.index
        )
        return X_train_scaled, X_test_scaled

    def _load_and_engineer(self, seasons: List[str], tier: int) -> pd.DataFrame:
        """Engineer every season with its own teams and fixtures."""
        engineered = []
        for season in seasons:
            try:
                gw_df = self.loader.load_gameweeks(season)
            except FileNotFoundError:
                print(f'   Skipping {season} (not found)')
                continue
            gw_df['season'] = season
            teams_df = self.loader.load_teams(season)
            try:
                fixtures_df = self.loader.load_fixtures(season)
            except FileNotFoundError:
                fixtures_df = None

            if ('team' in gw_df.columns
                    and (gw_df['team'].fillna(0) == 0).mean() > 0.5):
                print(f'   Fixing missing team IDs for {season}...')
                try:
                    players_df = self.loader.load_players(season)
                    if 'id' in players_df.columns and 'team' in players_df.columns:
                        team_map = players_df.set_index('id')['team'].to_dict()
                        gw_df['team'] = gw_df['element'].map(team_map)
                except FileNotFoundError:
                    pass

            print(f'   Engineering {season} features (Tier {tier})...')
            engineered.append(self.engineer.create_all_features(
                gw_df,
                teams_df=teams_df,
                fixtures_df=fixtures_df,
                tier=tier,
            ))

        if not engineered:
            raise ValueError('No season data could be loaded')
        return pd.concat(engineered, ignore_index=True, sort=False)

    def save(self, data: Dict, name: str = 'default'):
        """Save processed datasets and pipeline."""
        output_dir = PROCESSED_DATA_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)

        data['X_train'].to_csv(output_dir / 'X_train.csv', index=False)
        data['X_test'].to_csv(output_dir / 'X_test.csv', index=False)
        data['y_train'].to_csv(output_dir / 'y_train.csv', index=False, header=['total_points'])
        data['y_test'].to_csv(output_dir / 'y_test.csv', index=False, header=['total_points'])

        # Save pipeline objects
        pipeline_path = output_dir / 'pipeline.pkl'
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
        }, pipeline_path)

        print(f"\nSaved to {output_dir.name}/:")
        print(f"  X_train.csv  ({data['X_train'].shape})")
        print(f"  X_test.csv   ({data['X_test'].shape})")
        print(f"  y_train.csv  ({len(data['y_train'])} records)")
        print(f"  y_test.csv   ({len(data['y_test'])} records)")
        print(f"  pipeline.pkl")


def create_position_specific_data(df: pd.DataFrame,
                                  position: str) -> pd.DataFrame:
    """Filter data for position-specific modeling."""
    if 'position_label' not in df.columns:
        raise ValueError("position_label column not found")
    return df[df['position_label'] == position].copy()


def get_feature_columns_by_type(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize features by type for analysis."""
    features = {
        'rolling': [col for col in df.columns if 'rolling' in col or 'last_' in col],
        'form': [col for col in df.columns if 'form' in col or 'consistency' in col],
        'opponent': [col for col in df.columns if 'opponent' in col],
        'team': [col for col in df.columns if 'team_strength' in col],
        'position': [col for col in df.columns if col.startswith('pos_')],
        'cumulative': [col for col in df.columns if 'cumulative' in col or 'games_played' in col],
        'value': [col for col in df.columns if col in ('price', 'selected_pct')],
    }
    return {k: [col for col in v if col in df.columns] for k, v in features.items()}

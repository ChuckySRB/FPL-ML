'''Build the canonical leakage-safe OpenFPL reproduction dataset.'''

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.config import PROCESSED_DATA_DIR  # noqa: E402
from src.preprocessing import FPLPreprocessor  # noqa: E402


TRAIN_SEASONS = ['2020-21', '2021-22', '2022-23', '2023-24']
TEST_SEASON = '2024-25'
RUN_NAME = 'tier2_fixed_2020-21_to_2024_25'


def validate_dataset(data: dict) -> dict:
    '''Run structural checks before any artifact is saved.'''
    train = data['train_df']
    test = data['test_df']

    train_seasons = sorted(train['season'].dropna().unique().tolist())
    test_seasons = sorted(test['season'].dropna().unique().tolist())
    if train_seasons != TRAIN_SEASONS:
        raise AssertionError(f'Unexpected train seasons: {train_seasons}')
    if test_seasons != [TEST_SEASON]:
        raise AssertionError(f'Unexpected test seasons: {test_seasons}')

    for label, frame in [('train', train), ('test', test)]:
        duplicate_keys = ['season', 'element', 'round', 'fixture']
        if frame.duplicated(duplicate_keys).any():
            raise AssertionError(f'Exact player-fixture duplicates in {label}')
        same_gw = frame.groupby(
            ['season', 'element', 'round'])['form_last_5'].nunique(
                dropna=False)
        if same_gw.max() != 1:
            raise AssertionError(f'Intra-DGW feature leakage in {label}')

    return {
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'train_seasons': train_seasons,
        'test_seasons': test_seasons,
        'features': data['feature_names'],
        'n_features': len(data['feature_names']),
        'train_target_mean': float(data['y_train'].mean()),
        'test_target_mean': float(data['y_test'].mean()),
        'test_gw_min': int(test['round'].min()),
        'test_gw_max': int(test['round'].max()),
    }


def main() -> None:
    preprocessor = FPLPreprocessor()
    data = preprocessor.build_dataset(
        train_seasons=TRAIN_SEASONS,
        test_season=TEST_SEASON,
        tier=2,
        min_gw=6,
    )
    manifest = validate_dataset(data)

    preprocessor.save(data, name=RUN_NAME)
    output_dir = PROCESSED_DATA_DIR / RUN_NAME
    data['train_df'].to_csv(output_dir / 'train_full.csv', index=False)
    data['test_df'].to_csv(output_dir / 'test_full.csv', index=False)

    manifest.update({
        'run_name': RUN_NAME,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'unit_of_analysis': 'player-fixture',
        'minimum_gameweek': 6,
        'target': 'total_points',
        'source': (
            'https://github.com/vaastav/Fantasy-Premier-League'
        ),
    })
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

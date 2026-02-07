# FPL Data Collection Guide

This guide explains how to collect and load FPL data for your machine learning project.

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
.fpl\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Collect Historical Data

Download historical FPL data from past seasons:

```bash
# Download all configured seasons
python src/data_collection/historical_data_downloader.py --all

# Or download a specific season
python src/data_collection/historical_data_downloader.py --season 2023-24

# Verify downloads
python src/data_collection/historical_data_downloader.py --verify
```

### 3. Collect Current Season Data

Download current season data from the FPL API:

```bash
# Collect all current season data
python src/data_collection/current_season_collector.py --season 2025-26

# Collect only bootstrap data (faster)
python src/data_collection/current_season_collector.py --bootstrap-only

# Test with limited players
python src/data_collection/current_season_collector.py --max-players 50
```

### 4. Test the Pipeline

Run the test suite to verify everything works:

```bash
# Run all tests
python scripts/test_data_collection.py

# Run specific test
python scripts/test_data_collection.py --test historical
python scripts/test_data_collection.py --test current
python scripts/test_data_collection.py --test loading
```

## Using the Data Loader

### Basic Usage

```python
from src.preprocessing.data_loader import FPLDataLoader

# Initialize loader
loader = FPLDataLoader()

# Load data for a specific season
players_df = loader.load_players('2023-24')
gameweeks_df = loader.load_gameweeks('2023-24')
teams_df = loader.load_teams('2023-24')
fixtures_df = loader.load_fixtures('2023-24')
```

### Load Multiple Seasons

```python
# Load gameweek data from multiple seasons
multi_season_df = loader.load_multi_season_gameweeks(
    seasons=['2021-22', '2022-23', '2023-24']
)

# Load all available seasons
all_seasons_df = loader.load_multi_season_gameweeks()
```

### Quick Load Function

```python
from src.preprocessing.data_loader import quick_load

# Quick load for common use cases
df = quick_load(season='2023-24', data_type='gameweeks')
```

### Check Available Data

```python
# Get list of available seasons
seasons = loader.get_available_seasons()
print(f"Available seasons: {seasons}")

# Get summary for a specific season
summary = loader.get_data_summary('2023-24')
print(summary)
```

## Data Schemas

The project uses standardized data schemas defined in `src/preprocessing/schemas.py`.

### Using Data Schemas

```python
from src.preprocessing.schemas import DataSchema

schema = DataSchema()

# Automatically standardize data
players_df = schema.prepare_player_data(raw_players_df)
gameweeks_df = schema.prepare_gameweek_data(raw_gameweeks_df)
```

### Feature Groups

```python
from src.preprocessing.schemas import get_feature_groups

# Get grouped features for modeling
feature_groups = get_feature_groups()

# Access specific groups
attacking_features = feature_groups['attacking']
defensive_features = feature_groups['defensive']
```

### Position Mapping

```python
from src.preprocessing.schemas import POSITIONS, POSITION_IDS

# Convert position ID to label
position_label = POSITIONS[1]  # 'GK'

# Convert label to ID
position_id = POSITION_IDS['GK']  # 1
```

## Data Structure

After collection, your data will be organized as follows:

```
data/raw/
├── 2021-22/
│   ├── players_raw.csv          # Player overview data
│   ├── teams.csv                # Team data
│   ├── fixtures.csv             # Fixture schedule
│   ├── gws/
│   │   ├── merged_gw.csv        # All gameweeks merged
│   │   ├── gw1.csv              # Individual gameweek files
│   │   ├── gw2.csv
│   │   └── ...
│   └── players/
│       ├── player_1/
│       │   ├── gw_history.csv   # Player's GW performance
│       │   └── season_history.csv  # Past seasons
│       └── ...
├── 2022-23/
│   └── ...
└── 2025-26/
    └── ...
```

## Common Patterns

### Load and Merge Data

```python
from src.preprocessing.data_loader import FPLDataLoader
import pandas as pd

loader = FPLDataLoader()

# Load gameweeks and players
gw_df = loader.load_gameweeks('2023-24')
players_df = loader.load_players('2023-24')

# Merge to add player info
merged_df = gw_df.merge(
    players_df[['id', 'web_name', 'element_type', 'team']],
    left_on='element',
    right_on='id',
    how='left'
)
```

### Filter by Position

```python
from src.preprocessing.schemas import POSITION_IDS

# Load data
gw_df = loader.load_gameweeks('2023-24')
players_df = loader.load_players('2023-24')

# Merge
merged_df = gw_df.merge(players_df[['id', 'element_type']],
                        left_on='element', right_on='id')

# Filter forwards only
forwards_df = merged_df[merged_df['element_type'] == POSITION_IDS['FWD']]
```

### Time-Based Filtering

```python
# Filter specific gameweeks
early_season = gw_df[gw_df['round'] <= 10]
mid_season = gw_df[(gw_df['round'] > 10) & (gw_df['round'] <= 28)]
late_season = gw_df[gw_df['round'] > 28]
```

## Configuration

Edit `configs/config.py` to customize:

- **Seasons to download**: Update `SEASONS` list
- **Data directories**: Modify `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`
- **API URLs**: Change if FPL API endpoints update
- **Feature engineering settings**: Adjust `ROLLING_WINDOWS`, `LAG_FEATURES`

## Troubleshooting

### Missing Data

If data is missing for a season:

```python
# Check what's available
summary = loader.get_data_summary('2023-24')
print(summary)
```

### Download Errors

If downloads fail:
- Check internet connection
- Verify the season exists in the historical repository
- Try downloading a single season first
- Check if the FPL API is accessible

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root
cd d:\Caslav\Master\Машинско Учење\Projekat

# Add to PYTHONPATH (Windows)
set PYTHONPATH=%PYTHONPATH%;%CD%
```

## Next Steps

After collecting data:

1. **Exploratory Data Analysis**: Create notebooks in `notebooks/` to explore the data
2. **Feature Engineering**: Build preprocessing pipelines in `src/preprocessing/`
3. **Model Training**: Implement models in `src/models/`
4. **Evaluation**: Compare results with research papers using metrics in `src/evaluation/`

## Examples

Check out `scripts/test_data_collection.py` for working examples of:
- Downloading historical data
- Collecting current season data
- Loading and validating data
- Working with multiple seasons

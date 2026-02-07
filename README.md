# Fantasy Premier League Machine Learning Assistant

This project is an implementation for the Machine Learning Masters Course at the School of Electrical Engineering in Belgrade. The goal is to apply machine learning techniques to Fantasy Premier League (FPL) data and compare results with published research papers.

## Project Objective

Build a machine learning model to predict player performance in Fantasy Premier League, with results comparable to academic research papers. The system will eventually serve as a companion tool for FPL team selection.

## Project Structure

```
.
├── configs/                    # Configuration files
│   └── config.py              # Main project configuration
├── data/
│   ├── raw/                   # Raw data from FPL API
│   ├── processed/             # Cleaned and feature-engineered data
│   └── external/              # External data sources
├── models/                    # Saved trained models
├── notebooks/                 # Jupyter notebooks for exploration
├── outputs/
│   ├── figures/              # Visualization outputs
│   ├── logs/                 # Training logs
│   └── results/              # Evaluation results
├── papers/                    # Reference research papers
├── src/
│   ├── data_collection/      # FPL API data collection
│   │   ├── getters.py       # API getter functions
│   │   ├── parsers.py       # Data parsing utilities
│   │   └── teams_scraper.py # Team data scraper
│   ├── preprocessing/        # Data preprocessing modules
│   ├── models/               # ML model implementations
│   └── evaluation/           # Model evaluation metrics
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### 1. Virtual Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .fpl

# Activate (Windows)
.fpl\Scripts\activate

# Activate (Linux/Mac)
source .fpl/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
python -c "import pandas, sklearn, xgboost; print('Installation successful!')"
```

## Data Structure

The data folder contains data from past seasons as well as the current season:

- `season/cleaned_players.csv` : Overview stats for the season
- `season/gws/gw_number.csv` : GW-specific stats for the particular season
- `season/gws/merged_gws.csv` : GW-by-GW stats for each player in a single file
- `season/players/player_name/gws.csv` : GW-by-GW stats for that specific player
- `season/players/player_name/history.csv` : Prior seasons history stats for that specific player

## Accessing Data Programmatically

You can access FPL data using the modules in `src/data_collection/`:

```python
from src.data_collection import get_data, get_individual_player_data

# Get general FPL data
data = get_data()

# Get specific player data
player_data = get_individual_player_data(player_id=123)
```

### Using Historical Data from GitHub

```python
import pandas as pd

# URL of the CSV file (example)
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(url)
```

## Player Position Mapping

In the FPL data, `element_type` corresponds to positions:
- 1 = GK (Goalkeeper)
- 2 = DEF (Defender)
- 3 = MID (Midfielder)
- 4 = FWD (Forward)

## Downloading Team Data

You can download data for a specific team:

```bash
cd src/data_collection
python teams_scraper.py <team_id> <season_code> <start_gw>

# Example:
python teams_scraper.py 2572486 25_26 1
```

This creates a folder `team_<team_id>_data<season>` with all important data.

## Configuration

Project settings are centralized in [configs/config.py](configs/config.py):
- API URLs
- Data directories
- Model parameters
- Feature engineering settings
- Evaluation metrics

## Next Steps

1. **Data Collection**: Download historical FPL data for multiple seasons
2. **Data Preprocessing**: Clean and engineer features from raw data
3. **Model Training**: Implement and train ML models (scikit-learn, XGBoost, LightGBM)
4. **Evaluation**: Compare results with research papers using standard metrics
5. **Deployment**: Build a usable FPL companion tool

## Reference Papers

The project includes two research papers for baseline comparison:
- [Enhancing Fantasy Premier League with ML.pdf](papers/Enhancing Fantasy Premier League with ML.pdf)
- [OpenFPL.pdf](papers/OpenFPL.pdf)

## Technologies Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: matplotlib, seaborn, plotly
- **Hyperparameter Tuning**: Optuna
- **Experimentation**: Jupyter notebooks

## Contributing

This is an academic project. Suggestions and improvements are welcome through issues and pull requests.

## License

This project is for educational purposes as part of Master Studies at the University of Belgrade.

## Acknowledgments

- FPL API for providing data access
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) for historical data repository

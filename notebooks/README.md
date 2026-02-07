# Notebooks

This directory contains Jupyter notebooks for exploratory analysis and experimentation.

## Notebooks Overview

### 01_exploratory_data_analysis.ipynb
Comprehensive exploratory data analysis covering:
- Data loading and structure overview
- Player distribution analysis
- Points distribution and statistics
- Position-based performance analysis
- Temporal trends across gameweeks
- Performance metrics correlation
- Team strength analysis
- Data quality assessment
- Key insights summary

## Running Notebooks

1. **Activate virtual environment**:
   ```bash
   .fpl\Scripts\activate
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

3. **Open notebook** and run cells sequentially

## Tips

- Make sure you've collected data before running the notebooks
- Adjust the `ANALYSIS_SEASON` variable to analyze different seasons
- Save figures using the output directories: `outputs/figures/`
- Export results to: `outputs/results/`

## Planned Notebooks

- `02_feature_engineering.ipynb` - Creating features for ML models
- `03_model_training.ipynb` - Training and comparing models
- `04_model_evaluation.ipynb` - Detailed evaluation and comparison with papers
- `05_predictions.ipynb` - Making predictions for upcoming gameweeks

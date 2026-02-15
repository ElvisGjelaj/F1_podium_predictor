# F1 Podium Predictor (2000-2024)

This project trains an XGBoost model to predict Formula 1 podium finishes (Top 3) using race/driver/constructor features.

## What it does
- Trains on seasons **2000-2023**
- Uses **2024** as the test/prediction season
- Predicts Top 3 probability per driver for each race
- Outputs a 2024 podium prediction table (top 3 per race) with:
  - `year`
  - `round`
  - `location`
  - `driver`
  - `constructor`
  - `placement` (predicted 1/2/3)
  - `prediction` (`right` / `wrong`)
- Outputs a driver-level podium tally table with:
  - `driver`
  - `constructor`
  - `predicted_podiums`
  - `actual_podiums`
  - `prediction_to_actual_ratio`
- Includes all 2024 drivers/constructors in the tally table, including `0` predicted and `0` actual podium rows
- Sorts tally table by `actual_podiums` (descending)

## Project files
- `f1_podium_predictor.ipynb` - Main notebook (training + prediction)
- `F1 Races 2020-2024.csv` - Feature dataset used by the model (contains seasons 2000-2024)
- `drivers.csv` - Driver ID to name lookup
- `constructors.csv` - Constructor ID to name lookup
- `races.csv` - Race ID to race name lookup

## Tech stack
- Python
- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn
- plotly

## Installation
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly
```

## How to run
1. Open `f1_podium_predictor.ipynb`
2. Run cells in order.
3. The notebook will:
   - Train model on 2000-2023
   - Predict 2024 podium probabilities
   - Print table 1: row-level predictions with `year, round, location, driver, constructor, placement, prediction`
   - Print table 2: driver/constructor podium tally with predicted vs actual counts and ratio

## Notes
- Name output comes from joins with `drivers.csv`, `constructors.csv`, and `races.csv`.
- `prediction = right` means the predicted podium driver actually finished Top 3 in that race.
- `prediction_to_actual_ratio = predicted_podiums / actual_podiums`; when `actual_podiums = 0`, ratio is `NaN`.

## Future improvements
- Hyperparameter tuning with cross-validation
- Calibrated probabilities
- Better handling of class imbalance
- Per-race confidence analysis and explainability reports

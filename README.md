# F1 Podium Predictor (2000-2024)

This project trains an XGBoost model to predict Formula 1 podium finishes (Top 3) using race/driver/constructor features.

## What it does
- Trains on seasons **2000-2023**
- Uses **2024** as the test/prediction season
- Predicts Top 3 probability per driver for each race
- Outputs predicted podiums (top 3 per race) with:
  - Race name
  - Driver name
  - Constructor name
  - Prediction probability
  - `prediction_result` (`right` / `wrong`)

## Project files
- `f1_podium_predictor.ipynb` - Main notebook (training + prediction)
- `F1 Races 2020-2024.csv` - Feature dataset used by the model
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
   - Print final table with race/driver/constructor names and `prediction_result`

## Notes
- Name output comes from joins with `drivers.csv`, `constructors.csv`, and `races.csv`.
- `prediction_result = right` means the predicted podium driver actually finished Top 3 in that race.

## Future improvements
- Hyperparameter tuning with cross-validation
- Calibrated probabilities
- Better handling of class imbalance
- Per-race confidence analysis and explainability reports

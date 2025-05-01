
# Predicting Residential Burglary Counts in London

## Description
This project aims to predict monthly residential burglary counts at the Lower Layer Super Output Area (LSOA) level in London, UK. 
It utilizes data from data.police.uk (burglary records), Indices of Multiple Deprivation (IMD), and census data.

The prediction model is built using XGBoost, and SHAP (SHapley Additive exPlanations) is used for model interpretation.

## Folder Structure

- **/data/**: Contains raw and processed datasets.
  - `burglary_data.csv`: Raw burglary records.
  - `imd_data.csv`: Indices of Multiple Deprivation data.
  - `census_data.csv`: Census data relevant to demographics and socio-economic factors.
  - `features_data.csv`: Combined and processed features before splitting.
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: Training and testing data splits.
- **/models/**: Contains trained machine learning models.
  - `xgboost_model.pkl`: Trained XGBoost model saved using pickle or joblib.
- **/plots/**: Contains generated plots and visualizations.
  - `shap_summary.png`: SHAP summary plot.
  - `shap_dependence.png`: Example SHAP dependence plots.
- **/scripts/**: Contains Python scripts for the project workflow.
  - `setup_folders.py`: This script, used for initial folder setup.
  - `data_fetch.py`: Script to download or fetch raw data.
  - `preprocess.py`: Script for data cleaning, feature engineering, and preprocessing.
  - `train_model.py`: Script to train the prediction model.
  - `interpret_shap.py`: Script to generate SHAP interpretations.

## Setup Instructions
The Python virtual environment (`burglary_env`) and package installation are handled by `setup_env.sh` and `install_packages.py`.
The Git repository is initialized by `setup_env.sh` and connected to: https://github.com/NishDaswani/DC2.git

To activate the environment:
```bash
source ../burglary_env/bin/activate # If in scripts directory
# or
source burglary_env/bin/activate    # If in project root
```

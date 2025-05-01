# Project Title
Predictive Modeling for Residential Burglary Forecasting and Police Resource Allocation in London

# Project Overview
This project focuses on forecasting residential burglary risk at the ward level in London using predictive modeling. The goal is to assist law enforcement (Metropolitan Police) in making data-driven decisions about where and when to deploy burglary-focused patrol resources under real-world operational constraints.

# Objectives
- Predict monthly residential burglary counts for each electoral ward in London.
- Use forecasts to recommend weekly patrol hour allocations (max 200 hours/week/ward).
- Ensure recommendations are ethical, interpretable, and operationally feasible.
- Incorporate external data to improve prediction quality (e.g., deprivation, transport).
- Evaluate the limitations of prediction under non-stationary conditions and possible intervention effects.

# Methods
## Data
- Historical burglary records from [data.police.uk](https://data.police.uk)
- Aggregated at the ward-month level
- Additional sources:
  - Index of Multiple Deprivation (IMD)
  - Public transport locations
  - Census housing and population data

## Features
- Lagged burglary counts (1, 2, 3 months)
- Rolling averages
- Month and seasonal indicators
- Public holiday flags
- IMD score per ward
- Population density
- Transport access density (bus stops, train stations)
- Special operation flags (if applicable)

## Predictive Models
Candidate models to compare:
- ARIMA / SARIMA
- Prophet
- XGBoost
- LSTM (TensorFlow/Keras)

Evaluation metrics:
- RMSE
- MAPE
- Directional accuracy

# Patrol Allocation Logic
- Patrol hours are allocated per ward, constrained to 200 hours/week
- Allocation strategies considered:
  - Proportional to predicted burglary risk
  - Tiered thresholds (High, Medium, Low risk)
  - Baseline coverage + surplus allocation
  - Fairness-aware constraints (avoid over-policing vulnerable communities)

# Output
- Monthly predictions per ward
- Allocation recommendation engine (based on prediction + rule)
- Visualizations: forecast trends, heatmaps, patrol suggestions
- (Optional) Interactive dashboard for decision-makers

# Limitations
- Models may not capture causality (e.g., whether patrols caused crime drop)
- Crime trends are non-stationary; features designed to mitigate this
- Prediction is probabilistic; outputs are used for risk prioritization, not certainty

# Ethical Considerations
- Risk of biased targeting in high-deprivation areas
- Allocation fairness is part of the modeling design
- Transparency and explainability (e.g., feature importance) included

# Status
Early-stage: defining sub-questions, unit of analysis (ward-month), feature set, and initial model selection.

# Tags
predictive-modeling, time-series, crime-forecasting, resource-allocation, ethics, nonstationary-data, london-police, patrol-optimization

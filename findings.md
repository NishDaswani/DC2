# Residential Burglary Prediction Project - Key Findings

This document captures key insights and findings as the project progresses. These observations can guide future model iterations and help focus on the most promising approaches.

## Baseline Model Findings (XGBoost with Time-based Features Only)

### Model Performance
- **Average RMSE**: ~1.11 across 5 time series CV folds
- **Average MAE**: ~0.80 across 5 time series CV folds
- **Interpretation concern**: With many LSOAs having very low burglary counts (0-2 per month), an MAE of 0.8 indicates considerable relative error. For areas with just 1 burglary, being off by 0.8 represents an 80% error.

### Feature Importance
- **Most predictive feature**: `rolling_mean_12` (12-month rolling average of burglary counts)
- **Insight**: Long-term historical patterns (yearly averages) are more predictive than short-term fluctuations. This suggests:
  1. Burglary risk has significant geographical stability over time
  2. Areas don't rapidly change from low-risk to high-risk (or vice versa)
  3. Annual seasonality may be captured in this long-term average

### Implications for Future Models
1. **External data priority**: Given the importance of `rolling_mean_12`, we should prioritize adding features related to stable area characteristics in future iterations:
   - Indices of Multiple Deprivation (IMD)
   - Housing types and density
   - Demographic factors from census data
   - Transport accessibility

2. **Evaluation approaches to consider**:
   - Calculate distribution of burglary counts to better interpret error metrics
   - Consider Mean Absolute Percentage Error (MAPE) for relative error understanding
   - Evaluate model performance separately for different burglary count ranges (low/medium/high)
   - Consider classification approaches (predicting risk categories) rather than exact counts

3. **Data leakage vigilance**: The strong performance of long-term averages emphasizes the importance of proper time series cross-validation to prevent future data from influencing predictions of past periods.

## Future Sections to Add
*(These sections will be populated as the project progresses)*

### IMD Data Integration Findings

### Transport Data Integration Findings

### Census Data Integration Findings

### Final Model Performance 
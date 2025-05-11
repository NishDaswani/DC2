# Data Preprocessing Steps

- Original data shape: (416974, 30)
- Converted 'Month' column to datetime format
- Sorted data by LSOA and Month
- Created target column (next month's burglary count)
- Dropped rows with missing target values
- Filled missing values in lag and rolling features with 0
- Filled other missing numeric values with median
- Dropped non-numeric columns: ['Month', 'LSOA11CD', 'LSOA Name', 'Year']
- Final feature set shape: (416974, 24)
- Features used: Population, area_km2, population_density, claimant_rate, poi_count, IncScore, month_nr, burglary_lag_1, burglary_lag_3, burglary_lag_12, burglary_rolling_mean_3, burglary_volatility_3, burglary_rolling_mean_6, burglary_rolling_std_6, burglary_rolling_mean_12, burglary_volatility_12, burglary_rolling_max_3, burglary_rolling_min_3, burglary_rolling_max_6, burglary_rolling_min_6, burglary_rolling_max_12, burglary_rolling_min_12, burglary_trend_3_12, burglary_trend_6_12

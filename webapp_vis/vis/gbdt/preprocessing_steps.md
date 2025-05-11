# Data Preprocessing Steps for GBDT Model

- Original data shape: (416974, 32)
- Converted 'Month' column to datetime format
- Sorted data by LSOA and Month
- Created target column (next month's burglary count)
- Dropped rows with missing target values
- Filled missing values in lag and rolling features with 0
- Filled other missing numeric values with median
- Dropped non-numeric columns: ['Month', 'LSOA11CD', 'LSOA Name', 'Year']
- Final feature set shape: (416974, 26)
- Features used: Population, area_km2, population_density, claimant_rate, poi_count, IncScore, month_nr, month_1, month_2, month_3, month_4, month_5, month_6, month_7, month_8, month_9, month_10, month_11, month_12, burglary_lag_1, burglary_lag_3, burglary_lag_12, burglary_rolling_mean_3, burglary_rolling_std_3, burglary_rolling_mean_12, burglary_rolling_std_12

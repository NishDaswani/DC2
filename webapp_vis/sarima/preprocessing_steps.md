# Data Preprocessing Steps for SARIMA Model

- Original data shape: (416974, 32)
- Converted 'Month' column to datetime format
- Sorted data by LSOA and Month
- Created target column (next month's burglary count)
- Dropped rows with missing target values
- Filled missing values in lag and rolling features with 0
- Filled other missing numeric values with median

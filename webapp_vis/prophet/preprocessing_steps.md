# Data Preprocessing Steps for PROPHET Model

- Original data shape: (422347, 29)
- Ensured 'Month' column is datetime format
- Sorted data by LSOA and Month
- Data will be split into 'ds' and 'y' for Prophet within the training loop for each LSOA.
- Selected exogenous regressors will be added to the model.

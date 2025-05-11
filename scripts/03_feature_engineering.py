'''
Script for further feature engineering:
- One-hot encode categorical features.
- Scale numerical features.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import numpy as np

# Define file paths
PROCESSED_DATA_DIR = "data/processed" # Assuming script is run from scripts/
INPUT_CSV = os.path.join(PROCESSED_DATA_DIR, "merged_data.csv")
FEATURES_OUTPUT_DIR = "data/features" # For final features
FEATURES_OUTPUT_CSV = os.path.join(FEATURES_OUTPUT_DIR, "final_features.csv")

print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print("Data loaded successfully.")
print("Shape of a loaded DataFrame:", df.shape)
print("Columns:", df.columns.tolist())
print("\nData Info:")
df.info()
print("\nSample of data:")
print(df.head())

print("\nIdentifying features for preprocessing...")

# Target variable
target = 'burglary_count'

# Identifier columns (to keep as is, not for modeling directly in this way)
identifier_cols = ['LSOA code', 'Month']

# Categorical features for One-Hot Encoding
categorical_features = ['month_num']

# Numerical features for Scaling
# All other columns that are not target, identifiers, or categorical (to be OHE'd)
numerical_features = [col for col in df.columns if col not in identifier_cols + [target] + categorical_features]

print(f"Target: {target}")
print(f"Identifier columns: {identifier_cols}")
print(f"Categorical features for OHE: {categorical_features}")
print(f"Numerical features for Scaling: {numerical_features}")

# Separate features (X) and target (y) for clarity, though we transform X then add y back
X = df.drop(columns=[target])
y = df[target]

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for easier merge later

# Create a column transformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep identifier_cols as they are, they will appear at the end
)

print("\nApplying transformations (scaling and OHE)...")
# Apply the transformations - we want to transform X and then add y and identifiers back
# To ensure correct column order for passthrough, let's re-order X before fitting
X_ordered = X[numerical_features + categorical_features + identifier_cols]

X_processed_np = preprocessor.fit_transform(X_ordered)

# Get feature names after OHE
# Note: Order is important: numerical_features, then OHE names, then remainder columns
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
processed_feature_names = numerical_features + ohe_feature_names.tolist() + identifier_cols

X_processed = pd.DataFrame(X_processed_np, columns=processed_feature_names, index=X.index)

print("Transformation complete.")
print("Shape of processed X features:", X_processed.shape)
print("Columns in processed X:", X_processed.columns.tolist())

# Combine processed features with the target variable and original identifiers if they were dropped
# The passthrough in ColumnTransformer already keeps identifier_cols
# We need to re-attach the target variable
final_df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

# Ensure correct column order if LSOA code and Month were not at the end due to passthrough specifics
# For safety, explicitly reorder to have identifiers first, then features, then target
final_cols_ordered = identifier_cols + \
                     [col for col in processed_feature_names if col not in identifier_cols] + \
                     [target]
final_df = final_df[final_cols_ordered]

print("\nFinal DataFrame after processing and re-combining:")
print("Shape:", final_df.shape)
print("Columns:", final_df.columns.tolist())
print(final_df.head())

# --- Save processed data ---
print("\nSaving processed data...")
try:
    os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(FEATURES_OUTPUT_CSV, index=False)
    print(f"Processed data saved to {FEATURES_OUTPUT_CSV}")
except Exception as e:
    print(f"Error saving processed data: {e}")

print("\nScript 03_feature_engineering.py finished.") 
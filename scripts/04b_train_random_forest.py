'''
Script for training the RandomForest regression model using TimeSeriesSplit CV.
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor # Changed import
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib # For saving sklearn models
import os

# Define file paths
FEATURES_DATA_DIR = "data/features" # Assuming script is run from scripts/
INPUT_CSV = os.path.join(FEATURES_DATA_DIR, "final_features.csv")
MODELS_DIR = "models" # To save trained model later
MODEL_FILENAME = "rf_model.joblib" # Changed filename
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print("Data loaded successfully.")
print("Shape of loaded DataFrame:", df.shape)

# --- Data Preparation for Time Series --- 
df['Month'] = pd.to_datetime(df['Month'])
df.sort_values(by=['Month', 'LSOA code'], inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nData sorted by Month and LSOA code.")
print(df.head())

# Define features (X) and target (y)
target = 'burglary_count'
features_to_exclude = ['LSOA code', 'Month', target]
X = df.drop(columns=features_to_exclude)
y = df[target]

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("Feature columns:", X.columns.tolist())

# --- TimeSeriesSplit Cross-Validation --- 
print("\nSetting up TimeSeriesSplit Cross-Validation...")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

mse_scores = []
r2_scores = []

print(f"Performing {n_splits}-fold TimeSeriesSplit CV with RandomForest...") # Changed model name

fold = 0
for train_index, val_index in tscv.split(X):
    fold += 1
    print(f"\nFold {fold}/{n_splits}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    print(f"  Train set size: {X_train.shape[0]} samples")
    print(f"  Validation set size: {X_val.shape[0]} samples")

    # Initialize and train RandomForest model
    model = RandomForestRegressor( # Changed model
        n_estimators=100,      # Default, comparable to XGBoost n_estimators
        random_state=42,
        n_jobs=-1,
        oob_score=False,       # OOB score is useful but slower, can be enabled
        max_depth=None,        # Default, lets trees grow deep, can be tuned
        min_samples_split=2,   # Default
        min_samples_leaf=1     # Default
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_val = model.predict(X_val)
    
    # Evaluate
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    
    mse_scores.append(mse)
    r2_scores.append(r2)
    
    print(f"  Fold {fold} MSE: {mse:.4f}")
    print(f"  Fold {fold} R2: {r2:.4f}")

# --- Report Results --- 
print("\nCross-Validation Results (RandomForest):") # Changed model name
print(f"Average MSE: {np.mean(mse_scores):.4f} (Std: {np.std(mse_scores):.4f})")
print(f"Average R2: {np.mean(r2_scores):.4f} (Std: {np.std(r2_scores):.4f})")

# --- Train final model on the last training split for SHAP analysis --- 
print("\nTraining final RandomForest model on the last training split...") # Changed model name
final_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train) # X_train, y_train hold data from the last fold
print("Final RandomForest model trained.")

# Ensure model directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Saving final model to {MODEL_PATH}...")
try:
    joblib.dump(final_model, MODEL_PATH) # Use joblib to save sklearn models
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")


# --- TODO: Evaluate on a held-out test set (if one was created) ---

print("\nScript 04b_train_random_forest.py finished.") 
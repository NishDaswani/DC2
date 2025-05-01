# Placeholder for train_model.py

'''
Trains an XGBoost model to predict monthly burglary counts per LSOA.

Loads engineered features, uses Time Series Cross-Validation for evaluation,
trains a final model, and saves it. Also creates visualizations of model performance.
'''

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib # Using joblib for potentially better handling of large numpy arrays in models
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- Configuration ---
# Define the project root assuming this script is in the 'scripts' directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
INPUT_FILE = os.path.join(INPUT_DIR, 'features_engineered.csv')
OUTPUT_MODEL_FILE = os.path.join(MODEL_DIR, 'xgboost_model.joblib') # Changed to joblib

# Model & Evaluation Parameters
TARGET_VARIABLE = 'burglary_count'
# Features to exclude (identifiers, target, exact date)
FEATURES_TO_EXCLUDE = ['LSOA code', 'Month', TARGET_VARIABLE]

# XGBoost Parameters (Example - can be tuned later)
XGB_PARAMS = {
    'objective': 'count:poisson', # Good objective for count data
    'n_estimators': 100,          # Number of boosting rounds
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,            # Fraction of samples used per tree
    'colsample_bytree': 0.8,     # Fraction of features used per tree
    'random_state': 42,
    'n_jobs': -1,                 # Use all available CPU cores
    'early_stopping_rounds': 10  # Stop if validation metric doesn't improve for 10 rounds
}

# Time Series Cross-Validation Parameters
N_SPLITS = 5 # Number of folds for TSCV

# --- Helper Functions for Visualization ---
def plot_actual_vs_predicted(fold_data, output_file):
    '''Create a scatter plot of actual vs predicted values across all folds.'''
    plt.figure(figsize=(10, 8))
    
    # Plot each fold with a different color
    for fold, data in enumerate(fold_data):
        plt.scatter(data['y_true'], data['y_pred'], 
                    alpha=0.5, label=f'Fold {fold+1}')
    
    # Add y=x reference line (perfect predictions)
    max_val = max([max(data['y_true'].max(), data['y_pred'].max()) for data in fold_data])
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect predictions')
    
    plt.xlabel('Actual burglary count')
    plt.ylabel('Predicted burglary count')
    plt.title('Actual vs Predicted Burglary Counts Across CV Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_prediction_errors(fold_data, output_file):
    '''Create a histogram of prediction errors (residuals) across all folds.'''
    plt.figure(figsize=(10, 6))
    
    for fold, data in enumerate(fold_data):
        # Calculate residuals
        residuals = data['y_true'] - data['y_pred']
        plt.hist(residuals, bins=30, alpha=0.5, label=f'Fold {fold+1}')
    
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors Across CV Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_metrics_by_fold(metrics, output_file):
    '''Create a bar chart of evaluation metrics by fold.'''
    plt.figure(figsize=(12, 6))
    
    # Number of folds
    folds = range(1, len(metrics['rmse'])+1)
    bar_width = 0.35
    
    # Plotting RMSE bars
    plt.bar([i - bar_width/2 for i in folds], metrics['rmse'], 
            width=bar_width, label='RMSE', color='royalblue')
    
    # Plotting MAE bars
    plt.bar([i + bar_width/2 for i in folds], metrics['mae'], 
            width=bar_width, label='MAE', color='lightcoral')
    
    # Add horizontal line for average metrics
    plt.axhline(y=np.mean(metrics['rmse']), color='royalblue', 
                linestyle='--', alpha=0.7, label=f'Avg RMSE: {np.mean(metrics["rmse"]):.4f}')
    plt.axhline(y=np.mean(metrics['mae']), color='lightcoral', 
                linestyle='--', alpha=0.7, label=f'Avg MAE: {np.mean(metrics["mae"]):.4f}')
    
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.title('RMSE and MAE by Cross-Validation Fold')
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_feature_importance(model, feature_names, output_file):
    '''Create a bar chart of feature importance.'''
    # Get importance scores
    importance = model.feature_importances_
    # Sort features by importance
    indices = np.argsort(importance)
    
    plt.figure(figsize=(10, 8))
    # Plot as horizontal bar chart
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# --- Main Script Logic ---
if __name__ == "__main__":
    print("--- Starting Model Training ---")
    
    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Get timestamp for unique plot filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load Data
    print(f"Loading engineered features from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, parse_dates=['Month'])
    except FileNotFoundError:
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        print("Please run the feature_engineering.py script first.")
        sys.exit(1)
    except KeyError:
         print(f"Error: 'Month' column not found or couldn't be parsed as date in {INPUT_FILE}", file=sys.stderr)
         sys.exit(1)


    print(f"Loaded data shape: {df.shape}")

    # IMPORTANT: Sort data by time for TimeSeriesSplit
    df.sort_values(by='Month', inplace=True)
    print("Data sorted by Month.")

    # 2. Define Features (X) and Target (y)
    print("\n--- Defining Features and Target ---")
    features = [col for col in df.columns if col not in FEATURES_TO_EXCLUDE]
    X = df[features]
    y = df[TARGET_VARIABLE]
    print(f"Features ({len(features)}): {features}")
    print(f"Target: {TARGET_VARIABLE}")

    # 3. Time Series Cross-Validation Setup
    print(f"\n--- Setting up Time Series Cross-Validation (n_splits={N_SPLITS}) ---")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_metrics = {'rmse': [], 'mae': []}
    fold_data = [] # Store prediction data for visualization
    
    # 4. Training and Evaluation Loop
    print("\n--- Starting Cross-Validation Training ---")
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"  Fold {fold+1}/{N_SPLITS}...")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print(f"    Train size: {len(X_train)}, Validation size: {len(X_val)}")
        print(f"    Train period: {df.iloc[train_index]['Month'].min().strftime('%Y-%m')} to {df.iloc[train_index]['Month'].max().strftime('%Y-%m')}")
        print(f"    Validation period: {df.iloc[val_index]['Month'].min().strftime('%Y-%m')} to {df.iloc[val_index]['Month'].max().strftime('%Y-%m')}")


        # Initialize model for this fold
        model = xgb.XGBRegressor(**XGB_PARAMS)

        # Train the model with early stopping
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False) # Set verbose=True to see XGBoost training output

        # Make predictions
        y_pred = model.predict(X_val)

        # Ensure predictions are non-negative (important for count data)
        y_pred = np.maximum(0, y_pred)

        # Store actual and predicted values for visualization
        fold_data.append({
            'fold': fold + 1,
            'y_true': y_val.values,
            'y_pred': y_pred,
            'train_start': df.iloc[train_index]['Month'].min().strftime('%Y-%m'),
            'train_end': df.iloc[train_index]['Month'].max().strftime('%Y-%m'),
            'val_start': df.iloc[val_index]['Month'].min().strftime('%Y-%m'),
            'val_end': df.iloc[val_index]['Month'].max().strftime('%Y-%m'),
        })

        # Evaluate
        # Calculate MSE first
        mse = mean_squared_error(y_val, y_pred)
        # Then take sqrt for RMSE (compatible with older sklearn versions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)

        fold_metrics['rmse'].append(rmse)
        fold_metrics['mae'].append(mae)
        print(f"    Fold {fold+1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 5. Report Average CV Results
    print("\n--- Cross-Validation Results ---")
    avg_rmse = np.mean(fold_metrics['rmse'])
    avg_mae = np.mean(fold_metrics['mae'])
    print(f"Average RMSE across {N_SPLITS} folds: {avg_rmse:.4f}")
    print(f"Average MAE across {N_SPLITS} folds: {avg_mae:.4f}")

    # Generate and save visualizations
    print("\n--- Generating Performance Visualizations ---")
    
    # Actual vs Predicted plot
    actual_vs_pred_file = os.path.join(PLOTS_DIR, f"actual_vs_predicted_{timestamp}.png")
    plot_actual_vs_predicted(fold_data, actual_vs_pred_file)
    print(f"Actual vs Predicted plot saved to {actual_vs_pred_file}")
    
    # Error distribution plot
    errors_file = os.path.join(PLOTS_DIR, f"prediction_errors_{timestamp}.png")
    plot_prediction_errors(fold_data, errors_file)
    print(f"Prediction Errors plot saved to {errors_file}")
    
    # Metrics by fold plot
    metrics_file = os.path.join(PLOTS_DIR, f"metrics_by_fold_{timestamp}.png")
    plot_metrics_by_fold(fold_metrics, metrics_file)
    print(f"Metrics by Fold plot saved to {metrics_file}")

    # 6. Final Model Training
    print("\n--- Training Final Model on All Data ---")
    # Create parameters for the final model by removing early stopping
    final_model_params = XGB_PARAMS.copy()
    final_model_params.pop('early_stopping_rounds', None)

    # Initialize the final model with the modified parameters
    final_model = xgb.XGBRegressor(**final_model_params)

    final_model.fit(X, y, verbose=False) # Train on the entire dataset X, y
    print("Final model trained.")
    
    # Generate feature importance plot
    importance_file = os.path.join(PLOTS_DIR, f"feature_importance_{timestamp}.png")
    plot_feature_importance(final_model, features, importance_file)
    print(f"Feature Importance plot saved to {importance_file}")

    # 7. Save Final Model
    print(f"\n--- Saving Final Model to {OUTPUT_MODEL_FILE} ---")
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        joblib.dump(final_model, OUTPUT_MODEL_FILE)
        print("Model saved successfully using joblib.")
    except Exception as e:
        print(f"Error saving model: {e}", file=sys.stderr)

    print("\n--- Model Training Complete ---")

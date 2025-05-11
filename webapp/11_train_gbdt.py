import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import shap
from datetime import datetime
import pickle

# Define model name for directory creation and naming
MODEL_NAME = 'gbdt'
BASE_VIS_DIR = f'webapp_vis/vis/{MODEL_NAME}'

# Create output directories if they don't exist
os.makedirs(BASE_VIS_DIR, exist_ok=True)

def preprocess_data(df):
    """
    Preprocess the data for model training
    """
    print("Preprocessing data...")
    
    # Convert Month to datetime
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    
    # Sort by LSOA and Month
    df = df.sort_values(['LSOA11CD', 'Month'])
    
    # Create target column (next month's burglary count)
    df['target'] = df.groupby('LSOA11CD')['burglary_count'].shift(-1)
    
    # Drop rows with NaN target (last month for each LSOA)
    df = df.dropna(subset=['target'])
    
    # Fill NaN values in features
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64] and df[col].isna().sum() > 0:
            if 'lag' in col or 'rolling' in col:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
    
    drop_cols = ['Month', 'LSOA11CD', 'LSOA Name', 'Year']
    X = df.drop(drop_cols + ['target', 'burglary_count'], axis=1)
    y = df['target']
    
    with open(f'{BASE_VIS_DIR}/preprocessing_steps.md', 'w') as f:
        f.write(f"# Data Preprocessing Steps for {MODEL_NAME.upper()} Model\n\n")
        f.write(f"- Original data shape: {df.shape}\n")
        f.write("- Converted 'Month' column to datetime format\n")
        f.write("- Sorted data by LSOA and Month\n")
        f.write("- Created target column (next month's burglary count)\n")
        f.write("- Dropped rows with missing target values\n")
        f.write("- Filled missing values in lag and rolling features with 0\n")
        f.write("- Filled other missing numeric values with median\n")
        f.write(f"- Dropped non-numeric columns: {drop_cols}\n")
        f.write(f"- Final feature set shape: {X.shape}\n")
        f.write(f"- Features used: {', '.join(X.columns)}\n")
    
    return X, y, df

def train_gbdt_model(X, y, df):
    """
    Train GBDT model with time series cross-validation
    """
    print(f"Training {MODEL_NAME.upper()} model with time series cross-validation...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Parameters for GradientBoostingRegressor
    # These are example parameters; they might need tuning.
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5, # Adjusted from XGBoost's 6 for GBDT typical values
        'subsample': 0.8,
        'random_state': 42,
        # 'loss': 'squared_error' is default for regression
    }
    
    train_rmse_scores = []
    val_rmse_scores = []
    val_mae_scores = []
    val_r2_scores = []
    trained_models = []
    
    all_y_true_cv = []
    all_y_pred_cv = []
    all_fold_indices_cv = []
    
    fold = 0
    for train_index, val_index in tscv.split(X):
        fold += 1
        print(f"Training fold {fold}...")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Ensure predictions are non-negative
        y_train_pred = np.maximum(0, y_train_pred)
        y_val_pred = np.maximum(0, y_val_pred)
        
        train_rmse_fold = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae_fold = mean_absolute_error(y_val, y_val_pred)
        val_r2_fold = r2_score(y_val, y_val_pred)
        
        train_rmse_scores.append(train_rmse_fold)
        val_rmse_scores.append(val_rmse_fold)
        val_mae_scores.append(val_mae_fold)
        val_r2_scores.append(val_r2_fold)
        trained_models.append(model)
        
        all_y_true_cv.extend(y_val.tolist())
        all_y_pred_cv.extend(y_val_pred.tolist())
        all_fold_indices_cv.extend([fold] * len(y_val))
        
        print(f"Fold {fold} - Train RMSE: {train_rmse_fold:.4f}, Val RMSE: {val_rmse_fold:.4f}, Val MAE: {val_mae_fold:.4f}, Val R2: {val_r2_fold:.4f}")
    
    best_model_idx = np.argmin(val_rmse_scores)
    best_gbdt_model = trained_models[best_model_idx]
    
    with open(f'{BASE_VIS_DIR}/model_metrics.md', 'w') as f:
        f.write(f"# {MODEL_NAME.upper()} Model Performance Metrics\n\n")
        f.write("## Cross-Validation Results\n\n")
        f.write("| Fold | Train RMSE | Validation RMSE | Validation MAE | Validation R² |\n")
        f.write("|------|------------|-----------------|----------------|---------------|\n")
        for i in range(len(train_rmse_scores)):
            f.write(f"| {i+1}    | {train_rmse_scores[i]:.4f}      | {val_rmse_scores[i]:.4f}           | {val_mae_scores[i]:.4f}          | {val_r2_scores[i]:.4f}          |\n")
        f.write("\n## Average Metrics\n\n")
        f.write(f"- Average Train RMSE: {np.mean(train_rmse_scores):.4f}\n")
        f.write(f"- Average Validation RMSE: {np.mean(val_rmse_scores):.4f}\n")
        f.write(f"- Average Validation MAE: {np.mean(val_mae_scores):.4f}\n")
        f.write(f"- Average Validation R²: {np.mean(val_r2_scores):.4f}\n")
        f.write(f"\nBest model is from fold {best_model_idx+1} with validation RMSE of {val_rmse_scores[best_model_idx]:.4f}")
    
    return best_gbdt_model, all_y_true_cv, all_y_pred_cv, all_fold_indices_cv, X

def create_gbdt_visualizations(y_true, y_pred, fold_indices, X_features_df, model):
    """
    Create and save visualizations for GBDT model
    """
    print(f"Creating {MODEL_NAME.upper()} visualizations...")
    
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Fold': fold_indices
    })
    
    # 1. Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', hue='Fold', data=results_df, alpha=0.6)
    min_val = min(min(y_true, default=0), min(y_pred, default=0))
    max_val = max(max(y_true, default=0), max(y_pred, default=0))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--') # Diagonal line
    plt.title(f'{MODEL_NAME.upper()}: Actual vs Predicted Burglary Counts')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.tight_layout()
    plt.savefig(f'{BASE_VIS_DIR}/actual_vs_predicted.png')
    plt.close()
    
    # 2. Prediction error histogram
    errors = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title(f'{MODEL_NAME.upper()}: Distribution of Prediction Errors')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{BASE_VIS_DIR}/error_distribution.png')
    plt.close()
    
    # 3. Feature importance plot (from GBDT model)
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_n = 20 # Show top N features
    
    plt.figure(figsize=(12, 8))
    plt.title(f'{MODEL_NAME.upper()} Feature Importance')
    plt.bar(range(min(top_n, len(sorted_indices))), feature_importances[sorted_indices][:top_n], align='center')
    plt.xticks(range(min(top_n, len(sorted_indices))), X_features_df.columns[sorted_indices][:top_n], rotation=90)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{BASE_VIS_DIR}/feature_importance.png')
    plt.close()

    # Save GBDT feature importances to markdown
    feature_importance_dict = dict(zip(X_features_df.columns, feature_importances))
    sorted_features_gbdt = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    with open(f'{BASE_VIS_DIR}/feature_importance.md', 'w') as f:
        f.write(f"# {MODEL_NAME.upper()} Feature Importance (from model.feature_importances_)\n\n")
        f.write("| Feature | Importance |\n")
        f.write("|---------|------------|\n")
        for feature, importance in sorted_features_gbdt:
            f.write(f"| {feature} | {importance:.4f} |\n")

    # 4. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=errors, hue=fold_indices, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f'{MODEL_NAME.upper()}: Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(f'{BASE_VIS_DIR}/residuals_vs_predicted.png')
    plt.close()
    
    # 5. SHAP values
    print(f"Calculating SHAP values for {MODEL_NAME.upper()} model...")
    # Create a sample of data for SHAP analysis for speed
    # Ensure X_features_df is used here, which has the correct column names
    if len(X_features_df) > 1000:
         X_sample_shap = X_features_df.sample(1000, random_state=42)
    else:
         X_sample_shap = X_features_df.copy()

    try:
        explainer = shap.TreeExplainer(model, X_sample_shap) # Pass sample for background if needed for approximation
        shap_values = explainer.shap_values(X_sample_shap) # Calculate SHAP values for the sample
        
        plt.figure()
        shap.summary_plot(shap_values, X_sample_shap, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance (Bar) for {MODEL_NAME.upper()}')
        plt.tight_layout()
        plt.savefig(f'{BASE_VIS_DIR}/shap_summary_bar.png')
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_sample_shap, show=False) # Dot plot
        plt.title(f'SHAP Feature Importance (Summary) for {MODEL_NAME.upper()}')
        plt.tight_layout()
        plt.savefig(f'{BASE_VIS_DIR}/shap_summary_dot.png')
        plt.close()
        
        abs_shap_values = np.abs(shap_values)
        mean_abs_shap = np.mean(abs_shap_values, axis=0)
        feature_importance_shap_dict = dict(zip(X_sample_shap.columns, mean_abs_shap))
        sorted_features_shap = sorted(feature_importance_shap_dict.items(), key=lambda x: x[1], reverse=True)
        
        with open(f'{BASE_VIS_DIR}/shap_results.md', 'w') as f:
            f.write(f"# SHAP Feature Importance Results for {MODEL_NAME.upper()}\n\n")
            f.write("SHAP values show the contribution of each feature to the prediction for the sampled data.\n\n")
            f.write("## Top Features by Mean Absolute SHAP Value\n\n")
            f.write("| Feature | Mean |SHAP| Value |\n")
            f.write("|---------|----------------|\n")
            for feature, importance in sorted_features_shap:
                f.write(f"| {feature} | {importance:.4f}       |\n")
    except Exception as e:
        print(f"Error calculating or plotting SHAP values for {MODEL_NAME.upper()}: {e}")

def main():
    print(f"Starting {MODEL_NAME.upper()} model training script...")
    
    data_path = 'data/00_new/final_data_features.csv' #=======================================================Change path if needed===============================================
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Exiting.")
        return

    X, y, processed_df = preprocess_data(df)
    
    best_model, y_true, y_pred, fold_indices, X_for_viz = train_gbdt_model(X, y, processed_df)
    
    # Pass X_for_viz which has the correct column names for feature importance and SHAP
    create_gbdt_visualizations(y_true, y_pred, fold_indices, X_for_viz, best_model)
    
    model_pkl_path = f'{BASE_VIS_DIR}/{MODEL_NAME}_model.pkl'
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {model_pkl_path}")
    
    print(f"{MODEL_NAME.upper()} training and visualization completed!")

if __name__ == "__main__":
    main()

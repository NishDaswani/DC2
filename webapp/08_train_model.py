import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import shap
from datetime import datetime
import pickle

# Create output directories if they don't exist
os.makedirs('webapp_vis/vis/xgboost', exist_ok=True)

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
            # For lag features, fill with 0 (assume no burglaries if we don't have data)
            if 'lag' in col or 'rolling' in col:
                df[col] = df[col].fillna(0)
            # For other numeric columns, fill with median
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Drop non-numeric columns that won't be used for modeling
    drop_cols = ['Month', 'LSOA11CD', 'LSOA Name', 'Year']
    X = df.drop(drop_cols + ['target', 'burglary_count'], axis=1)
    y = df['target']
    
    # Document the preprocessing steps
    with open('webapp_vis/vis/xgboost/preprocessing_steps.md', 'w') as f:
        f.write("# Data Preprocessing Steps\n\n")
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

def train_model(X, y, df):
    """
    Train XGBoost model with time series cross-validation
    """
    print("Training model with time series cross-validation...")
    
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Lists to store results
    train_rmse = []
    val_rmse = []
    val_mae = []
    val_r2 = []
    models = []
    
    # Lists to store actual vs predicted values for plotting
    all_y_true = []
    all_y_pred = []
    all_fold_indices = []
    
    # Perform cross-validation
    fold = 0
    for train_index, val_index in tscv.split(X):
        fold += 1
        print(f"Training fold {fold}...")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions
        y_train_pred = model.predict(dtrain)
        y_val_pred = model.predict(dval)
        
        # Calculate metrics
        train_rmse_fold = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae_fold = mean_absolute_error(y_val, y_val_pred)
        val_r2_fold = r2_score(y_val, y_val_pred)
        
        # Store metrics
        train_rmse.append(train_rmse_fold)
        val_rmse.append(val_rmse_fold)
        val_mae.append(val_mae_fold)
        val_r2.append(val_r2_fold)
        models.append(model)
        
        # Store actual vs predicted for this fold
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_val_pred.tolist())
        all_fold_indices.extend([fold] * len(y_val))
        
        print(f"Fold {fold} - Train RMSE: {train_rmse_fold:.4f}, Val RMSE: {val_rmse_fold:.4f}, Val MAE: {val_mae_fold:.4f}, Val R2: {val_r2_fold:.4f}")
    
    # Select best model based on validation RMSE
    best_model_idx = np.argmin(val_rmse)
    best_model = models[best_model_idx]
    
    # Save model metrics
    with open('webapp_vis/vis/xgboost/model_metrics.md', 'w') as f:
        f.write("# XGBoost Model Performance Metrics\n\n")
        f.write("## Cross-Validation Results\n\n")
        f.write("| Fold | Train RMSE | Validation RMSE | Validation MAE | Validation R² |\n")
        f.write("|------|-----------|----------------|---------------|---------------|\n")
        for i in range(len(train_rmse)):
            f.write(f"| {i+1}    | {train_rmse[i]:.4f}     | {val_rmse[i]:.4f}          | {val_mae[i]:.4f}         | {val_r2[i]:.4f}         |\n")
        f.write("\n## Average Metrics\n\n")
        f.write(f"- Average Train RMSE: {np.mean(train_rmse):.4f}\n")
        f.write(f"- Average Validation RMSE: {np.mean(val_rmse):.4f}\n")
        f.write(f"- Average Validation MAE: {np.mean(val_mae):.4f}\n")
        f.write(f"- Average Validation R²: {np.mean(val_r2):.4f}\n")
        f.write(f"\nBest model is from fold {best_model_idx+1} with validation RMSE of {val_rmse[best_model_idx]:.4f}")
    
    return best_model, all_y_true, all_y_pred, all_fold_indices, X

def create_visualizations(y_true, y_pred, fold_indices, X, model):
    """
    Create and save visualizations
    """
    print("Creating visualizations...")
    
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Fold': fold_indices
    })
    
    # 1. Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', hue='Fold', data=results_df, alpha=0.6)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
    plt.title('Actual vs Predicted Burglary Counts')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/xgboost/actual_vs_predicted.png')
    plt.close()
    
    # 2. Prediction error histogram
    plt.figure(figsize=(10, 6))
    errors = np.array(y_true) - np.array(y_pred)
    sns.histplot(errors, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/xgboost/error_distribution.png')
    plt.close()
    
    # 3. Feature importance plot
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/xgboost/feature_importance.png')
    plt.close()
    
    # 4. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=errors, hue=fold_indices, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/xgboost/residuals_vs_predicted.png')
    plt.close()
    
    # 5. SHAP values
    print("Calculating SHAP values...")
    # Create a sample of data for SHAP analysis (for speed)
    X_sample = X.sample(min(1000, len(X)), random_state=42)
    X_sample_dmatrix = xgb.DMatrix(X_sample)
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
        shap_df.to_csv('webapp_vis/vis/xgboost/shap_values.csv', index=False)
        print("SHAP values saved to webapp_vis/vis/xgboost/shap_values.csv")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('webapp_vis/vis/xgboost/shap_summary.png')
        plt.close()
        
        # Save SHAP results to markdown
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_dict = dict(zip(X.columns, feature_importance))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        with open('webapp_vis/vis/xgboost/shap_results.md', 'w') as f:
            f.write("# SHAP Feature Importance Results\n\n")
            f.write("SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the prediction.\n\n")
            f.write("## Top Features by SHAP Importance\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|------------|\n")
            for feature, importance in sorted_features:
                f.write(f"| {feature} | {importance:.4f} |\n")
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        # Create a simple feature importance plot as fallback
        feature_importance = model.get_score(importance_type='gain')
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        with open('webapp_vis/vis/xgboost/feature_importance.md', 'w') as f:
            f.write("# XGBoost Feature Importance\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|------------|\n")
            for feature, importance in sorted_features:
                f.write(f"| {feature} | {importance:.4f} |\n")

def main():
    print("Starting XGBoost model training...")
    
    # Load the data
    data_path = 'data/00_new/final_data_features.csv' #=======================================================Change path if needed===============================================
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess the data
    X, y, processed_df = preprocess_data(df)
    
    # Train the model
    best_model, y_true, y_pred, fold_indices, X = train_model(X, y, processed_df)
    
    # Create visualizations
    create_visualizations(y_true, y_pred, fold_indices, X, best_model)
    
    # Save the model in multiple formats
    model_json_path = 'webapp_vis/vis/xgboost/xgboost_model.json'
    model_pkl_path = 'webapp_vis/vis/xgboost/xgboost_model.pkl'
    
    # Save as JSON (native XGBoost format)
    best_model.save_model(model_json_path)
    
    # Save as pickle file
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Model saved to {model_json_path} and {model_pkl_path}")
    
    print("Training and visualization completed!")

if __name__ == "__main__":
    main()

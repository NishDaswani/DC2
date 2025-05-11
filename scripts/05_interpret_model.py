'''
Script to interpret the trained XGBoost model using SHAP.
'''
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file paths
FEATURES_DATA_DIR = "data/features" # Assuming script is run from scripts/
INPUT_CSV = os.path.join(FEATURES_DATA_DIR, "final_features.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
OUTPUT_DIR = "reports/figures" # Directory to save plots

print("--- Model Interpretation using SHAP ---")

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = xgb.XGBRegressor()
try:
    model.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Exiting.")
    exit()

# --- Load Data (potentially a sample for SHAP speed) ---
# We need the feature data (X) corresponding to the model training or a representative sample.
# For now, let's load the full feature set used in the previous script.
# SHAP can be slow on very large datasets. Consider sampling if needed: df.sample(n=1000)
print(f"Loading data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
    # Recreate the feature set X exactly as used for training the final model
    target = 'burglary_count'
    features_to_exclude = ['LSOA code', 'Month', target]
    X = df.drop(columns=features_to_exclude)
    print(f"Data loaded successfully. Features shape: {X.shape}")
except Exception as e:
    print(f"Error loading data: {e}. Exiting.")
    exit()

# --- Calculate SHAP Values ---
# Use TreeExplainer for tree-based models like XGBoost
print("\nCalculating SHAP values (this may take a while)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X) # Calculate for all data (or sample)
print("SHAP values calculated.")

# --- Generate SHAP Plots ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Summary Plot (Global Feature Importance)
print("Generating SHAP summary plot...")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
summary_plot_path = os.path.join(OUTPUT_DIR, 'shap_summary_bar.png')
plt.savefig(summary_plot_path, bbox_inches='tight')
plt.close()
print(f"Summary plot saved to {summary_plot_path}")

plt.figure()
shap.summary_plot(shap_values, X, show=False)
summary_dot_plot_path = os.path.join(OUTPUT_DIR, 'shap_summary_dot.png')
plt.savefig(summary_dot_plot_path, bbox_inches='tight')
plt.close()
print(f"Summary dot plot saved to {summary_dot_plot_path}")

# 2. Dependence Plots (for top features)
# Let's plot for the top 5 features based on mean absolute SHAP value
print("\nGenerating SHAP dependence plots for top features...")
# Calculate mean absolute SHAP values for feature importance
vals= np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

top_n = 5
for i, feature in enumerate(feature_importance['col_name'].head(top_n)):
    print(f"  Generating dependence plot for: {feature}")
    plt.figure()
    shap.dependence_plot(feature, shap_values, X, interaction_index=None, show=False)
    dependence_plot_path = os.path.join(OUTPUT_DIR, f'shap_dependence_{i+1}_{feature}.png')
    plt.savefig(dependence_plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Dependence plot saved to {dependence_plot_path}")

print("\nScript 05_interpret_model.py finished.") 
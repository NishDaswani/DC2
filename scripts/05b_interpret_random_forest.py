'''
Script to interpret the trained RandomForest model using SHAP.
'''
import pandas as pd
import joblib # Changed import
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file paths
FEATURES_DATA_DIR = "data/features" # Assuming script is run from scripts/
INPUT_CSV = os.path.join(FEATURES_DATA_DIR, "final_features.csv")
MODELS_DIR = "models"
MODEL_FILENAME = "rf_model.joblib" # Changed filename
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
OUTPUT_DIR = "reports/figures" # Directory to save plots

print("--- RandomForest Model Interpretation using SHAP ---") # Changed

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH) # Use joblib to load
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Exiting.")
    exit()

# --- Load Data ---
print(f"Loading data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
    target = 'burglary_count'
    features_to_exclude = ['LSOA code', 'Month', target]
    X = df.drop(columns=features_to_exclude)
    print(f"Data loaded successfully. Features shape: {X.shape}")
except Exception as e:
    print(f"Error loading data: {e}. Exiting.")
    exit()

# --- Calculate SHAP Values ---
print("\nCalculating SHAP values (this may take a while)...")
# For RandomForest/sklearn trees, shap.TreeExplainer is still appropriate
# However, it can sometimes be slower than for XGBoost.
# Consider feature_perturbation="interventional" if needed, or sampling X.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X) 
print("SHAP values calculated.")

# --- Generate SHAP Plots --- 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Summary Plot (Global Feature Importance)
print("Generating SHAP summary plot...")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
summary_plot_path = os.path.join(OUTPUT_DIR, 'shap_summary_rf_bar.png') # Changed filename
plt.savefig(summary_plot_path, bbox_inches='tight')
plt.close()
print(f"Summary plot saved to {summary_plot_path}")

plt.figure()
shap.summary_plot(shap_values, X, show=False)
summary_dot_plot_path = os.path.join(OUTPUT_DIR, 'shap_summary_rf_dot.png') # Changed filename
plt.savefig(summary_dot_plot_path, bbox_inches='tight')
plt.close()
print(f"Summary dot plot saved to {summary_dot_plot_path}")

# 2. Dependence Plots (for top features)
print("\nGenerating SHAP dependence plots for top features...")
vals= np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

top_n = 5
for i, feature in enumerate(feature_importance['col_name'].head(top_n)):
    print(f"  Generating dependence plot for: {feature}")
    plt.figure()
    shap.dependence_plot(feature, shap_values, X, interaction_index=None, show=False)
    dependence_plot_path = os.path.join(OUTPUT_DIR, f'shap_dependence_rf_{i+1}_{feature}.png') # Changed filename
    plt.savefig(dependence_plot_path, bbox_inches='tight')
    plt.close()
    print(f"  Dependence plot saved to {dependence_plot_path}")

print("\nScript 05b_interpret_random_forest.py finished.") 
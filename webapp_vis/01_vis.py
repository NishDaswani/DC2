import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import xgboost as xgb
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import re
import matplotlib
import shutil
matplotlib.use('Agg')  # Use a non-interactive backend

# Ensure the output directory exists
os.makedirs('webapp_vis/vis/pres', exist_ok=True)

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
BLUE_PALETTE = sns.color_palette("Blues_r")
SEQUENTIAL_PALETTE = sns.color_palette("viridis", as_cmap=True)

def load_data():
    """Load the data files needed for visualizations"""
    print("Loading data files...")
    
    # Load the main data with features - use a path relative to the project root
    data_path = 'data/00_new/final_data_features.csv'
    print(f"Loading data from {data_path}")
    
    # Check if file exists before loading
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    
    # Load the trained XGBoost model
    with open('webapp_vis/vis/xgboost/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Load the trained SARIMA model
    with open('webapp_vis/sarima/best_sarima_model.pkl', 'rb') as f:
        sarima_model = pickle.load(f)
        
    return df, xgb_model, sarima_model

def create_pipeline_diagram():
    """Create a simple pipeline diagram (Slide 3)"""
    print("Creating pipeline diagram...")
    
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    ax.axis('off')
    
    # Define the pipeline steps
    steps = ["Data Collection", "Feature Engineering", "Modeling", "Forecast", "Allocation"]
    
    # Create the boxes
    box_width = 1.5
    box_height = 1
    for i, step in enumerate(steps):
        x = i * 2.5
        rect = plt.Rectangle((x, 0), box_width, box_height, 
                           facecolor=BLUE_PALETTE[i % len(BLUE_PALETTE)], 
                           alpha=0.8, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + box_width/2, box_height/2, step, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add arrows between boxes
        if i < len(steps) - 1:
            ax.arrow(x + box_width + 0.1, box_height/2, 0.8, 0, 
                    head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    plt.xlim(-0.5, (len(steps) - 1) * 2.5 + box_width + 0.5)
    plt.ylim(-0.5, box_height + 0.5)
    plt.title("Burglary Prediction Pipeline", fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_monthly_burglary_plot(df):
    """Create a line plot of monthly burglary counts (Slide 5)"""
    print("Creating monthly burglary counts plot...")
    
    # Aggregate burglaries by month
    monthly_counts = df.groupby(pd.Grouper(key='Month', freq='ME'))['burglary_count'].sum().reset_index()
    
    plt.figure(figsize=(18, 6))
    plt.plot(monthly_counts['Month'], monthly_counts['burglary_count'], 
             marker='o', linestyle='-', color='#1f77b4', 
             linewidth=1.5, markersize=5)
    
    # Add reference lines for important events (e.g., COVID lockdowns)
    plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', alpha=0.7)
    plt.text(pd.to_datetime('2020-03-15'), monthly_counts['burglary_count'].max() * 0.9, 
             'COVID-19 Lockdown', rotation=90, color='red')
    
    plt.title('Monthly Burglary Counts Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Burglary Count', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Format the x-axis to show fewer dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('webapp_vis/vis/pres/monthly_burglary_counts.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_burglary_heatmap(df):
    """Create a heatmap of burglary rates by LSOA (Slide 5)"""
    print("Creating burglary rate heatmap...")
    
    # Calculate average burglary rate per LSOA
    lsoa_avg = df.groupby('LSOA11CD')['burglary_count'].mean().reset_index()
    lsoa_avg = lsoa_avg.sort_values('burglary_count', ascending=False)
    
    # Take top 50 LSOAs for better visualization
    top_lsoas = lsoa_avg.head(50)
    
    plt.figure(figsize=(12, 10))
    
    # Create a pivot table for the heatmap
    # We'll use a subset of months for clarity
    recent_data = df[df['Month'] > pd.to_datetime('2021-01-01')]
    pivot_data = recent_data[recent_data['LSOA11CD'].isin(top_lsoas['LSOA11CD'])]
    
    # Create pivot table: LSOA vs Month with burglary count as values
    pivot = pivot_data.pivot_table(
        index='LSOA11CD', 
        columns=pd.Grouper(key='Month', freq='ME'),
        values='burglary_count',
        aggfunc='mean'
    )
    
    # Plot heatmap
    ax = sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5, linecolor='lightgray')
    plt.title('Burglary Rates Across Top 50 LSOAs Over Time', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('LSOA Code', fontsize=14)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/burglary_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_matrix(df):
    """Create a correlation matrix of features (Slide 5)"""
    print("Creating correlation matrix...")
    
    # Select relevant numeric columns
    numeric_cols = ['burglary_count', 'Population', 'population_density', 
                    'claimant_rate', 'poi_count', 'IncScore']
    
    # Add some lag and rolling features if they exist
    for col in df.columns:
        if ('lag' in col or 'rolling' in col) and col not in numeric_cols:
            numeric_cols.append(col)
    
    # Ensure all selected columns exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=True, fmt=".2f", linewidths=0.5, square=True)
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_actual_vs_predicted_comparison():
    """Create actual vs predicted comparison for both models (Slide 6)"""
    print("Creating actual vs predicted comparison plot...")
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # XGBoost plot (left)
    try:
        # Copy the existing plot from XGBoost
        shutil.copy('webapp_vis/vis/xgboost/actual_vs_predicted.png', 
                   'webapp_vis/vis/pres/xgboost_actual_vs_predicted.png')
        
        # Create a simplified version for the comparison plot
        xgb_img = plt.imread('webapp_vis/vis/xgboost/actual_vs_predicted.png')
        ax1.imshow(xgb_img)
        ax1.axis('off')
        ax1.set_title('XGBoost: Actual vs Predicted', fontsize=16)
    except Exception as e:
        print(f"Error with XGBoost plot: {e}")
        ax1.text(0.5, 0.5, "XGBoost plot unavailable", 
                ha='center', va='center', fontsize=14)
        ax1.set_title('XGBoost: Actual vs Predicted', fontsize=16)
    
    # SARIMA plot (right)
    try:
        # Copy the existing plot from SARIMA
        shutil.copy('webapp_vis/sarima/actual_vs_predicted.png', 
                   'webapp_vis/vis/pres/sarima_actual_vs_predicted.png')
        
        # Create a simplified version for the comparison plot
        sarima_img = plt.imread('webapp_vis/sarima/actual_vs_predicted.png')
        ax2.imshow(sarima_img)
        ax2.axis('off')
        ax2.set_title('SARIMA: Actual vs Predicted', fontsize=16)
    except Exception as e:
        print(f"Error with SARIMA plot: {e}")
        ax2.text(0.5, 0.5, "SARIMA plot unavailable", 
                ha='center', va='center', fontsize=14)
        ax2.set_title('SARIMA: Actual vs Predicted', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/model_comparison_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_performance_chart():
    """Create a bar chart of model performance metrics (Slide 6)"""
    print("Creating model performance chart...")
    
    metrics_data = {}
    model_order = ['XGBoost', 'SARIMA', 'GBDT', 'Prophet']

    # --- XGBoost --- 
    try:
        with open('webapp_vis/vis/xgboost/model_metrics.md', 'r') as f:
            xgb_metrics_text = f.read()
        metrics_data['XGBoost'] = {
            'rmse': float(re.search(r'Average Validation RMSE: (\d+\.\d+)', xgb_metrics_text).group(1)),
            'mae': float(re.search(r'Average Validation MAE: (\d+\.\d+)', xgb_metrics_text).group(1)),
            'r2': float(re.search(r'Average Validation R²: (-?\d+\.\d+)', xgb_metrics_text).group(1))
        }
    except Exception as e:
        print(f"Error reading/parsing XGBoost metrics: {e}")
        metrics_data['XGBoost'] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

    # --- SARIMA --- 
    try:
        with open('webapp_vis/sarima/model_metrics.md', 'r') as f:
            sarima_metrics_text = f.read()
        metrics_data['SARIMA'] = {
            'rmse': float(re.search(r'Average RMSE: (\d+\.\d+)', sarima_metrics_text).group(1)),
            'mae': float(re.search(r'Average MAE: (\d+\.\d+)', sarima_metrics_text).group(1)),
            'r2': float(re.search(r'Average R²: (-?\d+\.\d+)', sarima_metrics_text).group(1))
        }
    except Exception as e:
        print(f"Error reading/parsing SARIMA metrics: {e}")
        metrics_data['SARIMA'] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

    # --- GBDT --- 
    try:
        with open('webapp_vis/vis/gbdt/model_metrics.md', 'r') as f:
            gbdt_metrics_text = f.read()
        metrics_data['GBDT'] = {
            'rmse': float(re.search(r'Average Validation RMSE: (\d+\.\d+)', gbdt_metrics_text).group(1)),
            'mae': float(re.search(r'Average Validation MAE: (\d+\.\d+)', gbdt_metrics_text).group(1)),
            'r2': float(re.search(r'Average Validation R²: (-?\d+\.\d+)', gbdt_metrics_text).group(1))
        }
    except Exception as e:
        print(f"Error reading/parsing GBDT metrics: {e}")
        metrics_data['GBDT'] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

    # --- Prophet --- 
    # Prophet script saves metrics as webapp_vis/prophet/model_metrics_prophet.md
    try:
        with open('webapp_vis/prophet/model_metrics_prophet.md', 'r') as f:
            prophet_metrics_text = f.read()
        metrics_data['Prophet'] = {
            'rmse': float(re.search(r'Average RMSE: (\d+\.\d+)', prophet_metrics_text).group(1)),
            'mae': float(re.search(r'Average MAE: (\d+\.\d+)', prophet_metrics_text).group(1)),
            'r2': float(re.search(r'Average R²: (-?\d+\.\d+)', prophet_metrics_text).group(1))
        }
    except Exception as e:
        print(f"Error reading/parsing Prophet metrics: {e}")
        metrics_data['Prophet'] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
    
    # Prepare data for plotting in the specified order
    models_to_plot = [m for m in model_order if m in metrics_data]
    rmse_values = [metrics_data[m]['rmse'] for m in models_to_plot]
    mae_values = [metrics_data[m]['mae'] for m in models_to_plot]
    r2_values = [metrics_data[m]['r2'] for m in models_to_plot]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models_to_plot))
    width = 0.35 # width of the bars
    
    rects1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', color='#1f77b4')
    rects2 = ax.bar(x + width/2, mae_values, width, label='MAE', color='#ff7f0e')
    
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Error Metric Value', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models_to_plot)
    ax.legend()
    
    # Add R² values as text above the RMSE bars
    for i, model_name in enumerate(models_to_plot):
        if not np.isnan(r2_values[i]) and not np.isnan(rmse_values[i]):
            ax.text(x[i] - width/2, rmse_values[i] + (0.02 * max(filter(lambda v: not np.isnan(v), rmse_values), default=1)), 
                    f'R²={r2_values[i]:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_chart(xgb_model):
    """Create a feature importance bar chart (Slide 6)"""
    print("Creating feature importance chart...")
    
    # Use the existing plot from XGBoost visualizations but with better formatting
    plt.figure(figsize=(12, 8))
    
    # Get feature importance
    importance = xgb_model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 features
    top_features = sorted_importance[:10]
    features = [item[0] for item in top_features]
    scores = [item[1] for item in top_features]
    
    # Create horizontal bar chart
    plt.barh(range(len(features)), scores, align='center', color=BLUE_PALETTE)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance (gain)', fontsize=14)
    plt.title('Top 10 Features by Importance (XGBoost)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_shap_summary_plot(xgb_model, df):
    """Create a SHAP summary plot (Slide 7)"""
    print("Creating SHAP summary plot...")
    
    # Use the existing SHAP summary plot with better formatting
    # First, copy the existing image as a backup
    shutil.copy('webapp_vis/vis/xgboost/shap_summary.png', 
               'webapp_vis/vis/pres/shap_summary_original.png')
    
    # Create a new SHAP summary plot with better formatting
    try:
        # Sample data for SHAP analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X_sample = df[numeric_cols].sample(min(1000, len(df)), random_state=42)
        
        # Drop target if it exists
        if 'target' in X_sample.columns:
            X_sample = X_sample.drop('target', axis=1)
        if 'burglary_count' in X_sample.columns:
            X_sample = X_sample.drop('burglary_count', axis=1)
        
        # Create DMatrix for XGBoost
        X_sample_dmatrix = xgb.DMatrix(X_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(shap_values, features=X_sample, max_display=10, show=False)
        plt.title('SHAP Feature Importance', fontsize=16)
        plt.tight_layout()
        plt.savefig('webapp_vis/vis/pres/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating SHAP summary plot: {e}")
        print("Using original SHAP summary plot.")

def create_deployment_map():
    """Create a buffer-based deployment map (Slide 7)"""
    print("Creating deployment map...")
    
    # Create a simplified visualization of a buffer-based deployment strategy
    plt.figure(figsize=(12, 10))
    
    # Create a base map (simplified)
    x = np.random.rand(100)
    y = np.random.rand(100)
    
    # Risk scores (higher is more risk)
    risk = np.random.rand(100)
    
    # Plot the base points (LSOAs)
    plt.scatter(x, y, c=risk, cmap='YlOrRd', s=100, alpha=0.7)
    plt.colorbar(label='Burglary Risk Score')
    
    # Add buffer zones around high-risk areas
    high_risk_idx = risk > 0.8
    for i in range(len(x)):
        if high_risk_idx[i]:
            circle = plt.Circle((x[i], y[i]), 0.05, color='red', fill=False, 
                               linewidth=2, alpha=0.7)
            plt.gca().add_patch(circle)
            
            # Add a larger buffer for medium deployment
            circle2 = plt.Circle((x[i], y[i]), 0.1, color='orange', fill=False, 
                                linewidth=1.5, alpha=0.5)
            plt.gca().add_patch(circle2)
    
    # Add legend
    high_risk = mpatches.Patch(color='red', alpha=0.7, label='High Priority Patrol')
    med_risk = mpatches.Patch(color='orange', alpha=0.5, label='Medium Priority Patrol')
    plt.legend(handles=[high_risk, med_risk], loc='upper right')
    
    plt.title('Buffer-Based Patrol Allocation Strategy', fontsize=16)
    plt.xlabel('Longitude (simplified)', fontsize=14)
    plt.ylabel('Latitude (simplified)', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Remove axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('webapp_vis/vis/pres/deployment_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dashboard_mockup(df):
    """Create a dashboard mockup (Slide 8)"""
    print("Creating dashboard mockup...")
    
    # Create a mockup of a dashboard with multiple components
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Map area (main component)
    ax_map = fig.add_subplot(gs[0, :2])
    ax_map.set_title('Predicted Burglary Risk by LSOA', fontsize=14)
    
    # Create a simplified map visualization
    x = np.random.rand(100)
    y = np.random.rand(100)
    risk = np.random.rand(100)
    
    ax_map.scatter(x, y, c=risk, cmap='YlOrRd', s=80, alpha=0.7)
    ax_map.set_xlabel('Longitude (simplified)')
    ax_map.set_ylabel('Latitude (simplified)')
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    
    # 2. Control panel
    ax_controls = fig.add_subplot(gs[0, 2])
    ax_controls.set_title('Controls', fontsize=14)
    ax_controls.axis('off')
    
    # Add mock dropdown controls
    controls_text = """
    Borough: [All London Boroughs ▼]
    
    Time Period: [Next Month ▼]
    
    Risk Threshold: [High ▼]
    
    View Mode: [Heat Map ▼]
    
    [Update Map]
    """
    ax_controls.text(0.1, 0.5, controls_text, fontsize=12, va='center')
    
    # 3. Statistics panel
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.set_title('Burglary Statistics', fontsize=14)
    ax_stats.axis('off')
    
    # Create some mock statistics
    stats_text = """
    Total Predicted Burglaries: 1,247
    
    High Risk Areas: 32 LSOAs
    
    Medium Risk Areas: 87 LSOAs
    
    Low Risk Areas: 311 LSOAs
    
    Model Confidence: 82%
    """
    ax_stats.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    # 4. Patrol allocation
    ax_patrol = fig.add_subplot(gs[1, 1])
    ax_patrol.set_title('Recommended Patrol Allocation', fontsize=14)
    ax_patrol.axis('off')
    
    # Create mock patrol allocation
    patrol_text = """
    High Risk Areas:
    • 40 officers (8 teams)
    • Daily patrols
    
    Medium Risk Areas:
    • 25 officers (5 teams)
    • 3x weekly patrols
    
    Low Risk Areas:
    • 10 officers (2 teams)
    • Weekly monitoring
    """
    ax_patrol.text(0.1, 0.5, patrol_text, fontsize=12, va='center')
    
    # 5. Trend chart
    ax_trend = fig.add_subplot(gs[1, 2])
    ax_trend.set_title('Burglary Trend', fontsize=14)
    
    # Create a simple trend line
    months = pd.date_range(start='2022-01-01', periods=12, freq='ME')
    values = np.random.normal(100, 15, 12).cumsum()
    ax_trend.plot(months, values, marker='o', linestyle='-', color='#1f77b4')
    ax_trend.set_xlabel('Month')
    ax_trend.set_ylabel('Burglary Count')
    ax_trend.tick_params(axis='x', rotation=45)
    
    plt.suptitle('London Burglary Prediction Dashboard', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('webapp_vis/vis/pres/dashboard_mockup.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all visualizations"""
    print("Starting visualization generation...")
    
    # Create directories if they don't exist
    os.makedirs('webapp_vis/vis/pres', exist_ok=True)
    
    # Load data and models
    df, xgb_model, sarima_model = load_data()
    
    # Generate all visualizations
    create_pipeline_diagram()
    create_monthly_burglary_plot(df)
    create_burglary_heatmap(df)
    create_correlation_matrix(df)
    create_actual_vs_predicted_comparison()
    create_model_performance_chart()
    create_feature_importance_chart(xgb_model)
    create_shap_summary_plot(xgb_model, df)
    create_deployment_map()
    create_dashboard_mockup(df)
    
    print("All visualizations have been generated and saved to webapp_vis/vis/pres/")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Create directories for saving results
os.makedirs('webapp_vis/sarima', exist_ok=True)
os.makedirs('webapp_vis/sarima/forecasts', exist_ok=True)
os.makedirs('webapp_vis/sarima/sliding_window', exist_ok=True)

def preprocess_data(df):
    """
    Preprocess the data for SARIMA modeling
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
    
    # Document the preprocessing steps
    with open('webapp_vis/sarima/preprocessing_steps.md', 'w') as f:
        f.write("# Data Preprocessing Steps for SARIMA Model\n\n")
        f.write(f"- Original data shape: {df.shape}\n")
        f.write("- Converted 'Month' column to datetime format\n")
        f.write("- Sorted data by LSOA and Month\n")
        f.write("- Created target column (next month's burglary count)\n")
        f.write("- Dropped rows with missing target values\n")
        f.write("- Filled missing values in lag and rolling features with 0\n")
        f.write("- Filled other missing numeric values with median\n")
    
    return df

def grid_search_sarima(train_data):
    """
    Perform grid search to find optimal SARIMA parameters
    """
    print("Performing grid search for optimal SARIMA parameters...")
    
    # Define the p, d, q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
    # Generate all different combinations of p, d, q triplets
    pdq = list(product(p, d, q))
    # Generate all different combinations of seasonal P, D, Q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(product(range(0, 2), range(0, 2), range(0, 2)))]
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    # Limit the number of combinations to try (for speed)
    max_combinations = 10
    combinations_to_try = min(max_combinations, len(pdq) * len(seasonal_pdq))
    
    print(f"Trying {combinations_to_try} parameter combinations...")
    
    count = 0
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            if count >= combinations_to_try:
                break
                
            count += 1
            try:
                model = SARIMAX(train_data,
                                order=param,
                                seasonal_order=seasonal_param,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = seasonal_param
                    
                print(f"SARIMA{param}x{seasonal_param} - AIC:{results.aic:.3f}")
            except:
                continue
    
    if best_params is None:
        # Default parameters if grid search fails
        best_params = (1, 1, 1)
        best_seasonal_params = (1, 1, 1, 12)
        
    print(f"Best SARIMA parameters: {best_params}x{best_seasonal_params}")
    return best_params, best_seasonal_params

def train_sarima_models(df):
    """
    Train SARIMA models for each LSOA using time series cross-validation
    """
    print("Training SARIMA models with time series cross-validation...")
    
    # Define the time series cross-validation parameters
    test_start_date = pd.to_datetime('2020-01-01')  # Start of test period
    forecast_horizon = 6  # Forecast 6 months ahead
    
    # Select top LSOAs by burglary count for analysis (for speed)
    top_lsoa_count = 5
    top_lsoas = df.groupby('LSOA11CD')['burglary_count'].sum().nlargest(top_lsoa_count).index.tolist()
    print(f"Training models for top {top_lsoa_count} LSOAs with highest burglary counts")
    
    # Initialize lists to store results
    all_y_true = []
    all_y_pred = []
    all_fold_indices = []
    all_lsoa_indices = []
    
    # Store metrics for each LSOA
    lsoa_metrics = []
    
    # Create a dictionary to store models
    models = {}
    
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each LSOA
    for lsoa_idx, lsoa in enumerate(top_lsoas):
        print(f"\nTraining model for LSOA: {lsoa}")
        
        # Filter data for this LSOA
        lsoa_data = df[df['LSOA11CD'] == lsoa].copy()
        
        # Sort by month
        lsoa_data = lsoa_data.sort_values('Month')
        
        # Use sliding window approach similar to test.py
        window_start_date = pd.to_datetime('2018-01-01')  # Initial window start
        window_end_date = lsoa_data['Month'].max()
        
        # Store results for this LSOA
        lsoa_results = []
        
        # Store data for sliding window visualization
        sliding_window_dates = []
        sliding_window_actuals = []
        sliding_window_forecasts = []
        sliding_window_starts = []
        sliding_window_ends = []
        
        while window_start_date + pd.DateOffset(months=forecast_horizon) <= window_end_date:
            window_test_end = window_start_date + pd.DateOffset(months=forecast_horizon)
            
            # Split into training and test sets based on date
            train_data = lsoa_data[lsoa_data['Month'] <= window_start_date]['burglary_count']
            test_data = lsoa_data[(lsoa_data['Month'] > window_start_date) & 
                                 (lsoa_data['Month'] <= window_test_end)].copy()
            
            if len(test_data) < forecast_horizon / 2:
                # Skip windows with insufficient test data
                window_start_date += pd.DateOffset(months=3)  # Move forward by 3 months
                continue
            
            print(f"  Window: {window_start_date.strftime('%Y-%m')} to {window_test_end.strftime('%Y-%m')}")
            print(f"  Train data: {len(train_data)} months")
            print(f"  Test data: {len(test_data)} months")
            
            if len(train_data) < 24:  # At least 2 years of training data for seasonal models
                print(f"  Warning: Not enough training data. Skipping window.")
                window_start_date += pd.DateOffset(months=3)
                continue
            
            try:
                # Find optimal parameters for this LSOA and window
                if lsoa not in models:
                    # Only perform grid search once per LSOA for efficiency
                    order, seasonal_order = grid_search_sarima(train_data)
                else:
                    # Use previously found parameters
                    order, seasonal_order = models[lsoa]['order'], models[lsoa]['seasonal_order']
                
                # Fit model with optimal parameters
                model = SARIMAX(train_data,
                               order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                
                model_fit = model.fit(disp=False)
                
                # Store the model if it's the first window
                if lsoa not in models:
                    models[lsoa] = {
                        'model': model_fit,
                        'order': order,
                        'seasonal_order': seasonal_order
                    }
                
                # Forecast test period
                forecast = model_fit.forecast(steps=len(test_data))
                forecast = np.maximum(0, forecast)  # Ensure non-negative
                
                # Calculate metrics
                actual_values = test_data['burglary_count'].values
                
                rmse = np.sqrt(mean_squared_error(actual_values[:len(forecast)], forecast[:len(actual_values)]))
                mae = mean_absolute_error(actual_values[:len(forecast)], forecast[:len(actual_values)])
                r2 = r2_score(actual_values[:len(forecast)], forecast[:len(actual_values)])
                
                print(f"  SARIMA Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
                # Store results for this window
                window_result = {
                    'window_start': window_start_date,
                    'window_end': window_test_end,
                    'dates': test_data['Month'].values[:len(forecast)],
                    'actual_values': actual_values[:len(forecast)],
                    'forecast_values': forecast[:len(actual_values)],
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                lsoa_results.append(window_result)
                
                # Store for sliding window visualization
                sliding_window_dates.extend(test_data['Month'].values[:len(forecast)])
                sliding_window_actuals.extend(actual_values[:len(forecast)])
                sliding_window_forecasts.extend(forecast[:len(actual_values)])
                sliding_window_starts.extend([window_start_date] * len(actual_values[:len(forecast)]))
                sliding_window_ends.extend([window_test_end] * len(actual_values[:len(forecast)]))
                
                # Store for overall evaluation
                all_y_true.extend(actual_values[:len(forecast)])
                all_y_pred.extend(forecast[:len(actual_values)])
                all_fold_indices.extend([len(lsoa_results)] * len(actual_values[:len(forecast)]))
                all_lsoa_indices.extend([lsoa_idx] * len(actual_values[:len(forecast)]))
                
                # Create forecast plot for this window
                forecast_filename = f'webapp_vis/sarima/forecasts/forecast_{lsoa}_{window_start_date.strftime("%Y%m")}_to_{window_test_end.strftime("%Y%m")}_{timestamp}.png'
                create_forecast_plot(
                    lsoa,
                    window_start_date,
                    window_test_end,
                    test_data['Month'].values[:len(forecast)],
                    actual_values[:len(forecast)],
                    forecast[:len(actual_values)],
                    forecast_filename
                )
                
            except Exception as e:
                print(f"  Error in modeling for window: {e}")
                
            # Move to next window (3-month step for sliding window)
            window_start_date += pd.DateOffset(months=3)
        
        # Create sliding window visualization for this LSOA
        if len(sliding_window_dates) > 0:
            sliding_window_df = pd.DataFrame({
                'date': sliding_window_dates,
                'actual': sliding_window_actuals,
                'forecast': sliding_window_forecasts,
                'window_start': sliding_window_starts,
                'window_end': sliding_window_ends
            })
            
            # Create enhanced sliding window visualization
            create_enhanced_sliding_window_plot(
                lsoa,
                sliding_window_df,
                f'webapp_vis/sarima/sliding_window/sliding_window_{lsoa}_{timestamp}.png'
            )
            
            # Also create a combined forecast plot showing all windows
            create_combined_forecast_plot(
                lsoa,
                lsoa_results,
                f'webapp_vis/sarima/sliding_window/combined_forecast_{lsoa}_{timestamp}.png'
            )
        
        # Calculate average metrics for this LSOA
        if lsoa_results:
            avg_rmse = np.mean([r['rmse'] for r in lsoa_results])
            avg_mae = np.mean([r['mae'] for r in lsoa_results])
            avg_r2 = np.mean([r['r2'] for r in lsoa_results])
            
            lsoa_metrics.append({
                'lsoa': lsoa,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'r2': avg_r2,
                'windows': len(lsoa_results)
            })
            
            print(f"  Average metrics across {len(lsoa_results)} windows:")
            print(f"  RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}")
    
    # Create overall actual vs predicted plot
    create_overall_plot(all_y_true, all_y_pred, all_fold_indices, all_lsoa_indices, top_lsoas)
    
    # Save metrics to markdown file
    save_metrics_to_markdown(lsoa_metrics)
    
    # Save the best model for each LSOA
    for lsoa, model_data in models.items():
        with open(f'webapp_vis/sarima/sarima_model_{lsoa}.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    # Save a representative model for visualization comparison
    if models:
        best_lsoa = None
        best_r2 = -float('inf')
        
        for metric in lsoa_metrics:
            if metric['r2'] > best_r2:
                best_r2 = metric['r2']
                best_lsoa = metric['lsoa']
        
        if best_lsoa and best_lsoa in models:
            with open(f'webapp_vis/sarima/best_sarima_model.pkl', 'wb') as f:
                pickle.dump(models[best_lsoa], f)
            
            # Save model info
            with open(f'webapp_vis/sarima/model_info.md', 'w') as f:
                f.write("# SARIMA Model Information\n\n")
                f.write(f"Best performing LSOA: {best_lsoa}\n")
                f.write(f"Model order: {models[best_lsoa]['order']}\n")
                f.write(f"Seasonal order: {models[best_lsoa]['seasonal_order']}\n")
    
    return all_y_true, all_y_pred, models

def create_forecast_plot(lsoa, start_date, end_date, dates, actual, forecast, filename):
    """
    Create a plot comparing actual and forecasted values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, 'o-', label='Actual', color='blue')
    plt.plot(dates, forecast, 's--', label='SARIMA Forecast', color='red')
    
    plt.title(f'SARIMA Forecast for LSOA {lsoa}\n{start_date.strftime("%Y-%m")} to {end_date.strftime("%Y-%m")}')
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_enhanced_sliding_window_plot(lsoa, df, filename):
    """
    Create an enhanced sliding window visualization similar to the example
    """
    plt.figure(figsize=(14, 8))
    
    # Get unique window periods
    unique_windows = list(zip(df['window_start'].unique(), df['window_end'].unique()))
    
    # Create a colormap for different window shadings
    cmap_windows = plt.cm.get_cmap('viridis', max(1, len(unique_windows)))

    # Define fixed colors and styles for actual and forecast
    actual_color = 'green'
    forecast_color = 'blue'
    actual_style = 'o-'
    forecast_style = 's--'
    
    # Plot each window
    for i, (start, end) in enumerate(unique_windows):
        window_data = df[(df['window_start'] == start) & (df['window_end'] == end)]
        
        # Plot actual values
        plt.plot(window_data['date'], window_data['actual'], actual_style, 
                 color=actual_color, alpha=0.7, markersize=5,
                 label='Actual' if i == 0 else None) # Label only once for legend
        
        # Plot forecast values
        plt.plot(window_data['date'], window_data['forecast'], forecast_style, 
                 color=forecast_color, alpha=0.7, markersize=5,
                 label='Forecast' if i == 0 else None) # Label only once for legend
        
        # Add shaded area for this window using a distinct colormap
        min_date = window_data['date'].min()
        max_date = window_data['date'].max()
        plt.axvspan(min_date, max_date, 
                    alpha=0.1, color=cmap_windows(i), 
                    label=f'Window {i+1}: {start.strftime("%Y-%m")} to {end.strftime("%Y-%m")}' if i < 3 else None) # Label first few window spans
    
    # Add labels and legend
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Burglary Count', fontsize=14)
    plt.title(f'Sliding Window Evaluation for LSOA: {lsoa}', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Create a legend for Actual/Forecast and a few window spans
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Separate handles and labels for actual/forecast vs window spans
    actual_forecast_handles = []
    actual_forecast_labels = []
    window_span_handles = []
    window_span_labels = []

    for handle, label in zip(handles, labels):
        if label in ['Actual', 'Forecast']:
            if label not in actual_forecast_labels: # Ensure uniqueness
                actual_forecast_handles.append(handle)
                actual_forecast_labels.append(label)
        elif 'Window' in label:
            window_span_handles.append(handle)
            window_span_labels.append(label)

    # First legend for Actual/Forecast
    leg1 = plt.legend(actual_forecast_handles, actual_forecast_labels, loc='upper left', title='Data')
    plt.gca().add_artist(leg1) # Add the first legend manually

    # Second legend for Window Spans (if any are labeled)
    if window_span_handles:
        plt.legend(window_span_handles, window_span_labels, loc='upper right', title='Forecast Windows')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_forecast_plot(lsoa, window_results, filename):
    """
    Create a combined forecast plot showing all windows
    """
    plt.figure(figsize=(14, 8))
    
    # Sort window results by start date
    sorted_results = sorted(window_results, key=lambda x: x['window_start'])
    
    # Create a colormap for different windows
    cmap = plt.cm.get_cmap('viridis', max(1, len(sorted_results)))
    
    # Plot each window's forecast
    for i, result in enumerate(sorted_results):
        start_date = result['window_start']
        end_date = result['window_end']
        
        # Convert arrays to lists if they're pandas Series or numpy arrays
        dates = result['dates']
        if hasattr(dates, 'tolist'):
            dates = dates.tolist()
            
        actual = result['actual_values']
        if hasattr(actual, 'tolist'):
            actual = actual.tolist()
            
        forecast = result['forecast_values']
        if hasattr(forecast, 'tolist'):
            forecast = forecast.tolist()
        
        # Ensure all arrays have the same length
        min_len = min(len(dates), len(actual), len(forecast))
        dates = dates[:min_len]
        actual = actual[:min_len]
        forecast = forecast[:min_len]
        
        # Plot actual and forecast with window-specific color
        plt.plot(dates, actual, 'o-', color=cmap(i), alpha=0.5, markersize=4)
        plt.plot(dates, forecast, 's--', color=cmap(i), alpha=0.7, markersize=4)
        
        # Add window label at the middle of the window
        if len(dates) > 0:
            mid_idx = len(dates) // 2
            plt.text(dates[mid_idx], forecast[mid_idx], 
                    f'W{i+1}', fontsize=8, ha='center', va='bottom')
    
    # Add a single legend entry for actual and forecast
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='black', marker='o', linestyle='-'),
        Line2D([0], [0], color='black', marker='s', linestyle='--')
    ]
    plt.legend(custom_lines, ['Actual', 'Forecast'], loc='upper left')
    
    # Add title and labels
    plt.title(f'SARIMA Forecasts Across All Windows for LSOA: {lsoa}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Burglary Count', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_plot(y_true, y_pred, fold_indices, lsoa_indices, lsoa_codes):
    """
    Create an overall actual vs predicted plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create a DataFrame for easier plotting
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Fold': fold_indices,
        'LSOA_idx': lsoa_indices
    })
    
    # Create a scatter plot
    sns.scatterplot(x='Actual', y='Predicted', hue='LSOA_idx', 
                   palette='viridis', alpha=0.7, data=results_df)
    
    # Add the diagonal line (perfect predictions)
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    # Add labels and title
    plt.title('SARIMA: Actual vs Predicted Burglary Counts', fontsize=16)
    plt.xlabel('Actual Count', fontsize=14)
    plt.ylabel('Predicted Count', fontsize=14)
    
    # Create a legend mapping LSOA indices to codes
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [lsoa_codes[int(label)] for label in labels], 
              title='LSOA Code', title_fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('webapp_vis/sarima/actual_vs_predicted.png')
    plt.close()

def save_metrics_to_markdown(metrics):
    """
    Save model metrics to a markdown file
    """
    with open('webapp_vis/sarima/model_metrics.md', 'w') as f:
        f.write("# SARIMA Model Performance Metrics\n\n")
        
        # Overall metrics
        avg_rmse = np.mean([m['rmse'] for m in metrics])
        avg_mae = np.mean([m['mae'] for m in metrics])
        avg_r2 = np.mean([m['r2'] for m in metrics])
        
        f.write("## Overall Metrics\n\n")
        f.write(f"- Average RMSE: {avg_rmse:.4f}\n")
        f.write(f"- Average MAE: {avg_mae:.4f}\n")
        f.write(f"- Average R²: {avg_r2:.4f}\n\n")
        
        # Metrics by LSOA
        f.write("## Metrics by LSOA\n\n")
        f.write("| LSOA | RMSE | MAE | R² | Windows |\n")
        f.write("|------|------|-----|-----|--------|\n")
        
        for m in metrics:
            f.write(f"| {m['lsoa']} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {m['windows']} |\n")

def main():
    print("Starting SARIMA model training...")
    
    # Load the data
    data_path = 'data/00_new/final_data_features.csv' #=======================================================Change path if needed===============================================
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess the data
    processed_df = preprocess_data(df)
    
    # Train SARIMA models
    y_true, y_pred, models = train_sarima_models(processed_df)
    
    print("SARIMA model training completed.")
    print(f"Models and visualizations saved to webapp_vis/sarima/")

if __name__ == "__main__":
    main()

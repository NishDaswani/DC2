import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import json # For saving prophet model
from datetime import datetime
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Create directories for saving results
MODEL_NAME = 'prophet'
BASE_VIS_DIR = f'webapp_vis/{MODEL_NAME}'
FORECAST_DIR = f'{BASE_VIS_DIR}/forecasts'
SLIDING_WINDOW_DIR = f'{BASE_VIS_DIR}/sliding_window'

os.makedirs(BASE_VIS_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(SLIDING_WINDOW_DIR, exist_ok=True)

def preprocess_data_for_prophet(df_original):
    """
    Preprocess the data for Prophet modeling.
    Prophet expects columns 'ds' (datestamp) and 'y' (numeric value).
    """
    print("Preprocessing data for Prophet...")
    df = df_original.copy()
    
    # Convert Month to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Month']):
        df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    
    # Sort by LSOA and Month (important for consistent processing)
    df = df.sort_values(['LSOA11CD', 'Month'])
    
    # Prophet uses 'burglary_count' directly, no need to shift for 'target' initially
    # The sliding window will handle train/test splits.
    
    # Fill NaN values in 'burglary_count' if any (Prophet can handle some NaNs but explicit is better)
    # For simplicity, let's ensure 'burglary_count' is clean before passing to Prophet.
    # If 'burglary_count' can have NaNs that need imputation, it should be done here.
    # For this example, we assume 'burglary_count' is the value we want to predict.
    
    # Document the preprocessing steps
    with open(f'{BASE_VIS_DIR}/preprocessing_steps.md', 'w') as f:
        f.write(f"# Data Preprocessing Steps for {MODEL_NAME.upper()} Model\n\n")
        f.write(f"- Original data shape: {df.shape}\n")
        f.write("- Ensured 'Month' column is datetime format\n")
        f.write("- Sorted data by LSOA and Month\n")
        f.write("- Data will be split into 'ds' and 'y' for Prophet within the training loop for each LSOA.\n")
        f.write("- Selected exogenous regressors will be added to the model.\n")
        
    return df

def train_prophet_models(df):
    """
    Train Prophet models for each LSOA using time series cross-validation.
    """
    print(f"Training {MODEL_NAME.upper()} models with time series cross-validation and regressors...")
    
    forecast_horizon = 6  # Forecast 6 months ahead
    min_train_months = 24 # Minimum data for Prophet to train effectively

    # Define the list of regressors to use
    regressor_cols_to_use = [
        'month_nr',
        'burglary_lag_1',
        'burglary_rolling_mean_3',
        'burglary_volatility_3',
        'claimant_rate',
        'population_density'
        # Add other features here, e.g.:
        # 'poi_count', 'IncScore', 'burglary_rolling_max_3', 'burglary_trend_3_12'
    ]
    print(f"Using the following regressors: {regressor_cols_to_use}")

    # Select top LSOAs by burglary count for analysis
    top_lsoa_count = 5 # For speed, adjust as needed
    # Ensure 'burglary_count' is numeric for sum()
    df['burglary_count'] = pd.to_numeric(df['burglary_count'], errors='coerce').fillna(0)
    top_lsoas = df.groupby('LSOA11CD')['burglary_count'].sum().nlargest(top_lsoa_count).index.tolist()
    print(f"Training models for top {top_lsoa_count} LSOAs: {top_lsoas}")
    
    all_y_true = []
    all_y_pred = []
    all_fold_indices = [] # To mimic SARIMA structure if needed for plots
    all_lsoa_indices = [] # To mimic SARIMA structure if needed for plots
    
    lsoa_metrics_summary = []
    models_store = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for lsoa_idx, lsoa_code in enumerate(tqdm(top_lsoas, desc="Processing LSOAs")):
        print(f"\nTraining model for LSOA: {lsoa_code}")
        
        lsoa_data_full = df[df['LSOA11CD'] == lsoa_code].copy()
        lsoa_data_full = lsoa_data_full.sort_values('Month')
        
        # Prepare data for Prophet: rename columns and include regressors
        prophet_df_cols = ['Month', 'burglary_count'] + regressor_cols_to_use
        # Ensure all selected regressor columns exist in lsoa_data_full, otherwise handle missing ones
        missing_regressors = [col for col in regressor_cols_to_use if col not in lsoa_data_full.columns]
        if missing_regressors:
            print(f"  Warning: LSOA {lsoa_code} is missing regressor columns: {missing_regressors}. Skipping LSOA.")
            continue # or handle by filling/dropping, but skipping is safer for now

        lsoa_data_prophet_format = lsoa_data_full[prophet_df_cols].rename(
            columns={'Month': 'ds', 'burglary_count': 'y'}
        )

        # Ensure no NaNs in regressor columns (Prophet handles NaNs in y, but regressors should be clean for future)
        # Data cleaning in 07_seasonal_indicators.py should handle most, but double check here.
        for reg_col in regressor_cols_to_use:
            if lsoa_data_prophet_format[reg_col].isnull().any():
                # Fill with 0 as a simple strategy; could be median or ffill if more appropriate
                lsoa_data_prophet_format[reg_col] = lsoa_data_prophet_format[reg_col].fillna(0)
                print(f"  Filled NaNs in regressor '{reg_col}' with 0 for LSOA {lsoa_code}")

        # Sliding window approach
        # Define the overall period for this LSOA
        min_date_lsoa = lsoa_data_prophet_format['ds'].min()
        max_date_lsoa = lsoa_data_prophet_format['ds'].max()

        # Initial training window end date
        current_train_end_date = min_date_lsoa + pd.DateOffset(months=min_train_months -1) # Ensure at least min_train_months
        
        lsoa_window_results = []
        sliding_window_viz_data = {
            'date': [], 'actual': [], 'forecast': [], 
            'window_start': [], 'window_end': []
        }
        
        fold_count = 0
        while current_train_end_date + pd.DateOffset(months=forecast_horizon) <= max_date_lsoa:
            fold_count += 1
            test_window_end_date = current_train_end_date + pd.DateOffset(months=forecast_horizon)
            
            train_data = lsoa_data_prophet_format[lsoa_data_prophet_format['ds'] <= current_train_end_date]
            test_data = lsoa_data_prophet_format[
                (lsoa_data_prophet_format['ds'] > current_train_end_date) &
                (lsoa_data_prophet_format['ds'] <= test_window_end_date)
            ]

            if len(train_data) < min_train_months or len(test_data) == 0:
                print(f"  Skipping window: Train End {current_train_end_date.strftime('%Y-%m')}, Test End {test_window_end_date.strftime('%Y-%m')}. "
                      f"Train size: {len(train_data)}, Test size: {len(test_data)}")
                current_train_end_date += pd.DateOffset(months=3) # Slide window forward
                continue

            print(f"  Window {fold_count}: Train {train_data['ds'].min().strftime('%Y-%m')} to {train_data['ds'].max().strftime('%Y-%m')} ({len(train_data)} months), "
                  f"Test {test_data['ds'].min().strftime('%Y-%m')} to {test_data['ds'].max().strftime('%Y-%m')} ({len(test_data)} months)")

            try:
                # Initialize Prophet model
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                
                # Add regressors to the model
                for regressor in regressor_cols_to_use:
                    model.add_regressor(regressor)
                
                # Suppress Prophet's informational messages for cleaner logs during loops
                with HiddenPrints():
                    model.fit(train_data)
                
                # Create future dataframe for the forecast horizon
                future_df = model.make_future_dataframe(periods=len(test_data), freq='MS', include_history=False)
                
                # Add regressor values to the future_df
                # These values come from the test_data slice for the corresponding future dates
                # test_data already has 'ds' and regressor columns correctly named from lsoa_data_prophet_format
                
                # Create a temporary test_data slice that only includes ds and regressors for merging
                test_data_for_merge = test_data[['ds'] + regressor_cols_to_use]
                future_df = pd.merge(future_df, test_data_for_merge, on='ds', how='left')

                # Crucial: Ensure no NaNs in regressor columns of future_df after merge
                # This can happen if test_data didn't perfectly cover all future_df dates or had NaNs
                for reg_col in regressor_cols_to_use:
                    if future_df[reg_col].isnull().any():
                        # Fallback: Forward fill from last known value, then backfill for initial NaNs
                        future_df[reg_col] = future_df[reg_col].ffill().bfill()
                        # As a last resort, fill any remaining NaNs (e.g., if all values were NaN) with 0
                        future_df[reg_col] = future_df[reg_col].fillna(0)
                        print(f"  Filled NaNs in regressor '{reg_col}' in future_df for LSOA {lsoa_code} (post-merge)")

                # Predict
                forecast_df = model.predict(future_df)
                
                # Extract forecast, ensure non-negative
                # Align forecast with test data dates if necessary, Prophet's output `ds` should match
                predicted_values = np.maximum(0, forecast_df['yhat'].values[:len(test_data)])
                actual_values = test_data['y'].values
                dates_for_plot = test_data['ds'].values

                # Align lengths if forecast is longer due to make_future_dataframe behavior with freq
                min_len = min(len(actual_values), len(predicted_values))
                actual_values = actual_values[:min_len]
                predicted_values = predicted_values[:min_len]
                dates_for_plot = dates_for_plot[:min_len]
                
                if len(actual_values) == 0:
                    print("  Skipping window due to no actual values after alignment.")
                    current_train_end_date += pd.DateOffset(months=3)
                    continue

                rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
                mae = mean_absolute_error(actual_values, predicted_values)
                r2 = r2_score(actual_values, predicted_values)
                
                print(f"  {MODEL_NAME.upper()} Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
                window_result_data = {
                    'window_start': train_data['ds'].max(), # Start of test is after train_end
                    'window_end': test_data['ds'].max(),
                    'dates': dates_for_plot,
                    'actual_values': actual_values,
                    'forecast_values': predicted_values,
                    'rmse': rmse, 'mae': mae, 'r2': r2
                }
                lsoa_window_results.append(window_result_data)
                
                sliding_window_viz_data['date'].extend(dates_for_plot)
                sliding_window_viz_data['actual'].extend(actual_values)
                sliding_window_viz_data['forecast'].extend(predicted_values)
                sliding_window_viz_data['window_start'].extend([train_data['ds'].max()] * len(dates_for_plot))
                sliding_window_viz_data['window_end'].extend([test_data['ds'].max()] * len(dates_for_plot))
                
                all_y_true.extend(actual_values)
                all_y_pred.extend(predicted_values)
                all_fold_indices.extend([fold_count] * len(actual_values))
                all_lsoa_indices.extend([lsoa_idx] * len(actual_values))

                # Plot for this specific window
                forecast_plot_filename = f'{FORECAST_DIR}/forecast_{lsoa_code}_{current_train_end_date.strftime("%Y%m")}_{timestamp}.png'
                create_forecast_plot(
                    lsoa_code, train_data['ds'].max(), test_data['ds'].max(),
                    dates_for_plot, actual_values, predicted_values,
                    forecast_plot_filename, MODEL_NAME
                )

                # Store the last model for this LSOA (or best performing one based on some criteria)
                # For simplicity, storing the model from the last successfully trained window
                if lsoa_code not in models_store or test_window_end_date > models_store[lsoa_code]['last_window_end']:
                    models_store[lsoa_code] = {
                        'model_json': model_to_json(model),
                        'last_window_end': test_window_end_date,
                        'lsoa_name': lsoa_code, # Store LSOA name for model info
                        'regressors': regressor_cols_to_use
                    }

            except Exception as e:
                print(f"  Error training {MODEL_NAME.upper()} for LSOA {lsoa_code}, window ending {test_window_end_date.strftime('%Y-%m')}: {e}")
            
            current_train_end_date += pd.DateOffset(months=3) # Slide window forward by 3 months

        # After all windows for an LSOA
        if lsoa_window_results:
            avg_rmse = np.mean([r['rmse'] for r in lsoa_window_results])
            avg_mae = np.mean([r['mae'] for r in lsoa_window_results])
            avg_r2 = np.mean([r['r2'] for r in lsoa_window_results])
            lsoa_metrics_summary.append({
                'lsoa': lsoa_code, 'rmse': avg_rmse, 'mae': avg_mae, 'r2': avg_r2,
                'windows': len(lsoa_window_results)
            })
            print(f"  LSOA {lsoa_code} - Average Metrics over {len(lsoa_window_results)} windows: RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}")

            # Create sliding window visualization for this LSOA
            if len(sliding_window_viz_data['date']) > 0:
                sliding_df = pd.DataFrame(sliding_window_viz_data)
                # Ensure dates are datetime objects for plotting
                sliding_df['date'] = pd.to_datetime(sliding_df['date'])
                sliding_df['window_start'] = pd.to_datetime(sliding_df['window_start'])
                sliding_df['window_end'] = pd.to_datetime(sliding_df['window_end'])

                create_enhanced_sliding_window_plot(
                    lsoa_code, sliding_df,
                    f'{SLIDING_WINDOW_DIR}/sliding_window_{lsoa_code}_{timestamp}.png', MODEL_NAME
                )
                create_combined_forecast_plot(
                    lsoa_code, lsoa_window_results,
                    f'{SLIDING_WINDOW_DIR}/combined_forecast_{lsoa_code}_{timestamp}.png', MODEL_NAME
                )
    
    # After all LSOAs
    if all_y_true and all_y_pred:
        create_overall_plot(all_y_true, all_y_pred, all_fold_indices, all_lsoa_indices, top_lsoas, MODEL_NAME)
    
    save_metrics_to_markdown(lsoa_metrics_summary, MODEL_NAME)
    
    # Save models
    for lsoa_c, model_data in models_store.items():
        with open(f'{BASE_VIS_DIR}/{MODEL_NAME}_model_{lsoa_c}.json', 'w') as f_out:
            f_out.write(model_data['model_json'])
    print(f"Saved {len(models_store)} {MODEL_NAME.upper()} models to {BASE_VIS_DIR}")

    # Save a representative model for visualization comparison (e.g., best R2)
    if models_store and lsoa_metrics_summary:
        best_lsoa_metric = max(lsoa_metrics_summary, key=lambda x: x['r2'], default=None)
        if best_lsoa_metric and best_lsoa_metric['lsoa'] in models_store:
            best_lsoa_code = best_lsoa_metric['lsoa']
            with open(f'{BASE_VIS_DIR}/best_{MODEL_NAME}_model.json', 'w') as f_out:
                f_out.write(models_store[best_lsoa_code]['model_json'])
            
            with open(f'{BASE_VIS_DIR}/model_info.md', 'w') as f:
                f.write(f"# {MODEL_NAME.upper()} Model Information\n\n")
                f.write(f"Best performing LSOA based on R²: {best_lsoa_code}\n")
                f.write(f"Regressors used: {', '.join(models_store[best_lsoa_code]['regressors'])}\n") # List regressors
                f.write(f"Model saved from window ending: {models_store[best_lsoa_code]['last_window_end'].strftime('%Y-%m-%d')}\n")
                f.write(f"Note: Prophet models are generally re-trained. This saved model is a snapshot.\n")

    return all_y_true, all_y_pred, models_store


class HiddenPrints:
    """Context manager to hide prints."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        if exc_type is not None: # Reraise exception if any occurred
            raise exc_val


def create_forecast_plot(lsoa, train_end_date, test_end_date, dates, actual, forecast, filename, model_name_str):
    """
    Create a plot comparing actual and forecasted values for a window.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, 'o-', label='Actual', color='blue', markersize=5)
    plt.plot(dates, forecast, 's--', label=f'{model_name_str.upper()} Forecast', color='red', markersize=5)
    
    title_str = f'{model_name_str.upper()} Forecast for LSOA {lsoa}\n'
    title_str += f'Train End: {train_end_date.strftime("%Y-%m")}, Test: {pd.to_datetime(dates[0]).strftime("%Y-%m")} to {test_end_date.strftime("%Y-%m")}'
    plt.title(title_str)
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=150) # Lower DPI for faster saving during loops
    plt.close()

def create_enhanced_sliding_window_plot(lsoa, df_plot, filename, model_name_str):
    """
    Create an enhanced sliding window visualization.
    df_plot needs 'date', 'actual', 'forecast', 'window_start', 'window_end'
    """
    plt.figure(figsize=(15, 8))
    
    actual_color = 'blue'
    forecast_color = 'red'
    actual_style = 'o-'
    forecast_style = 's--'

    # Plot actual and forecast lines across all concatenated windows
    plt.plot(df_plot['date'], df_plot['actual'], actual_style, 
             color=actual_color, alpha=0.7, markersize=4, label='Actual')
    plt.plot(df_plot['date'], df_plot['forecast'], forecast_style, 
             color=forecast_color, alpha=0.7, markersize=4, label=f'{model_name_str.upper()} Forecast')

    # Shade window areas
    unique_windows = df_plot[['window_start', 'window_end']].drop_duplicates().sort_values(by='window_start')
    cmap_windows = plt.cm.get_cmap('coolwarm', max(1, len(unique_windows)))

    for i, (_, row) in enumerate(unique_windows.iterrows()):
        # For sliding window plot, window_start from df_plot is the training end date.
        # The actual forecast period starts after this.
        # Find the corresponding actual forecast dates for this span.
        forecast_dates_in_span = df_plot[
            (df_plot['window_start'] == row['window_start']) & 
            (df_plot['window_end'] == row['window_end'])
        ]['date']
        
        if not forecast_dates_in_span.empty:
            min_d = forecast_dates_in_span.min()
            max_d = forecast_dates_in_span.max()
            plt.axvspan(min_d, max_d, 
                        alpha=0.07, color=cmap_windows(i),
                        label=f'Test Window {i+1}' if i < 4 else None)
    
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Burglary Count', fontsize=14)
    plt.title(f'{model_name_str.upper()} Sliding Window Evaluation for LSOA: {lsoa}', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out duplicate labels before creating legend
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_forecast_plot(lsoa, window_results_list, filename, model_name_str):
    """
    Create a combined forecast plot showing all windows.
    window_results_list is a list of dicts, each with 'dates', 'actual_values', 'forecast_values'.
    """
    plt.figure(figsize=(15, 8))
    
    # Sort window results by the start date of their forecast period
    # 'dates'[0] would be the start of the forecast period
    sorted_results = sorted(window_results_list, key=lambda x: pd.to_datetime(x['dates'][0]) if len(x['dates']) > 0 else pd.Timestamp.max)
    
    cmap = plt.cm.get_cmap('viridis', max(1, len(sorted_results)))
    
    for i, result in enumerate(sorted_results):
        dates = pd.to_datetime(result['dates']) # Ensure datetime for plotting
        actual = result['actual_values']
        forecast = result['forecast_values']
        
        min_len = min(len(dates), len(actual), len(forecast))
        dates, actual, forecast = dates[:min_len], actual[:min_len], forecast[:min_len]
        
        if len(dates) == 0: continue

        plt.plot(dates, actual, 'o-', color=cmap(i), alpha=0.5, markersize=3, label='Actual' if i==0 else None)
        plt.plot(dates, forecast, 's--', color=cmap(i), alpha=0.7, markersize=3, label=f'{model_name_str.upper()} Forecast' if i==0 else None)
        
        if len(dates) > 0:
            mid_idx = len(dates) // 2
            # Only add text if there's space, avoid clutter
            if forecast[mid_idx] is not None : #and (i % max(1, len(sorted_results)//5) == 0): # Label every few
                 plt.text(dates[mid_idx], forecast[mid_idx] + 0.5, f'W{i+1}', 
                          fontsize=7, ha='center', va='bottom', color=cmap(i), alpha=0.9)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.title(f'{model_name_str.upper()} Forecasts Across Sliding Windows for LSOA: {lsoa}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Burglary Count', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_plot(y_true_list, y_pred_list, fold_indices_list, lsoa_indices_list, lsoa_codes_list, model_name_str):
    """
    Create an overall actual vs predicted plot.
    """
    if not y_true_list or not y_pred_list:
        print(f"Skipping overall plot for {model_name_str} due to empty prediction lists.")
        return

    plt.figure(figsize=(10, 8))
    
    results_df = pd.DataFrame({
        'Actual': y_true_list,
        'Predicted': y_pred_list,
        # 'Fold': fold_indices_list, # Less relevant for this combined plot view
        'LSOA_idx': lsoa_indices_list
    })
    
    # Map LSOA index to LSOA code for legend
    lsoa_map = {i: code for i, code in enumerate(lsoa_codes_list)}
    results_df['LSOA Code'] = results_df['LSOA_idx'].map(lsoa_map)

    sns.scatterplot(x='Actual', y='Predicted', hue='LSOA Code', 
                   palette='viridis', alpha=0.7, data=results_df)
    
    max_val = max(max(y_true_list, default=0), max(y_pred_list, default=0))
    min_val = min(min(y_true_list, default=0), min(y_pred_list, default=0))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) # Perfect prediction line
    
    plt.title(f'{model_name_str.upper()}: Actual vs Predicted Burglary Counts (All LSOAs, All Windows)', fontsize=14)
    plt.xlabel('Actual Count', fontsize=12)
    plt.ylabel('Predicted Count', fontsize=12)
    plt.legend(title='LSOA Code', title_fontsize='10', fontsize='8')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{BASE_VIS_DIR}/actual_vs_predicted_{model_name_str}.png', dpi=300)
    plt.close()

def save_metrics_to_markdown(metrics_list, model_name_str):
    """
    Save model metrics to a markdown file.
    """
    if not metrics_list:
        print(f"No metrics to save for {model_name_str}.")
        return

    with open(f'{BASE_VIS_DIR}/model_metrics_{model_name_str}.md', 'w') as f:
        f.write(f"# {model_name_str.upper()} Model Performance Metrics\n\n")
        
        # Overall average metrics
        if metrics_list:
            avg_rmse_overall = np.mean([m['rmse'] for m in metrics_list if m['rmse'] is not None])
            avg_mae_overall = np.mean([m['mae'] for m in metrics_list if m['mae'] is not None])
            avg_r2_overall = np.mean([m['r2'] for m in metrics_list if m['r2'] is not None])
            f.write("## Overall Average Metrics (across LSOAs)\n\n")
            f.write(f"- Average RMSE: {avg_rmse_overall:.4f}\n")
            f.write(f"- Average MAE: {avg_mae_overall:.4f}\n")
            f.write(f"- Average R²: {avg_r2_overall:.4f}\n\n")
        
        f.write("## Average Metrics by LSOA (across sliding windows)\n\n")
        f.write("| LSOA | Avg RMSE | Avg MAE | Avg R² | Num Windows |\n")
        f.write("|------|----------|---------|--------|-------------|\n")
        for m in metrics_list:
            f.write(f"| {m['lsoa']} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {m['windows']} |\n")
    print(f"Metrics saved to {BASE_VIS_DIR}/model_metrics_{model_name_str}.md")

def main():
    print(f"Starting {MODEL_NAME.upper()} model training script...")
    
    # Load the data (ensure this path is correct for your project)
    data_path = 'data/00_new/final_data_features.csv' # Assumes this file has 'Month', 'LSOA11CD', 'burglary_count'
    print(f"Loading data from {data_path}...")
    try:
        df_loaded = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please ensure the file exists.")
        return

    # Preprocess data for Prophet
    processed_df_prophet = preprocess_data_for_prophet(df_loaded)
    
    # Train Prophet models
    train_prophet_models(processed_df_prophet)
    
    print(f"{MODEL_NAME.upper()} model training completed.")
    print(f"Models, metrics, and visualizations saved to {BASE_VIS_DIR}/")

if __name__ == "__main__":
    # Need to import sys for HiddenPrints if it's used standalone in this script
    import sys 
    main()

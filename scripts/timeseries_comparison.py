'''
Compares ARIMA and Prophet models for forecasting monthly burglary counts per LSOA.

Loads aggregated burglary data, uses time series cross-validation, 
trains both ARIMA and Prophet models, evaluates and compares their performance.
Creates visualizations to compare the forecasting accuracy of both models.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import sys
import datetime
from tqdm import tqdm

# Suppress specific warning categories
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
# Define the project root assuming this script is in the 'scripts' directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')  # Changed to data from results
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
INPUT_FILE = os.path.join(INPUT_DIR, 'burglary_lsoa_monthly.csv')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
TARGET_VARIABLE = 'burglary_count'

# Time series cross-validation parameters
MIN_TRAIN_PERIODS = 6  # Minimum number of periods needed for training
FORECAST_PERIODS = 6   # Increased to 6 months for longer forecasts
STEP_SIZE = 3          # Step size between consecutive training sets
MAX_FOLDS = 4          # Maximum number of CV folds

# ARIMA parameters
ARIMA_ORDER = (1, 1, 1)  # (p, d, q) - can be tuned

# Extended forecast periods (for final prediction)
EXTENDED_FORECAST_PERIODS = 12  # Forecast a full year ahead

# --- Helper Functions for Visualization ---
def plot_forecast_comparison(lsoa_results, output_file):
    '''Create a plot comparing forecasts for a selected LSOA.'''
    plt.figure(figsize=(14, 8))
    
    # Plot actual values
    plt.plot(lsoa_results['dates'], lsoa_results['actual'], 
             'o-', label='Actual', color='black', alpha=0.7)
    
    # Plot ARIMA forecast
    plt.plot(lsoa_results['forecast_dates'], lsoa_results['arima_forecast'], 
             's-', label='ARIMA Forecast', color='blue', alpha=0.7)
    
    # Plot Prophet forecast
    plt.plot(lsoa_results['forecast_dates'], lsoa_results['prophet_forecast'], 
             '^-', label='Prophet Forecast', color='green', alpha=0.7)
    
    # Add shaded area for test period
    plt.axvspan(lsoa_results['forecast_dates'][0], lsoa_results['forecast_dates'][-1], 
                alpha=0.2, color='gray', label='Test Period')
    
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.title(f'Burglary Forecast Comparison for LSOA: {lsoa_results["lsoa"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_multifold_forecast(lsoa_results, output_file):
    '''Create a plot showing forecasts from multiple folds.'''
    plt.figure(figsize=(16, 10))
    
    # Plot actual values for the entire period
    plt.plot(lsoa_results['all_dates'], lsoa_results['all_actual'], 
             'o-', label='Actual', color='black', alpha=0.7)
    
    # Plot forecasts from each fold
    for fold_idx, fold_data in enumerate(lsoa_results['folds']):
        # Calculate relative position for clearer visualization
        fold_pos = fold_idx / (len(lsoa_results['folds']) + 1)
        
        # Plot ARIMA forecast for this fold
        plt.plot(fold_data['forecast_dates'], fold_data['arima_forecast'], 
                 's-', label=f'ARIMA Fold {fold_idx+1}' if fold_idx == 0 else "_nolegend_", 
                 color=plt.cm.Blues(0.5 + fold_pos/2), alpha=0.8)
        
        # Plot Prophet forecast for this fold
        plt.plot(fold_data['forecast_dates'], fold_data['prophet_forecast'], 
                 '^-', label=f'Prophet Fold {fold_idx+1}' if fold_idx == 0 else "_nolegend_", 
                 color=plt.cm.Greens(0.5 + fold_pos/2), alpha=0.8)
        
        # Add shaded area for each test period
        plt.axvspan(fold_data['forecast_dates'][0], fold_data['forecast_dates'][-1], 
                    alpha=0.1, color=plt.cm.Greys(0.3 + fold_pos/3), 
                    label=f'Test Period {fold_idx+1}' if fold_idx == 0 else "_nolegend_")
    
    # Add legend entries for the fold groups
    plt.plot([], [], 's-', color='blue', label='ARIMA Forecasts')
    plt.plot([], [], '^-', color='green', label='Prophet Forecasts')
    plt.fill_between([], [], [], color='lightgray', alpha=0.3, label='Test Periods')
    
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.title(f'Multi-Fold Forecast Comparison for LSOA: {lsoa_results["lsoa"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_extended_forecast(lsoa_results, output_file):
    '''Create a plot showing an extended forecast into the future.'''
    plt.figure(figsize=(14, 8))
    
    # Plot historical values
    plt.plot(lsoa_results['historical_dates'], lsoa_results['historical_values'], 
             'o-', label='Historical', color='black', alpha=0.7)
    
    # Plot ARIMA forecast
    plt.plot(lsoa_results['forecast_dates'], lsoa_results['arima_forecast'], 
             's-', label='ARIMA Forecast', color='blue', alpha=0.7)
    
    # Plot Prophet forecast
    plt.plot(lsoa_results['forecast_dates'], lsoa_results['prophet_forecast'], 
             '^-', label='Prophet Forecast', color='green', alpha=0.7)
    
    # Add shaded area for forecast period
    plt.axvspan(lsoa_results['forecast_dates'][0], lsoa_results['forecast_dates'][-1], 
                alpha=0.2, color='gray', label='Forecast Period')
    
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.title(f'Extended Forecast Comparison for LSOA: {lsoa_results["lsoa"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_model_comparison(metrics, output_file):
    '''Create bar charts comparing ARIMA and Prophet performance metrics.'''
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    models = ['ARIMA', 'Prophet']
    rmse_values = [metrics['arima_rmse'], metrics['prophet_rmse']]
    mae_values = [metrics['arima_mae'], metrics['prophet_mae']]
    
    # Plot RMSE comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, rmse_values, color=['blue', 'green'])
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison')
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot MAE comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, mae_values, color=['blue', 'green'])
    plt.ylabel('MAE')
    plt.title('MAE Comparison')
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.suptitle('Model Performance Comparison: ARIMA vs Prophet')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_cv_folds(time_series, min_train_periods, forecast_periods, step_size, max_folds):
    """Generate time series cross-validation folds."""
    total_periods = len(time_series)
    folds = []
    
    # Start with initial training set
    for i in range(0, min(max_folds*step_size, total_periods-forecast_periods-min_train_periods+1), step_size):
        train_end = min_train_periods + i
        test_start = train_end
        test_end = min(test_start + forecast_periods, total_periods)
        
        # Skip if we don't have enough data left for testing
        if test_end - test_start < forecast_periods / 2:
            continue
            
        folds.append({
            'train_start': 0,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
    return folds

# --- Main Script Logic ---
if __name__ == "__main__":
    print("--- Starting Time Series Model Comparison ---")
    
    # Get timestamp for unique plot filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load Data
    print(f"Loading burglary data from: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        # Convert 'Month' to datetime
        df['Month'] = pd.to_datetime(df['Month'])
    except FileNotFoundError:
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        print("Please run the preprocess.py script first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded data shape: {df.shape}")
    
    # Print data date range
    min_date = df['Month'].min()
    max_date = df['Month'].max()
    print(f"Date range in data: {min_date} to {max_date}")
    
    # 2. Prepare for Analysis
    # Let's select a sample of LSOAs to analyze (to avoid excessive runtime)
    top_lsoas = df.groupby('LSOA code')['burglary_count'].sum().nlargest(10).index.tolist()
    print(f"Selected {len(top_lsoas)} LSOAs with highest burglary counts for analysis")
    
    # Initialize results storage
    all_metrics = {
        'lsoa': [],
        'arima_rmse': [],
        'arima_mae': [],
        'prophet_rmse': [],
        'prophet_mae': []
    }
    
    all_forecasts = []
    all_multifold_forecasts = []
    all_extended_forecasts = []
    
    # 3. Process each LSOA
    for lsoa in tqdm(top_lsoas, desc="Processing LSOAs"):
        print(f"\nAnalyzing LSOA: {lsoa}")
        
        # Filter data for this LSOA
        lsoa_data = df[df['LSOA code'] == lsoa].copy()
        lsoa_data.sort_values('Month', inplace=True)
        
        # Skip LSOAs with too little data
        if len(lsoa_data) < MIN_TRAIN_PERIODS + FORECAST_PERIODS:
            print(f"  Warning: Not enough data for LSOA {lsoa}. Need at least {MIN_TRAIN_PERIODS + FORECAST_PERIODS} periods, but only have {len(lsoa_data)}. Skipping.")
            continue
        
        # Generate cross-validation folds
        folds = generate_cv_folds(
            lsoa_data, 
            MIN_TRAIN_PERIODS, 
            FORECAST_PERIODS, 
            STEP_SIZE, 
            MAX_FOLDS
        )
        
        if not folds:
            print(f"  Warning: Could not generate any CV folds for LSOA {lsoa}. Skipping.")
            continue
            
        print(f"  Generated {len(folds)} cross-validation folds")
        
        # Initialize metrics for this LSOA
        lsoa_arima_rmse = []
        lsoa_arima_mae = []
        lsoa_prophet_rmse = []
        lsoa_prophet_mae = []
        
        # Store the last fold's forecast for visualization
        last_fold_forecast = None
        
        # Store all fold forecasts for multifold visualization
        fold_forecasts = []
        
        # Process each fold
        for fold_idx, fold in enumerate(folds):
            print(f"  Processing fold {fold_idx+1}/{len(folds)}")
            
            # Split data for this fold
            train_data = lsoa_data.iloc[fold['train_start']:fold['train_end']].copy()
            test_data = lsoa_data.iloc[fold['test_start']:fold['test_end']].copy()
            
            test_dates = test_data['Month'].values
            actual_values = test_data[TARGET_VARIABLE].values
            
            # --- ARIMA Model ---
            try:
                print("    Training ARIMA model...")
                # Create time series
                ts_data = train_data.set_index('Month')[TARGET_VARIABLE]
                
                # Fit ARIMA model
                arima_model = ARIMA(ts_data, order=ARIMA_ORDER)
                arima_results = arima_model.fit()
                
                # Forecast
                arima_forecast = arima_results.forecast(steps=len(test_data))
                arima_forecast = np.maximum(0, arima_forecast)  # Ensure non-negative
                
                # Calculate metrics
                fold_arima_rmse = np.sqrt(mean_squared_error(actual_values, arima_forecast))
                fold_arima_mae = mean_absolute_error(actual_values, arima_forecast)
                
                # Store metrics
                lsoa_arima_rmse.append(fold_arima_rmse)
                lsoa_arima_mae.append(fold_arima_mae)
                
                print(f"    ARIMA RMSE: {fold_arima_rmse:.4f}, MAE: {fold_arima_mae:.4f}")
            except Exception as e:
                print(f"    Error in ARIMA modeling: {e}")
                # Use NaN for this fold
                lsoa_arima_rmse.append(np.nan)
                lsoa_arima_mae.append(np.nan)
                arima_forecast = np.full(len(test_data), np.nan)
            
            # --- Prophet Model ---
            try:
                print("    Training Prophet model...")
                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                prophet_train = train_data.rename(columns={'Month': 'ds', TARGET_VARIABLE: 'y'})
                
                # Initialize and fit Prophet model
                prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                prophet_model.fit(prophet_train)
                
                # Create future dataframe for forecasting
                future = prophet_model.make_future_dataframe(periods=len(test_data), freq='MS')
                forecast = prophet_model.predict(future)
                
                # Extract forecast for test period
                prophet_forecast = forecast.tail(len(test_data))['yhat'].values
                prophet_forecast = np.maximum(0, prophet_forecast)  # Ensure non-negative
                
                # Calculate metrics
                fold_prophet_rmse = np.sqrt(mean_squared_error(actual_values, prophet_forecast))
                fold_prophet_mae = mean_absolute_error(actual_values, prophet_forecast)
                
                # Store metrics
                lsoa_prophet_rmse.append(fold_prophet_rmse)
                lsoa_prophet_mae.append(fold_prophet_mae)
                
                print(f"    Prophet RMSE: {fold_prophet_rmse:.4f}, MAE: {fold_prophet_mae:.4f}")
            except Exception as e:
                print(f"    Error in Prophet modeling: {e}")
                # Use NaN for this fold
                lsoa_prophet_rmse.append(np.nan)
                lsoa_prophet_mae.append(np.nan)
                prophet_forecast = np.full(len(test_data), np.nan)
            
            # Store forecast data for this fold
            fold_forecasts.append({
                'forecast_dates': test_dates,
                'arima_forecast': arima_forecast,
                'prophet_forecast': prophet_forecast
            })
            
            # Store the last fold's data for visualization
            if fold_idx == len(folds) - 1:
                last_fold_forecast = {
                    'lsoa': lsoa,
                    'dates': lsoa_data['Month'].values,
                    'actual': lsoa_data[TARGET_VARIABLE].values,
                    'forecast_dates': test_dates,
                    'arima_forecast': arima_forecast,
                    'prophet_forecast': prophet_forecast
                }
        
        # Calculate average metrics across folds
        avg_arima_rmse = np.nanmean(lsoa_arima_rmse)
        avg_arima_mae = np.nanmean(lsoa_arima_mae)
        avg_prophet_rmse = np.nanmean(lsoa_prophet_rmse)
        avg_prophet_mae = np.nanmean(lsoa_prophet_mae)
        
        print(f"  Average metrics across {len(folds)} folds:")
        print(f"  ARIMA - RMSE: {avg_arima_rmse:.4f}, MAE: {avg_arima_mae:.4f}")
        print(f"  Prophet - RMSE: {avg_prophet_rmse:.4f}, MAE: {avg_prophet_mae:.4f}")
        
        # Store metrics
        all_metrics['lsoa'].append(lsoa)
        all_metrics['arima_rmse'].append(avg_arima_rmse)
        all_metrics['arima_mae'].append(avg_arima_mae)
        all_metrics['prophet_rmse'].append(avg_prophet_rmse)
        all_metrics['prophet_mae'].append(avg_prophet_mae)
        
        # Store forecast data for visualization
        if last_fold_forecast is not None:
            all_forecasts.append(last_fold_forecast)
        
        # Store data for multifold visualization
        if fold_forecasts:
            all_multifold_forecasts.append({
                'lsoa': lsoa,
                'all_dates': lsoa_data['Month'].values,
                'all_actual': lsoa_data[TARGET_VARIABLE].values,
                'folds': fold_forecasts
            })
        
        # 4. Create extended forecast for future periods
        try:
            print(f"\n  Creating extended {EXTENDED_FORECAST_PERIODS}-month forecast")
            # Use all available data for the final model
            extended_train_data = lsoa_data.copy()
            
            # Create time series for ARIMA
            ts_data_full = extended_train_data.set_index('Month')[TARGET_VARIABLE]
            
            # Fit ARIMA model
            arima_model_full = ARIMA(ts_data_full, order=ARIMA_ORDER)
            arima_results_full = arima_model_full.fit()
            
            # Generate future dates for forecasting
            last_date = extended_train_data['Month'].max()
            future_dates = pd.date_range(
                start=pd.Timestamp(last_date) + pd.DateOffset(months=1),
                periods=EXTENDED_FORECAST_PERIODS,
                freq='MS'
            )
            
            # Forecast with ARIMA
            arima_forecast_ext = arima_results_full.forecast(steps=EXTENDED_FORECAST_PERIODS)
            arima_forecast_ext = np.maximum(0, arima_forecast_ext)  # Ensure non-negative
            
            # Prepare data for Prophet
            prophet_train_full = extended_train_data.rename(columns={'Month': 'ds', TARGET_VARIABLE: 'y'})
            
            # Initialize and fit Prophet model
            prophet_model_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            prophet_model_full.fit(prophet_train_full)
            
            # Create future dataframe for forecasting
            future_df = prophet_model_full.make_future_dataframe(periods=EXTENDED_FORECAST_PERIODS, freq='MS')
            forecast_full = prophet_model_full.predict(future_df)
            
            # Extract forecast for future period
            prophet_forecast_ext = forecast_full.tail(EXTENDED_FORECAST_PERIODS)['yhat'].values
            prophet_forecast_ext = np.maximum(0, prophet_forecast_ext)  # Ensure non-negative
            
            # Store extended forecast
            all_extended_forecasts.append({
                'lsoa': lsoa,
                'historical_dates': extended_train_data['Month'].values,
                'historical_values': extended_train_data[TARGET_VARIABLE].values,
                'forecast_dates': future_dates,
                'arima_forecast': arima_forecast_ext,
                'prophet_forecast': prophet_forecast_ext
            })
            
            print(f"  Extended forecast created successfully")
        except Exception as e:
            print(f"  Error creating extended forecast: {e}")
    
    # 5. Create summary metrics dataframe
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n--- Overall Model Performance ---")
    if not metrics_df.empty:
        print("Average ARIMA RMSE:", metrics_df['arima_rmse'].mean())
        print("Average ARIMA MAE:", metrics_df['arima_mae'].mean())
        print("Average Prophet RMSE:", metrics_df['prophet_rmse'].mean())
        print("Average Prophet MAE:", metrics_df['prophet_mae'].mean())
        
        # Count wins for each model
        arima_rmse_wins = sum(metrics_df['arima_rmse'] < metrics_df['prophet_rmse'])
        prophet_rmse_wins = sum(metrics_df['arima_rmse'] > metrics_df['prophet_rmse'])
        ties = sum(metrics_df['arima_rmse'] == metrics_df['prophet_rmse'])
        
        print(f"\nRMSE Win count - ARIMA: {arima_rmse_wins}, Prophet: {prophet_rmse_wins}, Ties: {ties}")
    else:
        print("No valid metrics collected. Check the data and parameters.")
    
    # 6. Generate Visualizations
    print("\n--- Generating Visualizations ---")
    
    if not metrics_df.empty:
        # Overall metrics comparison
        overall_metrics = {
            'arima_rmse': metrics_df['arima_rmse'].mean(),
            'arima_mae': metrics_df['arima_mae'].mean(),
            'prophet_rmse': metrics_df['prophet_rmse'].mean(),
            'prophet_mae': metrics_df['prophet_mae'].mean()
        }
        
        plot_model_comparison(
            overall_metrics,
            os.path.join(PLOTS_DIR, f'model_comparison_{timestamp}.png')
        )
        
        # Generate single fold forecast visualizations
        for i, forecast_data in enumerate(all_forecasts[:3]):  # First 3 LSOAs
            plot_forecast_comparison(
                forecast_data,
                os.path.join(PLOTS_DIR, f'forecast_comparison_lsoa_{i}_{timestamp}.png')
            )
        
        # Generate multi-fold forecast visualizations
        for i, multifold_data in enumerate(all_multifold_forecasts[:3]):  # First 3 LSOAs
            plot_multifold_forecast(
                multifold_data,
                os.path.join(PLOTS_DIR, f'multifold_forecast_lsoa_{i}_{timestamp}.png')
            )
        
        # Generate extended forecast visualizations
        for i, extended_data in enumerate(all_extended_forecasts[:3]):  # First 3 LSOAs
            plot_extended_forecast(
                extended_data,
                os.path.join(PLOTS_DIR, f'extended_forecast_lsoa_{i}_{timestamp}.png')
            )
        
        # Save metrics to CSV
        metrics_file = os.path.join(OUTPUT_DIR, 'timeseries_comparison_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to: {metrics_file}")
    else:
        print("No visualizations generated due to lack of valid results.")
    
    print("\n--- Time Series Model Comparison Complete ---") 
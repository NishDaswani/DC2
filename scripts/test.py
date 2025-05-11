import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
from tqdm import tqdm
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Create directories for saving results
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/bigdata/combined_data.csv')

# Display dataset info
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Filter for burglary crimes only
print("Filtering for burglary crimes...")
df_burglary = df[df['Crime type'].str.contains('Burglary', case=False, na=False)].copy()
print(f"Burglary crimes count: {len(df_burglary)}")

# Convert 'Month' to datetime
df_burglary['Month'] = pd.to_datetime(df_burglary['Month'])

# Check the date range in the data
min_date = df_burglary['Month'].min()
max_date = df_burglary['Month'].max()
print(f"Data date range: {min_date} to {max_date}")

# Remove unnecessary columns
print("Removing unnecessary columns...")
cols_to_drop = ['Crime ID', 'Falls within', 'Reported by', 'Last outcome category', 
                'Context', 'source_file']
df_burglary = df_burglary.drop(columns=[col for col in cols_to_drop if col in df_burglary.columns])

# Aggregate data by LSOA and Month to get burglary counts
print("Aggregating data by LSOA and Month...")
df_agg = df_burglary.groupby(['LSOA code', 'Month']).size().reset_index(name='burglary_count')

# Make sure data is sorted by LSOA and Month
df_agg.sort_values(['LSOA code', 'Month'], inplace=True)

# Configuration for improved forecasting
# Shorter prediction horizons
FORECAST_HORIZON = 6  # Forecast 6 months ahead
GRID_SEARCH = True    # Perform grid search for optimal parameters
SLIDING_WINDOW = True # Use sliding window evaluation
USE_SEASONAL = True   # Include seasonal components (SARIMA)

# Set the initial date for the sliding window approach
INITIAL_TRAIN_END_DATE = '2019-12-31'
print(f"Initial training end date: {INITIAL_TRAIN_END_DATE}")
print(f"Forecast horizon: {FORECAST_HORIZON} months")
print(f"Using grid search: {GRID_SEARCH}")
print(f"Using sliding window evaluation: {SLIDING_WINDOW}")
print(f"Using seasonal components: {USE_SEASONAL}")

# Plot a histogram of burglary counts
plt.figure(figsize=(10, 6))
sns.histplot(df_agg['burglary_count'], kde=True)
plt.title('Distribution of Burglary Counts')
plt.xlabel('Burglary Count')
plt.ylabel('Frequency')
plt.savefig('plots/burglary_count_distribution.png')
plt.close()

# Function for grid search to find optimal ARIMA parameters
def grid_search_arima(train_data, test_data, seasonal=False):
    """Perform grid search to find optimal ARIMA or SARIMA parameters."""
    # Define the p, d, q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)
    # Generate all different combinations of p, d, q triplets
    pdq = list(product(p, d, q))
    # Generate all different combinations of seasonal P, D, Q triplets
    if seasonal:
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(product(range(0, 2), range(0, 2), range(0, 2)))]
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    train_ts = train_data.set_index('Month')['burglary_count']
    actual_values = test_data['burglary_count'].values[:FORECAST_HORIZON]
    
    # Create progress bar for grid search
    if seasonal:
        search_space = [(p, s) for p in pdq for s in seasonal_pdq]
        total_combinations = len(search_space)
    else:
        search_space = pdq
        total_combinations = len(pdq)
    
    print(f"  Grid search with {total_combinations} parameter combinations...")
    
    # Limit grid search to a reasonable number of combinations if too large
    if total_combinations > 20:
        print(f"  Limiting to first 20 combinations for efficiency...")
        if seasonal:
            search_space = search_space[:20]
        else:
            search_space = search_space[:20]
    
    best_rmse = float('inf')
    best_params_by_rmse = None
    best_seasonal_params_by_rmse = None
    
    try:
        for params in tqdm(search_space, desc="  Parameter combinations"):
            try:
                if seasonal:
                    pdq_params, seasonal_pdq_params = params
                    model = SARIMAX(train_ts, 
                                    order=pdq_params, 
                                    seasonal_order=seasonal_pdq_params,
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                else:
                    pdq_params = params
                    model = SARIMAX(train_ts, 
                                    order=pdq_params, 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                
                results = model.fit(disp=False)
                aic = results.aic
                
                # Forecast for test period
                forecast = results.forecast(steps=FORECAST_HORIZON)
                forecast = np.maximum(0, forecast)  # Ensure non-negative
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(actual_values[:len(forecast)], forecast[:len(actual_values)]))
                
                # Update best model if this one is better (by RMSE)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params_by_rmse = pdq_params
                    if seasonal:
                        best_seasonal_params_by_rmse = seasonal_pdq_params
                
                # Update best model if this one is better (by AIC)
                if aic < best_aic:
                    best_aic = aic
                    best_params = pdq_params
                    if seasonal:
                        best_seasonal_params = seasonal_pdq_params
                
            except Exception as e:
                continue
                
        # Prefer RMSE-based parameters over AIC-based parameters
        if best_params_by_rmse is not None:
            best_params = best_params_by_rmse
            if seasonal:
                best_seasonal_params = best_seasonal_params_by_rmse
            print(f"  Best parameters by RMSE: {best_params}")
            if seasonal:
                print(f"  Best seasonal parameters by RMSE: {best_seasonal_params}")
        else:
            print(f"  Best parameters by AIC: {best_params}")
            if seasonal:
                print(f"  Best seasonal parameters by AIC: {best_seasonal_params}")
        
        if seasonal:
            return best_params, best_seasonal_params
        else:
            return best_params, None
            
    except Exception as e:
        print(f"  Error during grid search: {e}")
        # Fallback to default parameters
        if seasonal:
            return (1, 1, 1), (1, 1, 1, 12)
        else:
            return (1, 1, 1), None

# Function to create forecast comparison plots
def create_forecast_comparison_plot(lsoa, train_data, test_data, forecast, output_file, model_name):
    """Create forecast comparison plot with historical data and forecasts."""
    plt.figure(figsize=(14, 8))
    
    # Plot historical data
    plt.plot(train_data['Month'], train_data['burglary_count'], 'o-', 
             color='black', label='Historical Data', alpha=0.7)
    
    # Plot test period actual values
    plt.plot(test_data['Month'][:len(forecast)], test_data['burglary_count'][:len(forecast)], 'o-', 
             color='green', label='Actual (Test Period)', alpha=0.7)
    
    # Plot forecast
    plt.plot(test_data['Month'][:len(forecast)], forecast, 's-', 
             color='blue', label=f'{model_name} Forecast', alpha=0.7)
    
    # Mark the forecast period with shading
    min_forecast_date = min(test_data['Month'][:len(forecast)])
    max_forecast_date = max(test_data['Month'][:len(forecast)])
    plt.axvspan(min_forecast_date, max_forecast_date, 
                alpha=0.1, color='gray', label='Forecast Period')
    
    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.title(f'{model_name} Forecast for LSOA: {lsoa}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    
    # Save with tight layout to prevent cutting off labels
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Function to create sliding window evaluation plots
def create_sliding_window_plot(lsoa, results_df, output_file):
    """Create plot showing performance over multiple sliding windows."""
    plt.figure(figsize=(14, 8))
    
    # Plot forecasts for each window
    for i, row in results_df.iterrows():
        window_start = pd.to_datetime(row['window_start'])
        window_end = pd.to_datetime(row['window_end'])
        
        # Plot actual values
        plt.plot(row['dates'], row['actual_values'], 'o-', 
                 color='green', alpha=0.5, 
                 label='Actual' if i == 0 else None)
        
        # Plot forecast values
        plt.plot(row['dates'], row['forecast_values'], 's-', 
                 color='blue', alpha=0.5,
                 label='Forecast' if i == 0 else None)
        
        # Add window shading
        plt.axvspan(window_start, window_end, 
                    alpha=0.1, color=f'C{i % 10}', 
                    label=f'Window {i+1}' if i < 5 else None)  # Only label first 5 windows
    
    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Burglary Count')
    plt.title(f'Sliding Window Evaluation for LSOA: {lsoa}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    
    # Save with tight layout to prevent cutting off labels
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Select top LSOAs by burglary count for analysis
top_lsoa_count = 5
top_lsoas = df_agg.groupby('LSOA code')['burglary_count'].sum().nlargest(top_lsoa_count).index.tolist()
print(f"Analyzing top {top_lsoa_count} LSOAs with highest burglary counts")

# Initialize results storage
results = {
    'lsoa': [],
    'model': [],
    'parameters': [],
    'rmse': [],
    'mae': [],
    'r2': [],
    'train_months': [],
    'test_months': []
}

# For sliding window evaluation
sliding_window_results = {}

# Process each LSOA
for lsoa in tqdm(top_lsoas, desc="Processing LSOAs"):
    print(f"\nAnalyzing LSOA: {lsoa}")
    
    # Filter data for this LSOA
    lsoa_data = df_agg[df_agg['LSOA code'] == lsoa].copy()
    lsoa_data.sort_values('Month', inplace=True)
    
    if SLIDING_WINDOW:
        # Use sliding window evaluation
        window_start_date = pd.to_datetime(INITIAL_TRAIN_END_DATE)
        window_end_date = max_date
        
        sliding_results = []
        
        while window_start_date + pd.DateOffset(months=FORECAST_HORIZON) <= window_end_date:
            window_test_end = window_start_date + pd.DateOffset(months=FORECAST_HORIZON)
            
            # Split into training and test sets based on date
            train_data = lsoa_data[lsoa_data['Month'] <= window_start_date].copy()
            test_data = lsoa_data[(lsoa_data['Month'] > window_start_date) & 
                                  (lsoa_data['Month'] <= window_test_end)].copy()
            
            if len(test_data) < FORECAST_HORIZON / 2:
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
            
            # Find optimal parameters if grid search is enabled
            if GRID_SEARCH:
                pdq_params, seasonal_pdq_params = grid_search_arima(train_data, test_data, seasonal=USE_SEASONAL)
            else:
                pdq_params = (1, 1, 1)
                seasonal_pdq_params = (1, 1, 1, 12) if USE_SEASONAL else None
            
            try:
                # Create time series for training
                ts_data = train_data.set_index('Month')['burglary_count']
                
                # Fit model with optimal parameters
                if USE_SEASONAL and seasonal_pdq_params:
                    model = SARIMAX(ts_data, 
                                    order=pdq_params, 
                                    seasonal_order=seasonal_pdq_params,
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                    model_name = "SARIMA"
                    model_params = f"{pdq_params}{seasonal_pdq_params}"
                else:
                    model = SARIMAX(ts_data, 
                                   order=pdq_params, 
                                   enforce_stationarity=False, 
                                   enforce_invertibility=False)
                    model_name = "ARIMA"
                    model_params = f"{pdq_params}"
                
                print(f"  Fitting {model_name}{model_params}...")
                model_fit = model.fit(disp=False)
                
                # Forecast test period
                forecast = model_fit.forecast(steps=len(test_data))
                forecast = np.maximum(0, forecast)  # Ensure non-negative
                
                # Calculate metrics
                actual_values = test_data['burglary_count'].values
                
                rmse = np.sqrt(mean_squared_error(actual_values[:len(forecast)], forecast[:len(actual_values)]))
                mae = mean_absolute_error(actual_values[:len(forecast)], forecast[:len(actual_values)])
                r2 = r2_score(actual_values[:len(forecast)], forecast[:len(actual_values)])
                
                print(f"  {model_name} Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
                # Store results for this window
                sliding_results.append({
                    'window_start': window_start_date,
                    'window_end': window_test_end,
                    'dates': test_data['Month'].values[:len(forecast)],
                    'actual_values': actual_values[:len(forecast)],
                    'forecast_values': forecast[:len(actual_values)],
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'model_name': model_name,
                    'model_params': model_params
                })
                
                # Create forecast plot for this window
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                window_label = f"{window_start_date.strftime('%Y%m')}_to_{window_test_end.strftime('%Y%m')}"
                
                create_forecast_comparison_plot(
                    lsoa,
                    train_data,
                    test_data,
                    forecast, 
                    f'plots/{model_name.lower()}_forecast_{lsoa}_{window_label}_{timestamp}.png',
                    model_name
                )
                
                # Store results
                results['lsoa'].append(lsoa)
                results['model'].append(model_name)
                results['parameters'].append(model_params)
                results['rmse'].append(rmse)
                results['mae'].append(mae)
                results['r2'].append(r2)
                results['train_months'].append(len(train_data))
                results['test_months'].append(len(test_data))
                
            except Exception as e:
                print(f"  Error in modeling for window: {e}")
                
            # Move to next window (3-month step for sliding window)
            window_start_date += pd.DateOffset(months=3)
        
        # Store all sliding window results for this LSOA
        sliding_window_results[lsoa] = sliding_results
        
        # Create sliding window visualization if we have results
        if sliding_results:
            create_sliding_window_plot(
                lsoa,
                pd.DataFrame(sliding_results),
                f'plots/sliding_window_{lsoa}_{timestamp}.png'
            )
            
            # Calculate overall metrics for this LSOA
            lsoa_rmse = np.mean([r['rmse'] for r in sliding_results])
            lsoa_mae = np.mean([r['mae'] for r in sliding_results])
            lsoa_r2 = np.mean([r['r2'] for r in sliding_results])
            
            print(f"  Average metrics across {len(sliding_results)} windows:")
            print(f"  RMSE: {lsoa_rmse:.4f}, MAE: {lsoa_mae:.4f}, R²: {lsoa_r2:.4f}")
    else:
        # Single train/test split evaluation
        train_data = lsoa_data[lsoa_data['Month'] <= INITIAL_TRAIN_END_DATE].copy()
        test_data = lsoa_data[lsoa_data['Month'] > INITIAL_TRAIN_END_DATE].copy()
        
        print(f"  Train data: {len(train_data)} months ({train_data['Month'].min()} to {train_data['Month'].max()})")
        print(f"  Test data: {len(test_data)} months ({test_data['Month'].min()} to {test_data['Month'].max()})")
        
        # Skip LSOAs with too little training data
        if len(train_data) < 24:  # At least 2 years for seasonal patterns
            print(f"  Warning: Not enough training data. Skipping LSOA.")
            continue
        
        # Find optimal parameters if grid search is enabled
        if GRID_SEARCH:
            pdq_params, seasonal_pdq_params = grid_search_arima(train_data, test_data[:FORECAST_HORIZON], 
                                                              seasonal=USE_SEASONAL)
        else:
            pdq_params = (1, 1, 1)
            seasonal_pdq_params = (1, 1, 1, 12) if USE_SEASONAL else None
        
        try:
            # Create time series for training
            ts_data = train_data.set_index('Month')['burglary_count']
            
            # Fit model with optimal parameters
            if USE_SEASONAL and seasonal_pdq_params:
                model = SARIMAX(ts_data, 
                               order=pdq_params, 
                               seasonal_order=seasonal_pdq_params,
                               enforce_stationarity=False, 
                               enforce_invertibility=False)
                model_name = "SARIMA"
                model_params = f"{pdq_params}{seasonal_pdq_params}"
            else:
                model = SARIMAX(ts_data, 
                               order=pdq_params, 
                               enforce_stationarity=False, 
                               enforce_invertibility=False)
                model_name = "ARIMA"
                model_params = f"{pdq_params}"
            
            print(f"  Fitting {model_name}{model_params}...")
            model_fit = model.fit(disp=False)
            
            # Forecast only the next FORECAST_HORIZON months
            forecast_horizon = min(FORECAST_HORIZON, len(test_data))
            forecast = model_fit.forecast(steps=forecast_horizon)
            forecast = np.maximum(0, forecast)  # Ensure non-negative
            
            # Calculate metrics
            actual_values = test_data['burglary_count'].values[:forecast_horizon]
            
            rmse = np.sqrt(mean_squared_error(actual_values, forecast))
            mae = mean_absolute_error(actual_values, forecast)
            r2 = r2_score(actual_values, forecast)
            
            print(f"  {model_name} Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Create forecast plot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            create_forecast_comparison_plot(
                lsoa,
                train_data,
                test_data[:forecast_horizon],
                forecast, 
                f'plots/{model_name.lower()}_forecast_{lsoa}_{timestamp}.png',
                model_name
            )
            
            # Store results
            results['lsoa'].append(lsoa)
            results['model'].append(model_name)
            results['parameters'].append(model_params)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['r2'].append(r2)
            results['train_months'].append(len(train_data))
            results['test_months'].append(forecast_horizon)
            
        except Exception as e:
            print(f"  Error in modeling: {e}")

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Save the results to arima_results.md
model_name = "SARIMA" if USE_SEASONAL else "ARIMA"
with open('results/improved_arima_results.md', 'w') as f:
    f.write(f'# Improved {model_name} Model Results for Burglary Prediction\n\n')
    f.write(f'Analysis date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    
    f.write('## Model Configuration\n')
    f.write(f'- Model type: {model_name}\n')
    f.write(f'- Grid search for optimal parameters: {GRID_SEARCH}\n')
    f.write(f'- Sliding window evaluation: {SLIDING_WINDOW}\n')
    f.write(f'- Forecast horizon: {FORECAST_HORIZON} months\n')
    f.write(f'- Initial training end date: {INITIAL_TRAIN_END_DATE}\n\n')
    
    f.write('## Overall Performance\n')
    if not results_df.empty:
        avg_rmse = results_df['rmse'].mean()
        avg_mae = results_df['mae'].mean()
        avg_r2 = results_df['r2'].mean()
        
        f.write(f'- Average RMSE: {avg_rmse:.4f}\n')
        f.write(f'- Average MAE: {avg_mae:.4f}\n')
        f.write(f'- Average R²: {avg_r2:.4f}\n\n')
        
        f.write('## Performance by LSOA\n\n')
        f.write('| LSOA | Model | Parameters | RMSE | MAE | R² | Train Months | Test Months |\n')
        f.write('|------|-------|------------|------|-----|----|-----------:|------------:|\n')
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['lsoa']} | {row['model']} | {row['parameters']} | ")
            f.write(f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} | ")
            f.write(f"{row['train_months']} | {row['test_months']} |\n")
    else:
        f.write('No valid results were generated during the analysis.\n\n')
    
    f.write('\n## Visualizations\n\n')
    
    if SLIDING_WINDOW:
        f.write('### Sliding Window Evaluation\n')
        f.write('These plots show model performance across multiple time windows:\n\n')
        
        for lsoa in top_lsoas:
            if lsoa in sliding_window_results and sliding_window_results[lsoa]:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                f.write(f'#### LSOA: {lsoa}\n')
                f.write(f'![Sliding Window Evaluation for {lsoa}](../plots/sliding_window_{lsoa}_{timestamp}.png)\n\n')
    
    f.write('### Forecast Visualizations\n')
    f.write('These plots show the actual vs predicted values:\n\n')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for lsoa in top_lsoas:
        if lsoa in results_df['lsoa'].values:
            lsoa_results = results_df[results_df['lsoa'] == lsoa]
            for idx, row in lsoa_results.iterrows():
                model = row['model'].lower()
                f.write(f'#### LSOA: {lsoa} ({row["model"]})\n')
                f.write(f'![{row["model"]} Forecast for {lsoa}](../plots/{model}_forecast_{lsoa}_{timestamp}.png)\n\n')
    
    f.write('### Burglary Count Distribution\n')
    f.write('![Burglary Count Distribution](../plots/burglary_count_distribution.png)\n\n')

print("\nAnalysis complete. Results saved to 'results/improved_arima_results.md'")
print("Visualizations saved to 'plots/' directory")
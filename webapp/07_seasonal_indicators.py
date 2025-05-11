import pandas as pd
import numpy as np
import os
from datetime import datetime

def main():
    print("Adding seasonal indicators and lag features to the data...")
    
    # Use relative paths from the project root
    data_path = 'data/00_new/final_data.csv'
    output_path = 'data/00_new/final_data_features.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Input file not found at {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert Month column to datetime for proper time series handling
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    
    # Sort by LSOA and Month to ensure proper time series order
    df = df.sort_values(['LSOA11CD', 'Month'])
    
    # Extract month number and create one-hot encoding
    df['month_nr'] = df['Month'].dt.month
    
    # Group by LSOA to create lag features for each area separately
    grouped = df.groupby('LSOA11CD')
    
    # Initialize new columns for lag features
    lag_periods = [1, 3, 12]
    for lag in lag_periods:
        df[f'burglary_lag_{lag}'] = np.nan
        
    # Initialize new columns for rolling statistics
    rolling_windows = [3, 6, 12]
    for window in rolling_windows:
        df[f'burglary_rolling_mean_{window}'] = np.nan
        if window == 3:
            df['burglary_volatility_3'] = np.nan 
        elif window == 12:
            df['burglary_volatility_12'] = np.nan
        else:
            df[f'burglary_rolling_std_{window}'] = np.nan

    # Initialize new columns for rolling max/min features
    rolling_max_min_windows = [3, 6, 12]
    for window in rolling_max_min_windows:
        df[f'burglary_rolling_max_{window}'] = np.nan
        df[f'burglary_rolling_min_{window}'] = np.nan

    # Initialize new columns for trend features
    df['burglary_trend_3_12'] = np.nan
    df['burglary_trend_6_12'] = np.nan
    
    # Calculate lag features and rolling statistics for each LSOA
    for lsoa, group in grouped:
        # Create lag features
        for lag in lag_periods:
            df.loc[group.index, f'burglary_lag_{lag}'] = group['burglary_count'].shift(lag)
        
        # Create rolling statistics (mean, std/volatility, max, min)
        for window in rolling_windows:
            df.loc[group.index, f'burglary_rolling_mean_{window}'] = group['burglary_count'].rolling(window=window, min_periods=1).mean()
            if window == 3:
                df.loc[group.index, 'burglary_volatility_3'] = group['burglary_count'].rolling(window=window, min_periods=1).std()
            elif window == 12:
                df.loc[group.index, 'burglary_volatility_12'] = group['burglary_count'].rolling(window=window, min_periods=1).std()
            elif window == 6:
                 if f'burglary_rolling_std_{window}' in df.columns:
                    df.loc[group.index, f'burglary_rolling_std_{window}'] = group['burglary_count'].rolling(window=window, min_periods=1).std()

        for window in rolling_max_min_windows:
            df.loc[group.index, f'burglary_rolling_max_{window}'] = group['burglary_count'].rolling(window=window, min_periods=1).max()
            df.loc[group.index, f'burglary_rolling_min_{window}'] = group['burglary_count'].rolling(window=window, min_periods=1).min()
    
    # Calculate trend features (after all rolling means are computed)
    if 'burglary_rolling_mean_3' in df.columns and 'burglary_rolling_mean_12' in df.columns:
        df['burglary_trend_3_12'] = df['burglary_rolling_mean_3'] - df['burglary_rolling_mean_12']
    if 'burglary_rolling_mean_6' in df.columns and 'burglary_rolling_mean_12' in df.columns:
        df['burglary_trend_6_12'] = df['burglary_rolling_mean_6'] - df['burglary_rolling_mean_12']

    # Convert Month back to string format for saving
    df['Month'] = df['Month'].dt.strftime('%Y-%m')
    
    # Save the enhanced dataset
    df.to_csv(output_path, index=False)
    
    print(f"Added features: month number, lag features, rolling means/max/min, volatilities, and trends")
    print(f"Data saved to {output_path}")
    print(f"Shape of final dataset: {df.shape}")

if __name__ == "__main__":
    main()

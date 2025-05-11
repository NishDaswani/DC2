import pandas as pd
import os

# Input file path
input_csv_path = "data/01_final_data/poi_added.csv"
# Output file path
output_csv_path = "data/01_final_data/processed_data.csv"

# --- 1. Read the CSV file ---
try:
    df = pd.read_csv(input_csv_path)
    print(f"Successfully read '{input_csv_path}'")
    print("Original DataFrame head:")
    print(df.head())
    print("\nOriginal DataFrame columns:")
    print(df.columns.tolist())

except FileNotFoundError:
    print(f"Error: Input file not found at '{input_csv_path}'.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- 2. Define desired column order ---
# IMPORTANT: Verify that 'LSOA11CD' is the correct column name in your poi_added.csv.
# If it's 'LSOA code' or something else, please adjust the list below.
_desired_column_order = [
    'Month', 
    'LSOA11CD',  # Verify this column name
    'LSOA name', 
    'Year', 
    'Population', 
    'area_km2', 
    'population_density', 
    'claimant_rate', 
    'poi_count', 
    'burglary_count'
]

# Check if all desired columns exist in the DataFrame
missing_cols = [col for col in _desired_column_order if col not in df.columns]
if missing_cols:
    print(f"\nError: The following desired columns are missing from the DataFrame: {missing_cols}")
    print(f"Please check the column names in '{input_csv_path}' and update the '_desired_column_order' list.")
    exit()

# --- 3. Reorder columns ---
try:
    df_reordered = df[_desired_column_order]
    print("\nDataFrame after reordering columns:")
    print(df_reordered.head())
except KeyError as e:
    print(f"\nError reordering columns: {e}. One or more specified columns not found.")
    print(f"Available columns: {df.columns.tolist()}")
    print("Please ensure the '_desired_column_order' list matches available column names.")
    exit()

# --- 4. Sort DataFrame by 'Month' ---
# Assuming 'Month' is in a format that sorts chronologically (e.g., 'YYYY-MM' or datetime)
# If 'Month' needs conversion to datetime first (e.g., from 'Month Year' string):
# df_reordered['Month'] = pd.to_datetime(df_reordered['Month'], format='%B %Y', errors='coerce') # Example

try:
    df_sorted = df_reordered.sort_values(by='Month', ascending=True)
    print("\nDataFrame after sorting by 'Month':")
    print(df_sorted.head())
except KeyError:
    print("\nError: 'Month' column not found for sorting. Please check column names.")
    exit()
except Exception as e:
    print(f"\nError during sorting: {e}")
    exit()

# --- 5. Save the processed DataFrame ---
try:
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df_sorted.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully processed and saved DataFrame to '{output_csv_path}'")
except Exception as e:
    print(f"\nError saving DataFrame to CSV: {e}")



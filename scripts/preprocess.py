'''
Loads raw street crime data from monthly folders, filters for London burglaries 
using geographic coordinates, aggregates counts by LSOA and month, 
and saves the processed data.

Assumes raw data is stored in subdirectories named YYYY-MM within 
'data/burglary_data/', and each subdirectory contains a '*-street.csv' file.
'''

import os
import glob
import pandas as pd
import sys

# --- Configuration ---
# Define the project root assuming this script is in the 'scripts' directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'burglary_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'burglary_lsoa_monthly.csv')

# Columns to load from the raw CSVs (adjust if needed)
# Added Longitude and Latitude for filtering
REQUIRED_COLUMNS = ['Month', 'LSOA code', 'Crime type', 'Longitude', 'Latitude']
# LONDON_LSOA_PREFIX = 'E09' # Removed - using coordinates instead
CRIME_FILTER = 'Burglary' # Ensure this matches the value in your 'Crime type' column

# London Bounding Box (Latitude: 51.3 to 51.7, Longitude: -0.5 to 0.2)
MIN_LAT, MAX_LAT = 51.3, 51.7
MIN_LON, MAX_LON = -0.5, 0.2

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"--- Starting Burglary Data Preprocessing --- ")
    print(f"Looking for raw data in subfolders of: {RAW_DATA_BASE_DIR}")

    # Find all YYYY-MM subdirectories
    try:
        month_folders = [f.path for f in os.scandir(RAW_DATA_BASE_DIR) if f.is_dir()]
    except FileNotFoundError:
        print(f"Error: Base directory not found: {RAW_DATA_BASE_DIR}", file=sys.stderr)
        print("Please ensure you have downloaded the data and placed it in the correct location.")
        sys.exit(1)

    if not month_folders:
        print(f"Error: No monthly subdirectories found in {RAW_DATA_BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(month_folders)} potential monthly data folders.")

    all_data_frames = []
    files_processed = 0
    files_skipped_missing_cols = 0
    files_skipped_no_street_csv = 0

    # Iterate through each month's folder
    for folder in sorted(month_folders):
        folder_name = os.path.basename(folder)
        print(f"Processing folder: {folder_name}...", end=' ')

        # Find the street crime CSV file within the folder
        street_csv_files = glob.glob(os.path.join(folder, '*-street.csv'))

        if not street_csv_files:
            print("No '*-street.csv' file found. Skipping.")
            files_skipped_no_street_csv += 1
            continue

        # Assuming only one relevant street file per folder
        file_path = street_csv_files[0]
        print(f"Reading {os.path.basename(file_path)}...", end=' ')

        try:
            # Load necessary columns
            df_month = pd.read_csv(file_path, usecols=REQUIRED_COLUMNS)
            all_data_frames.append(df_month)
            files_processed += 1
            print("OK.")
        except FileNotFoundError:
            print(f"Error: File not found {file_path}. Skipping.", file=sys.stderr)
        except ValueError as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}. Check REQUIRED_COLUMNS. Skipping.", file=sys.stderr)
            files_skipped_missing_cols += 1
        except Exception as e:
            print(f"Unexpected error reading {os.path.basename(file_path)}: {e}. Skipping.", file=sys.stderr)

    print(f"\n--- Data Loading Summary ---")
    print(f"Total monthly folders found: {len(month_folders)}")
    print(f"Street CSV files processed successfully: {files_processed}")
    print(f"Folders skipped (no street CSV): {files_skipped_no_street_csv}")
    print(f"Files skipped (missing required columns): {files_skipped_missing_cols}")

    if not all_data_frames:
        print("\nError: No data loaded successfully. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Combine, Filter, and Aggregate Data ---
    print("\n--- Combining and Filtering Data ---")
    df_combined = pd.concat(all_data_frames, ignore_index=True)
    print(f"Total raw records combined: {len(df_combined)}")

    # Drop rows with missing essential columns (LSOA code, Crime type, Month, Lon, Lat)
    initial_rows = len(df_combined)
    df_combined.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    rows_after_na_essential = len(df_combined)
    print(f"Records after dropping rows with missing essential values: {rows_after_na_essential} ({initial_rows - rows_after_na_essential} dropped)")

    # Convert coordinates to numeric, coercing errors to NaN
    df_combined['Longitude'] = pd.to_numeric(df_combined['Longitude'], errors='coerce')
    df_combined['Latitude'] = pd.to_numeric(df_combined['Latitude'], errors='coerce')

    # Drop rows where coordinate conversion failed
    rows_before_coord_na = len(df_combined)
    df_combined.dropna(subset=['Longitude', 'Latitude'], inplace=True)
    rows_after_coord_na = len(df_combined)
    if rows_before_coord_na > rows_after_coord_na:
        print(f"Records after dropping rows with invalid coordinates: {rows_after_coord_na} ({rows_before_coord_na - rows_after_coord_na} dropped)")

    # Filter by London coordinates
    df_london = df_combined[
        (df_combined['Latitude'] >= MIN_LAT) & (df_combined['Latitude'] <= MAX_LAT) &
        (df_combined['Longitude'] >= MIN_LON) & (df_combined['Longitude'] <= MAX_LON)
    ].copy()
    print(f"Records after filtering by London coordinates: {len(df_london)}")

    # Filter for the specified crime type
    df_london_burglary = df_london[df_london['Crime type'].str.contains(CRIME_FILTER, case=False, na=False)].copy()
    print(f"Records after filtering for '{CRIME_FILTER}': {len(df_london_burglary)}")

    # --- Removed LSOA prefix filter ---
    # df_burglary['LSOA code'] = df_burglary['LSOA code'].astype(str)
    # df_london_burglary = df_burglary[df_burglary['LSOA code'].str.startswith(LONDON_LSOA_PREFIX, na=False)].copy()
    # print(f"Records after filtering for London LSOAs (prefix '{LONDON_LSOA_PREFIX}'): {len(df_london_burglary)}")

    if df_london_burglary.empty:
        print("\nError: No London burglary records found after filtering. Check filters and data. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Aggregating Data by LSOA and Month ---")
    # Group by LSOA code and Month, then count the number of crimes in each group
    df_aggregated = df_london_burglary.groupby(['LSOA code', 'Month']).size().reset_index(name='burglary_count')

    print(f"Aggregated data shape: {df_aggregated.shape}")
    print("First 5 rows of aggregated data:")
    print(df_aggregated.head())

    total_burglaries = df_aggregated['burglary_count'].sum()
    print(f"\nTotal aggregated London burglary counts: {total_burglaries}")
    unique_lsoas = df_aggregated['LSOA code'].nunique()
    print(f"Number of unique London LSOAs with data: {unique_lsoas}")

    # --- Save Processed Data ---
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n--- Saving Aggregated Data to {OUTPUT_FILE} ---")
    try:
        df_aggregated.to_csv(OUTPUT_FILE, index=False)
        print("Data saved successfully.")
    except IOError as e:
        print(f"Error saving data to CSV: {e}", file=sys.stderr)

    print("\n--- Data Preprocessing Complete ---")

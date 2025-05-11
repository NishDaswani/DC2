import pandas as pd
import geopandas as gpd
import os

def merge_imd_scores():
    """
    Loads IMD scores from a shapefile, merges them with the processed burglary data,
    reorders columns, and saves the final dataset.
    """
    # --- Configuration: User needs to verify these paths ---
    # IMPORTANT: Replace with the actual path to your shapefile
    shapefile_path = "data/English IMD 2019/IMD_2019.shp"  # <--- USER MUST CHANGE THIS
    processed_data_csv_path = "data/00_new/processed_data.csv"
    output_csv_path = "data/00_new/final_data.csv"

    # --- 1. Load IMD scores from shapefile ---
    try:
        print(f"Reading shapefile from: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        print(f"Shapefile loaded successfully. Columns: {gdf.columns.tolist()}")

        # Ensure required columns exist in shapefile
        if 'lsoa11cd' not in gdf.columns or 'IncScore' not in gdf.columns:
            print(f"Error: Shapefile must contain 'lsoa11cd' and 'IncScore' columns.")
            print(f"Available columns in shapefile: {gdf.columns.tolist()}")
            return

        # Create a DataFrame with LSOA code and Income Score
        imd_df = gdf[['lsoa11cd', 'IncScore']].copy()
        imd_df.rename(columns={'lsoa11cd': 'LSOA11CD'}, inplace=True)
        print("\nIMD Scores DataFrame created (head):")
        print(imd_df.head())

    except FileNotFoundError:
        print(f"Error: Shapefile not found at '{shapefile_path}'. Please check the path and filename.")
        return
    except Exception as e:
        print(f"Error reading or processing shapefile: {e}")
        print("Please ensure you have 'geopandas' and its dependencies installed (e.g., fiona, pyproj, shapely).")
        return

    # --- 2. Load the processed data ---
    try:
        print(f"\nReading processed data from: {processed_data_csv_path}")
        main_df = pd.read_csv(processed_data_csv_path)
        print("Processed data loaded successfully (head):")
        print(main_df.head())
        if 'LSOA11CD' not in main_df.columns:
            print(f"Error: 'LSOA11CD' column not found in '{processed_data_csv_path}'.")
            return
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{processed_data_csv_path}'.")
        return
    except Exception as e:
        print(f"Error reading processed data CSV: {e}")
        return

    # --- 3. Merge IMD scores with the main DataFrame ---
    try:
        print("\nMerging IMD scores into the main DataFrame...")
        merged_df = pd.merge(main_df, imd_df, on='LSOA11CD', how='left')
        print("Merge complete. DataFrame after merge (head):")
        print(merged_df.head())

        # Check for rows that didn't get an IncScore
        if merged_df['IncScore'].isnull().any():
            print("\nWarning: Some rows did not find a matching 'IncScore'.")
            lsoas_without_incscore = merged_df[merged_df['IncScore'].isnull()]['LSOA11CD'].nunique()
            print(f"Number of unique LSOA11CDs without a matching IncScore: {lsoas_without_incscore}")

    except Exception as e:
        print(f"Error during merge operation: {e}")
        return

    # --- 4. Define desired column order and reorder ---
    desired_column_order = [
        'Month', 'LSOA11CD', 'LSOA Name', 'Year', 'Population', 'area_km2',
        'population_density', 'claimant_rate', 'poi_count', 'IncScore', 'burglary_count'
    ]

    # Check if all desired columns exist
    missing_cols = [col for col in desired_column_order if col not in merged_df.columns]
    if missing_cols:
        print(f"\nError: The following desired columns for the final output are missing: {missing_cols}")
        print(f"Available columns after merge: {merged_df.columns.tolist()}")
        return
    
    try:
        final_df = merged_df[desired_column_order]
        print("\nDataFrame after reordering columns (head):")
        print(final_df.head())
    except KeyError as e:
        print(f"\nError reordering columns: {e}. One or more specified columns not found in merged data.")
        print(f"Available columns: {merged_df.columns.tolist()}")
        return

    # --- 5. Save the final DataFrame ---
    # The 'Month' column should already be sorted from 'processed_data.csv'
    # and a left merge typically preserves the order of the left DataFrame's keys.
    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir: # Ensure directory exists if it's not the root
            os.makedirs(output_dir, exist_ok=True)
        
        final_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully created and saved final dataset to '{output_csv_path}'")
    except Exception as e:
        print(f"\nError saving final DataFrame to CSV: {e}")

if __name__ == "__main__":
    merge_imd_scores() 
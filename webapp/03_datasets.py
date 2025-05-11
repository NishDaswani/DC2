# I have a population dataset that uses LSOA 2021 codes and a shapefile or GeoJSON with LSOA 2011 boundaries. I also have a lookup table that maps LSOA 2021 codes to LSOA 2011 codes.
# I want to:
# Load the population dataset and rename the column to clearly identify it as the 2021 LSOA code.
# Load the lookup CSV and merge it with the population data so that each row also includes its corresponding LSOA 2011 code.
# Drop the LSOA 2021 code if it's no longer needed, and rename the 2011 code as the new standard identifier (LSOA code).
# Load the LSOA boundary file (2011 edition) as a GeoDataFrame and reproject it to British National Grid (EPSG:27700) so that I can accurately compute the area in square kilometers.
# Keep only the LSOA code and computed area.
# Merge the harmonized population data with this area data using the LSOA 2011 code as the key.
# Calculate the population density by dividing population by area (in km²).

import pandas as pd
import geopandas as gpd
import re
import os

def prepare_lsoa_base_data(final_processed_data_path: str) -> pd.DataFrame:
    """
    Loads the final processed data, filters for 2022, and selects necessary columns.
    """
    df = pd.read_csv(final_processed_data_path)
    df_2022 = df[df['Year'] == 2022].copy()
    # Assuming 'LSOA code' is the LSOA11CD and 'Population' is the 2022 population for these rows
    lsoa_base = df_2022[['LSOA code', 'Population', 'area_km2']].drop_duplicates(subset=['LSOA code'])
    lsoa_base.rename(columns={'Population': 'population_2022'}, inplace=True)
    return lsoa_base

def calculate_borough_growth_rates(projections_path: str, sheet_name: str) -> pd.DataFrame:
    # 1. Load the Excel sheet
    xls = pd.ExcelFile(projections_path)
    # Normalize sheet name casing
    if sheet_name not in xls.sheet_names:
        sheet_name = next((s for s in xls.sheet_names if s.lower() == sheet_name.lower()), sheet_name)
    gla = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 2. Convert _all_ column names to strings
    gla.columns = list(map(str, gla.columns))
    
    # 3. Identify columns that are four-digit years 2000–2099
    year_cols = [c for c in gla.columns if re.fullmatch(r'20\d{2}', c)]
    # We only need up to 2025 for projections
    year_cols = [c for c in year_cols if c in {'2022','2023','2024','2025'}]
    print("Using year columns:", year_cols)
    
    # 4. Aggregate across age & sex to get total borough population per year
    borough_totals = (
        gla
        .groupby('borough')[year_cols]
        .sum()
        .reset_index()
    )
    # Normalize borough names
    borough_totals['borough'] = borough_totals['borough'].str.strip().str.lower()
    
    # 5. Compute annual growth rates
    borough_totals['growth_2023'] = borough_totals['2023'].astype(float) / borough_totals['2022'] - 1
    borough_totals['growth_2024'] = borough_totals['2024'] / borough_totals['2023'] - 1
    borough_totals['growth_2025'] = borough_totals['2025'] / borough_totals['2024'] - 1
    
    return borough_totals[['borough','growth_2023','growth_2024','growth_2025']]

def load_lsoa_to_borough_lookup(lookup_path: str) -> pd.DataFrame:
    """
    Loads the LSOA to Borough lookup table and normalizes borough names.
    """
    lookup_df = pd.read_csv(lookup_path)
    # Select relevant columns and normalize LAD22NM for merging
    lookup_df = lookup_df[['LSOA11CD', 'LAD22NM']].copy()
    lookup_df['LAD22NM'] = lookup_df['LAD22NM'].str.strip().str.lower()
    lookup_df.rename(columns={'LAD22NM': 'borough'}, inplace=True)
    return lookup_df

def project_lsoa_populations_and_density(
    final_processed_data_path: str = "data/processed/final_processed_data_new.csv",
    projections_path: str = "data/pop_est/London Datastore Population Projections 10yr.xlsx",
    projections_sheet_name: str = "persons",
    lsoa_lookup_path: str = "data/pop_est/LSOA Best Fit Lookup 2011-2022.csv"
) -> pd.DataFrame:
    """
    Orchestrates the loading, merging, projection, and density calculation for LSOA populations.
    """
    lsoa_base = prepare_lsoa_base_data(final_processed_data_path)
    borough_growth_rates = calculate_borough_growth_rates(projections_path, projections_sheet_name)
    lsoa_to_borough = load_lsoa_to_borough_lookup(lsoa_lookup_path)

    # Merge LSOA base data with LSOA-to-Borough lookup
    merged_df = pd.merge(lsoa_base, lsoa_to_borough, left_on='LSOA code', right_on='LSOA11CD', how='left')

    # Merge with borough growth rates
    # Ensure 'borough' column exists and is correctly named after the first merge.
    # If 'LSOA11CD' is redundant after merge, it can be dropped.
    # merged_df.drop(columns=['LSOA11CD'], inplace=True, errors='ignore') # Optional: drop if not needed
    
    projected_df = pd.merge(merged_df, borough_growth_rates, on='borough', how='left')

    # Project LSOA populations
    projected_df['population_2023'] = projected_df['population_2022'] * (1 + projected_df['growth_2023'])
    projected_df['population_2024'] = projected_df['population_2023'] * (1 + projected_df['growth_2024'])
    projected_df['population_2025'] = projected_df['population_2024'] * (1 + projected_df['growth_2025'])

    # Calculate population densities
    projected_df['density_2022'] = projected_df['population_2022'] / projected_df['area_km2']
    projected_df['density_2023'] = projected_df['population_2023'] / projected_df['area_km2']
    projected_df['density_2024'] = projected_df['population_2024'] / projected_df['area_km2']
    projected_df['density_2025'] = projected_df['population_2025'] / projected_df['area_km2']
    
    # Select and reorder columns for clarity (optional)
    final_columns = [
        'LSOA code', 'borough', 'area_km2',
        'population_2022', 'population_2023', 'population_2024', 'population_2025',
        'density_2022', 'density_2023', 'density_2024', 'density_2025',
        'LSOA11CD' # keeping this if it's useful, or it can be dropped
    ]
    # Filter out any columns that might not exist if merges fail etc.
    final_columns = [col for col in final_columns if col in projected_df.columns]
    projected_df = projected_df[final_columns]

        # === CLEANING STEP START ===
    cleaned = projected_df.dropna(subset=['borough', 'area_km2'])
    cleaned = cleaned.dropna(subset=['population_2022'])
    cleaned = cleaned.dropna(subset=['population_2023'])   
    for year in ['2022','2023','2024','2025']:
        cleaned[f'density_{year}'] = cleaned[f'population_{year}'] / cleaned['area_km2']
    print("After cleaning:", cleaned.shape, "rows remain")
    print("Remaining NaNs per column:\n", cleaned.isna().sum())
    # === CLEANING STEP END ===

    return cleaned

def merge_projected_data_into_main(
    main_data_path: str,
    projected_lsoa_data_path: str,
    output_path: str
):
    """
    Merges LSOA projected population and density data back into the main dataset.

    Args:
        main_data_path (str): Path to the main data CSV (e.g., final_processed_data.csv).
        projected_lsoa_data_path (str): Path to the CSV containing LSOA projected data (wide format).
        output_path (str): Path to save the final merged data CSV.
    """
    try:
        main_df = pd.read_csv(main_data_path)
        projected_wide_df = pd.read_csv(projected_lsoa_data_path)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return

    # Prepare projected_wide_df for pd.wide_to_long by renaming columns
    # Example: 'population_2022' -> 'population2022'
    df_to_melt = projected_wide_df.copy()
    rename_map = {}
    for year in [2022, 2023, 2024, 2025]:
        rename_map[f'population_{year}'] = f'population{year}'
        rename_map[f'density_{year}'] = f'density{year}'
    df_to_melt.rename(columns=rename_map, inplace=True)

    # Identify ID variables for wide_to_long.
    # These are columns in projected_wide_df that are unique per LSOA and not year-specific.
    # Common ones: 'LSOA code', 'borough', 'area_km2', 'LSOA11CD'
    id_vars = ['LSOA code', 'borough', 'area_km2', 'LSOA11CD']
    # Ensure id_vars are actually in the dataframe
    id_vars = [col for col in id_vars if col in df_to_melt.columns]
    if not id_vars or 'LSOA code' not in id_vars : # 'LSOA code' is essential
        print("Error: 'LSOA code' missing from ID variables for melting projected data.")
        return

    try:
        projected_long_df = pd.wide_to_long(
            df_to_melt,
            stubnames=['population', 'density'],
            i=id_vars,
            j='Year',
            sep='', # No separator between stubname and year
            suffix='\\d+' # Year is numeric
        ).reset_index()
    except Exception as e:
        print(f"Error reshaping projected data with wide_to_long: {e}")
        return

    # Rename new 'population' and 'density' columns to avoid clashes and indicate they are projected
    projected_long_df.rename(
        columns={'population': 'ProjectedPopulation', 'density': 'ProjectedDensity'},
        inplace=True
    )
    
    # Merge with the main dataset
    # Ensure 'Year' in main_df is integer type for merging
    if 'Year' in main_df.columns and main_df['Year'].dtype != projected_long_df['Year'].dtype:
        try:
            main_df['Year'] = main_df['Year'].astype(int)
        except ValueError:
            print("Error: Could not convert 'Year' column in main_df to integer for merging.")
            return
            
    merged_df = pd.merge(
        main_df,
        projected_long_df[['LSOA code', 'Year', 'ProjectedPopulation', 'ProjectedDensity']],
        on=['LSOA code', 'Year'],
        how='left'
    )

    # Update 'Population' and 'population_density' columns in the main_df logic
    # Where projected data is available, use it. Otherwise, keep original.
    merged_df['Population'] = merged_df['ProjectedPopulation'].fillna(merged_df['Population'])
    merged_df['population_density'] = merged_df['ProjectedDensity'].fillna(merged_df['population_density'])

    # Drop the temporary projected columns
    merged_df.drop(columns=['ProjectedPopulation', 'ProjectedDensity'], inplace=True)

    try:
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully saved final merged data to: {output_path}")
        print(f"Shape of the final data: {merged_df.shape}")
        print("Columns in final data:", merged_df.columns.tolist())
    except Exception as e:
        print(f"Error saving final merged data: {e}")

# Example of how to run the main function and see the output:
if __name__ == '__main__':
    # Define base paths (assuming script is run from workspace root or paths are relative to it)
    data_processed_dir = "data/processed"
    data_pop_est_dir = "data/pop_est"

    # Ensure directories exist (optional, for robustness if creating dummy files later)
    # os.makedirs(data_processed_dir, exist_ok=True)
    # os.makedirs(data_pop_est_dir, exist_ok=True)

    # File paths
    final_processed_csv = os.path.join(data_processed_dir, "final_processed_data_new.csv")
    projections_xlsx = os.path.join(data_pop_est_dir, "London Datastore Population Projections 10yr.xlsx")
    lsoa_lookup_csv = os.path.join(data_pop_est_dir, "LSOA Best Fit Lookup 2011-2022.csv")
    
    # Path for the intermediate projected LSOA data
    projected_lsoa_output_csv = os.path.join(data_processed_dir, "lsoa_projected_population_density_2022_2025.csv")
    # Path for the final merged dataset
    final_density_output_csv = os.path.join(data_processed_dir, "final_density_data.csv")

    # --- Step 1: Generate and save LSOA projected populations and densities ---
    print("Attempting to run LSOA population projection pipeline...")
    # Note: The dummy file creation lines from the previous example are removed for brevity.
    # Ensure your actual data files are in the paths specified or update paths.
    try:
        projected_lsoa_data = project_lsoa_populations_and_density(
            final_processed_data_path=final_processed_csv,
            projections_path=projections_xlsx,
            lsoa_lookup_path=lsoa_lookup_csv
        )
        
        if projected_lsoa_data is not None and not projected_lsoa_data.empty:
            print("\nProjected LSOA Data (Head):")
            print(projected_lsoa_data.head())
            print(f"Shape of the projected LSOA data: {projected_lsoa_data.shape}")
            print(f"Columns in projected LSOA data: {projected_lsoa_data.columns.tolist()}")
            
            # Save this intermediate "cleaned" DataFrame
            projected_lsoa_data.to_csv(projected_lsoa_output_csv, index=False)
            print(f"\nSuccessfully saved LSOA projected data to: {projected_lsoa_output_csv}")

            # Check for NaNs which might indicate merge issues or missing data in projections
            print("\nNaN counts per column in projected LSOA data:")
            print(projected_lsoa_data.isnull().sum())

            # --- Step 2: Merge projected data back into the main dataset ---
            print(f"\nAttempting to merge projected data into {final_processed_csv}...")
            merge_projected_data_into_main(
                main_data_path=final_processed_csv,
                projected_lsoa_data_path=projected_lsoa_output_csv,
                output_path=final_density_output_csv
            )
        else:
            print("LSOA population projection did not return data. Skipping merge.")
            
    except FileNotFoundError as e:
        print(f"Error during pipeline execution: {e}. Please check file paths and availability.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
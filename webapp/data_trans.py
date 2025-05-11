import pandas as pd
import os
import re # For parsing year from sheet names
import geopandas as gpd # Added for geospatial operations

# Define file paths
INPUT_CSV_PATH = "data/01_final_data/Burglary-Classified-Data.csv"
OUTPUT_DIR = "data/bigdata" # Relative to the workspace root
OUTPUT_CSV_NAME = "burglary_data.csv"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
BURGLARY_DATA_PATH = OUTPUT_CSV_PATH # Path to the output of the first function

POP_EST_DIR = "data/pop_est"
POPULATION_FILES = [
    "mid2011_mid2014.xlsx",
    "mid2015_mid2018.xlsx",
    "mid2019_mid2022.xlsx"
]
LSOA_BOUNDARIES_GEOJSON = os.path.join(POP_EST_DIR, "LSOA_2011_boundaries.geojson") # Path to LSOA boundaries

PROCESSED_DATA_DIR = "data/processed" # Directory for final output
FINAL_OUTPUT_CSV_NAME = "final_processed_data.csv"
FINAL_OUTPUT_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, FINAL_OUTPUT_CSV_NAME)


# Define columns to keep from the initial burglary data filtering
COLUMNS_TO_KEEP = [
    'Month',
    'Longitude',
    'Latitude',
    'Location',
    'LSOA code',
    'LSOA name',
    'Crime type',
    'Falls within'
]


def filter_and_save_burglary_data(input_path, output_path, columns_to_keep):
    """
    Loads data, filters for burglaries, selects relevant columns,
    and saves the processed data.
    """
    try:
        print(f"--- Running filter_and_save_burglary_data ---")
        print(f"Reading data from {input_path}...")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Successfully read {len(df)} rows.")

        print(f"Filtering for 'Crime type' == 'Burglary'...")
        burglary_df = df[df['Crime type'] == "Burglary"].copy()
        print(f"Found {len(burglary_df)} burglary records.")

        if burglary_df.empty:
            print("No 'Burglary' records found. Output file will not be created.")
            return False # Indicate failure or no data

        print(f"Selecting relevant columns: {', '.join(columns_to_keep)}")
        existing_columns_to_keep = [col for col in columns_to_keep if col in burglary_df.columns]
        missing_cols = set(columns_to_keep) - set(existing_columns_to_keep)
        if missing_cols:
            print(f"Warning: The following specified columns were not found and will be skipped: {', '.join(missing_cols)}")

        if not existing_columns_to_keep:
            print("Error: None of the specified columns to keep exist in the filtered data. Output file will not be created.")
            return False

        processed_df = burglary_df[existing_columns_to_keep]

        output_dir_path = os.path.dirname(output_path)
        if not os.path.exists(output_dir_path):
            print(f"Creating directory: {output_dir_path}")
            os.makedirs(output_dir_path)

        print(f"Saving processed data to {output_path}...")
        processed_df.to_csv(output_path, index=False)
        print(f"Successfully saved burglary data to {output_path}")
        print(f"--- filter_and_save_burglary_data finished ---")
        return True # Indicate success
    except FileNotFoundError:
        print(f"Error: The input file was not found at {input_path}")
    except KeyError as e:
        print(f"Error: A specified column was not found in the CSV. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in filter_and_save_burglary_data: {e}")
    return False

def get_year_from_sheet_name(name_to_parse):
    """Extracts a 4-digit year (20xx) from a string."""
    match = re.search(r'(20\d{2})', str(name_to_parse))
    if match:
        return int(match.group(1))
    return None

def find_column_by_priority(df, potential_names):
    """Finds the first existing column name from a list of potential names."""
    for name in potential_names:
        if name in df.columns:
            return name
    return None

def load_and_merge_population_data(burglary_data_path, pop_est_dir, population_files):
    """
    Loads population estimates from Excel files, maps LSOA2021 to LSOA2011 codes,
    merges with burglary data, and returns the combined DataFrame.
    """
    print(f"--- Running load_and_merge_population_data ---")

    # --- Load LSOA 2021 to LSOA 2011 Lookup Table ---
    # IMPORTANT: User needs to verify these column names in their lookup CSV.
    lsoa_lookup_path = "data/01_final_data/LSOA2011_LSOA2021_LAD2022.csv"
    lsoa21_col_in_lookup = 'LSOA21CD'  # Expected column name for LSOA 2021 codes in lookup
    lsoa11_col_in_lookup = 'LSOA11CD'  # Expected column name for LSOA 2011 codes in lookup
    
    try:
        print(f"Loading LSOA lookup table from: {lsoa_lookup_path}")
        lookup_df = pd.read_csv(lsoa_lookup_path)
        # Ensure necessary columns exist in lookup_df
        if lsoa21_col_in_lookup not in lookup_df.columns or lsoa11_col_in_lookup not in lookup_df.columns:
            print(f"Error: Lookup CSV \'{lsoa_lookup_path}\' must contain columns \'{lsoa21_col_in_lookup}\' and \'{lsoa11_col_in_lookup}\'.")
            print(f"Found columns: {lookup_df.columns.tolist()}")
            return None
        # Keep only necessary columns and drop duplicates to ensure one-to-one or many-to-one mapping from 2021 to 2011
        lookup_df = lookup_df[[lsoa21_col_in_lookup, lsoa11_col_in_lookup]].drop_duplicates(subset=[lsoa21_col_in_lookup])
        lookup_df[lsoa21_col_in_lookup] = lookup_df[lsoa21_col_in_lookup].astype(str)
        lookup_df[lsoa11_col_in_lookup] = lookup_df[lsoa11_col_in_lookup].astype(str)
        print(f"Successfully loaded LSOA lookup table. {len(lookup_df)} unique LSOA21 codes found.")
    except FileNotFoundError:
        print(f"Error: LSOA lookup file not found at \'{lsoa_lookup_path}\'.")
        return None
    except Exception as e:
        print(f"Error reading LSOA lookup CSV: {e}")
        return None

    all_population_data = []

    # LSOA_CODE_POTENTIAL_NAMES should now ideally find LSOA 2021 codes in the Excel files
    LSOA_CODE_POTENTIAL_NAMES = [
    'LSOA 2021 Code', 'LSOA21CD', 'LSOA Code (2021 boundaries)', 
    'Area Codes', 'Area Code', 'LSOA Code', 'LSOA code', 'GEOGRAPHY_CODE' 
    # Added LSOA21 specific names at higher priority
    ]

    POPULATION_POTENTIAL_NAMES = [
        'Total', 'All Ages', 'Total Population', 'Population', 'LSOA_COUNT', 'All ages', 'Persons'
    ]


    for pop_file_name in population_files:
        file_path = os.path.join(pop_est_dir, pop_file_name)
        print(f"Processing population file: {file_path}")
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            print(f"  Sheets found: {sheet_names}")
            for sheet_name in sheet_names:
                year = get_year_from_sheet_name(sheet_name)
                if year:
                    print(f"    Processing sheet: '{sheet_name}' for year {year}")
                    try:
                        df_sheet = None
                        lsoa_col_name_in_excel = None # Store the identified LSOA 2021 col name
                        for skip in range(5): # Try skipping up to 4 header rows
                            try:
                                temp_df = pd.read_excel(xls, sheet_name, skiprows=skip)
                                lsoa_col_name_in_excel_cand = find_column_by_priority(temp_df, LSOA_CODE_POTENTIAL_NAMES)
                                pop_col_name = find_column_by_priority(temp_df, POPULATION_POTENTIAL_NAMES)
                                if lsoa_col_name_in_excel_cand and pop_col_name:
                                    df_sheet = temp_df
                                    lsoa_col_name_in_excel = lsoa_col_name_in_excel_cand
                                    print(f"      Successfully read sheet '{sheet_name}' with LSOA col '{lsoa_col_name_in_excel}' and Pop col '{pop_col_name}' by skipping {skip} rows.")
                                    break
                            except Exception:
                                continue
                        if df_sheet is None:
                            print(f"      Error: Could not find LSOA code (expected LSOA21 types like {LSOA_CODE_POTENTIAL_NAMES}) or Population columns in sheet '{sheet_name}' or failed to read. Population columns: {POPULATION_POTENTIAL_NAMES}. Please check the Excel file structure.")
                            continue

                        # lsoa_col_name_in_excel should be set if df_sheet is not None
                        pop_col_name = find_column_by_priority(df_sheet, POPULATION_POTENTIAL_NAMES) # Re-confirm pop col

                        if lsoa_col_name_in_excel and pop_col_name:
                            # Select LSOA 2021 code and Population
                            sheet_data_raw = df_sheet[[lsoa_col_name_in_excel, pop_col_name]].copy()
                            sheet_data_raw.rename(columns={pop_col_name: 'Population'}, inplace=True)
                            sheet_data_raw[lsoa_col_name_in_excel] = sheet_data_raw[lsoa_col_name_in_excel].astype(str)
                            
                            # Merge with lookup table to get LSOA 2011 codes
                            sheet_data_mapped = pd.merge(
                                sheet_data_raw,
                                lookup_df,
                                left_on=lsoa_col_name_in_excel,
                                right_on=lsoa21_col_in_lookup,
                                how='inner' # Use inner to keep only LSOAs that have a 2011 equivalent ============================ 'left'
                            )
                            
                            if sheet_data_mapped.empty:
                                print(f"      Warning: No LSOA 2021 codes from sheet '{sheet_name}' found in the lookup table or resulted in empty data after merge.")
                                continue

                            # Now use LSOA11 code from lookup as 'LSOA code'
                            sheet_data_final = sheet_data_mapped[[lsoa11_col_in_lookup, 'Population']].copy()
                            sheet_data_final.rename(columns={lsoa11_col_in_lookup: 'LSOA code'}, inplace=True)
                            sheet_data_final['Year'] = year
                            sheet_data_final.dropna(subset=['LSOA code', 'Population'], inplace=True)
                            
                            all_population_data.append(sheet_data_final)
                            print(f"      Extracted and mapped {len(sheet_data_final)} rows for year {year} from sheet '{sheet_name}' using LSOA2011 codes.")
                        else:
                            print(f"      Skipping sheet '{sheet_name}': Could not find required LSOA (2021) or Population columns after attempting read.")
                    except Exception as e_sheet:
                        print(f"      Error reading sheet '{sheet_name}': {e_sheet}")
                else:
                    print(f"    Skipping sheet '{sheet_name}': Could not determine year directly from sheet name.")
        except FileNotFoundError:
            print(f"  Error: Population file not found at {file_path}")
        except Exception as e_file:
            print(f"  Error processing file {file_path}: {e_file}")

    if not all_population_data:
        print("No population data extracted. Cannot proceed with merge.")
        return None

    combined_pop_df = pd.concat(all_population_data, ignore_index=True)
    combined_pop_df.drop_duplicates(subset=['LSOA code', 'Year'], keep='first', inplace=True)
    print(f"Combined population data has {len(combined_pop_df)} rows.")

    print(f"Loading burglary data from {burglary_data_path}...")
    try:
        burglary_df = pd.read_csv(burglary_data_path, low_memory=False)
        print(f"Successfully loaded {len(burglary_df)} burglary records.")
    except FileNotFoundError:
        print(f"Error: Burglary data file not found at {burglary_data_path}. Make sure 'filter_and_save_burglary_data' ran successfully if not commented out.")
        return None
    except Exception as e:
        print(f"Error loading burglary data: {e}")
        return None

    burglary_df['Year'] = pd.to_datetime(burglary_df['Month']).dt.year
    combined_pop_df['Year'] = combined_pop_df['Year'].astype(int)
    burglary_df['Year'] = burglary_df['Year'].astype(int)
    burglary_df['LSOA code'] = burglary_df['LSOA code'].astype(str)
    combined_pop_df['LSOA code'] = combined_pop_df['LSOA code'].astype(str)

    print("Merging population data (now keyed by LSOA2011) with burglary data...")
    pop_data = pd.merge(burglary_df, combined_pop_df, on=['LSOA code', 'Year'], how='left')
    print(f"Merged data has {len(pop_data)} rows.")
    pop_merged_count = pop_data['Population'].notna().sum()
    print(f"Number of rows with successfully merged population data: {pop_merged_count} out of {len(pop_data)}.")
    if pop_merged_count == 0 and len(pop_data) > 0:
        print("Warning: Population data did not merge. Check LSOA codes and Year matching, and content of population files.")

    print(f"--- load_and_merge_population_data finished ---\n")
    return pop_data

def calculate_and_merge_population_density(input_df, lsoa_boundaries_geojson_path):
    """
    Calculates LSOA-level population density and merges it with the input DataFrame.
    The input_df is assumed to have LSOA codes that match the GeoJSON (i.e., LSOA2011).
    Args:
        input_df (pd.DataFrame): DataFrame containing at least 'LSOA code' (LSOA2011) and 'Population'.
        lsoa_boundaries_geojson_path (str): Path to the LSOA 2011 boundaries GeoJSON file.
    Returns:
        pd.DataFrame: Input DataFrame merged with 'area_km2' and 'population_density'.
                     Returns None if an error occurs.
    """
    print(f"--- Running calculate_and_merge_population_density ---")
    if 'LSOA code' not in input_df.columns or 'Population' not in input_df.columns:
        print("Error: Input DataFrame for density calculation must contain 'LSOA code' and 'Population' columns.")
        return None
    if input_df['Population'].isnull().all():
        print("Warning: 'Population' column is all NaN. Density calculation will result in NaN.")

    try:
        print(f"Loading LSOA boundaries from: {lsoa_boundaries_geojson_path}")
        gdf = gpd.read_file(lsoa_boundaries_geojson_path)
        print(f"Successfully loaded {len(gdf)} LSOA boundaries. Original CRS: {gdf.crs}")

        print("Reprojecting to EPSG:27700...")
        gdf_reprojected = gdf.to_crs(epsg=27700)
        print(f"Reprojection successful. New CRS: {gdf_reprojected.crs}")

        gdf_reprojected["area_km2"] = gdf_reprojected["geometry"].area / 1e6
        print("Calculated area_km2.")

        # LSOA_CODE_POTENTIAL_NAMES in GeoJSON should be LSOA2011 types
        lsoa_code_col_in_geojson = 'LSOA11CD' 
        if lsoa_code_col_in_geojson not in gdf_reprojected.columns:
            potential_lsoa_cols_geojson = ['LSOA11CD', 'LSOA_CODE', 'lsoa_code', 'LSOACD', 'LSOA code', 'LSOA CODE'] # Prioritize LSOA11CD
            found_col = find_column_by_priority(gdf_reprojected, potential_lsoa_cols_geojson)
            if found_col:
                lsoa_code_col_in_geojson = found_col
                print(f"Using LSOA code column '{found_col}' from GeoJSON.")
            else:
                print(f"Error: Could not find a suitable LSOA code column (e.g., LSOA11CD) in GeoJSON columns: {gdf_reprojected.columns.tolist()}")
                return None
        
        area_df = gdf_reprojected[[lsoa_code_col_in_geojson, "area_km2"]].copy()
        area_df[lsoa_code_col_in_geojson] = area_df[lsoa_code_col_in_geojson].astype(str)

        print(f"Merging area data with input DataFrame on LSOA code ('{input_df['LSOA code'].name}' and '{lsoa_code_col_in_geojson}')...")
        output_df = input_df.copy() # Work on a copy
        output_df['LSOA code'] = output_df['LSOA code'].astype(str)
        
        merged_df = pd.merge(output_df, area_df, left_on="LSOA code", right_on=lsoa_code_col_in_geojson, how="left")

        if lsoa_code_col_in_geojson != "LSOA code" and lsoa_code_col_in_geojson in merged_df.columns:
            merged_df.drop(columns=[lsoa_code_col_in_geojson], inplace=True)

        merged_df["population_density"] = merged_df["Population"] / merged_df["area_km2"]
        merged_df["population_density"].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        print("Calculated population_density.")

        density_merged_count = merged_df['population_density'].notna().sum()
        area_merged_count = merged_df['area_km2'].notna().sum()
        print(f"Number of rows with successfully merged area_km2: {area_merged_count} out of {len(merged_df)}.")
        print(f"Number of rows with successfully calculated population_density: {density_merged_count} out of {len(merged_df)}.")
        if area_merged_count == 0 and len(merged_df) > 0:
            print("Warning: Area data did not merge. Check LSOA codes in GeoJSON vs. burglary data.")

        print(f"--- calculate_and_merge_population_density finished ---\n")
        return merged_df

    except FileNotFoundError:
        print(f"Error: LSOA boundaries GeoJSON file not found at {lsoa_boundaries_geojson_path}")
        return None
    except ImportError:
        print("Error: geopandas library is required for population density calculation. Please install it: pip install geopandas")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in calculate_and_merge_population_density: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Filter and save burglary data (Commented out as requested by user)
    # print("Ensuring burglary_data.csv is up to date...")
    # success_step1 = filter_and_save_burglary_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH, COLUMNS_TO_KEEP)
    # if not success_step1:
    #     print("Problem with filter_and_save_burglary_data. Exiting.")
    #     exit()
    # else:
    #     print("burglary_data.csv is ready.")

    if not os.path.exists(BURGLARY_DATA_PATH):
        print(f"Error: Expected burglary data file not found at {BURGLARY_DATA_PATH}.")
        print("Please run the script once without commenting out Step 1, or ensure the file is correctly placed.")
        exit()

    # Step 2: Load population data and merge it with burglary data
    pop_data_df = load_and_merge_population_data(BURGLARY_DATA_PATH, POP_EST_DIR, POPULATION_FILES)

    if pop_data_df is not None and not pop_data_df.empty:
        print("\n--- pop_data_df (burglary data with population) Info: ---")
        pop_data_df.info(verbose=True, show_counts=True)
        print("\n--- First 5 rows of pop_data_df: ---")
        print(pop_data_df.head())
        
        # Step 3: Calculate population density
        final_df = calculate_and_merge_population_density(pop_data_df, LSOA_BOUNDARIES_GEOJSON)

        if final_df is not None and not final_df.empty:
            print("\n--- Final DataFrame (with population density) Info: ---")
            final_df.info(verbose=True, show_counts=True)
            print("\n--- First 5 rows of Final DataFrame: ---")
            print(final_df.head())
            print("\n--- Last 5 rows of Final DataFrame: ---")
            print(final_df.tail())
            print(f"\n--- Population Density Stats ---\n{final_df['population_density'].describe()}")
            print(f"\n--- Area (km2) Stats ---\n{final_df['area_km2'].describe()}")


            # Step 4: Save the final processed data
            try:
                if not os.path.exists(PROCESSED_DATA_DIR):
                    print(f"Creating directory: {PROCESSED_DATA_DIR}")
                    os.makedirs(PROCESSED_DATA_DIR)
                print(f"\nSaving final processed data to {FINAL_OUTPUT_CSV_PATH}...")
                final_df.to_csv(FINAL_OUTPUT_CSV_PATH, index=False)
                print(f"Successfully saved final data to {FINAL_OUTPUT_CSV_PATH}")
            except Exception as e:
                print(f"Error saving final data: {e}")
        else:
            print("Failed to calculate population density or merge area data.")
    else:
        print("Failed to generate pop_data_df (burglary data with population). Aborting density calculation.")

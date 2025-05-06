'''
Script to load pre-engineered burglary features and integrate external datasets 
(IMD, Census, PTAL).
'''

import pandas as pd
import geopandas as gpd
import os

# Define file paths
DATA_DIR = "data" # Assuming script is run from scripts/ directory
FEATURES_PATH = os.path.join(DATA_DIR, "features_engineered.csv")
IMD_DIR = os.path.join(DATA_DIR, "English IMD 2019")
IMD_SHAPEFILE = os.path.join(IMD_DIR, "IMD_2019.shp")
CENSUS_DIR = os.path.join(DATA_DIR, "census_data")
PTAL_PATH = os.path.join(DATA_DIR, "PTAL_2008-2014.csv")
LOOKUP_PATH = os.path.join(CENSUS_DIR, "postcode_oa_lsoa_msoa_lad_2011_ew.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
MERGED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "merged_data.csv")

print("Loading pre-engineered burglary features...")
burglary_df = pd.read_csv(FEATURES_PATH)
print(f"Loaded {burglary_df.shape[0]} rows and {burglary_df.shape[1]} columns.")
print("Burglary data columns:", burglary_df.columns.tolist())
print(burglary_df.head())

print("\nLoading IMD shapefile...")
try:
    imd_gdf = gpd.read_file(IMD_SHAPEFILE)
    print(f"Loaded {imd_gdf.shape[0]} rows and {imd_gdf.shape[1]} columns.")
    print("IMD CRS:", imd_gdf.crs)

    # --- Merge IMD Data ---
    print("\nMerging IMD data...")
    # Select and rename columns
    # Ensure 'lsoa11cd' and 'IMDScore' exist in the columns printed previously
    if 'lsoa11cd' in imd_gdf.columns and 'IMDScore' in imd_gdf.columns:
        imd_to_merge = imd_gdf[['lsoa11cd', 'IMDScore']].copy()
        imd_to_merge.rename(columns={'lsoa11cd': 'LSOA code'}, inplace=True)

        # Perform the merge
        burglary_df = pd.merge(burglary_df, imd_to_merge, on='LSOA code', how='left')

        # Check merge results
        print(f"Shape after merging IMD: {burglary_df.shape}")
        print("Columns after merging IMD:", burglary_df.columns.tolist())
        missing_imd = burglary_df['IMDScore'].isnull().sum()
        print(f"Missing IMDScore values after merge: {missing_imd}")
        if missing_imd > 0:
            print(f"Percentage missing: {missing_imd / len(burglary_df) * 100:.2f}%")
            # Identify which LSOAs are missing IMD scores (optional troubleshooting)
            # missing_lsoas = burglary_df[burglary_df['IMDScore'].isnull()]['LSOA code'].unique()
            # print("LSOAs missing IMD scores:", missing_lsoas)
        # print(burglary_df.head()) # Optional: view head after merge
    else:
        print("Error: Required columns ('lsoa11cd' or 'IMDScore') not found in IMD data.")

    # --- Next steps: Load and Merge Census Data ---
    print("\nLoading lookup table...")
    try:
        # Anticipate potential column names - adjust if needed after inspection
        oa_col_lookup = 'OA11CD'
        lsoa_col_lookup = 'LSOA11CD'
        lookup_df = pd.read_csv(LOOKUP_PATH, usecols=[oa_col_lookup, lsoa_col_lookup])
        lookup_df = lookup_df.drop_duplicates()
        print(f"Loaded lookup table: {lookup_df.shape}")
        print(lookup_df.head())
    except FileNotFoundError:
        print(f"Error: Lookup file not found at {LOOKUP_PATH}")
        lookup_df = None # Set to None so subsequent steps can check
    except ValueError as e:
        print(f"Error loading lookup file. Check column names ('{oa_col_lookup}', '{lsoa_col_lookup}') exist? Error: {e}")
        lookup_df = None
    except Exception as e:
        print(f"Error processing lookup file: {e}")
        lookup_df = None

    print("\nLoading and preparing Census data...")
    census_files = {
        'age': os.path.join(CENSUS_DIR, 'census_age.csv'),
        'accommodation': os.path.join(CENSUS_DIR, 'census_accommodation.csv'),
        'dwellings': os.path.join(CENSUS_DIR, 'census_dwellings.csv'),
        'tenure': os.path.join(CENSUS_DIR, 'census_tenure.csv')
    }
    
    # Load Age data first
    if lookup_df is not None:
        try:
            age_df = pd.read_csv(census_files['age'])
            print(f"Loaded age data: {age_df.shape}")
            # print("Age data columns:", age_df.columns.tolist()) # Optional: Reduce verbosity
            # print(age_df.head()) # Optional: Reduce verbosity
            
            # --- Calculate Mean Age (OA level) ---
            print("Calculating mean age at OA level...")
            # Extract age columns (assuming they are consistently named like 'Age: Age X; measures: Value')
            age_cols = [col for col in age_df.columns if col.startswith('Age: Age ') and '; measures: Value' in col]
            
            weighted_age_sum = 0
            for col in age_cols:
                try:
                    # Extract age number from column name
                    age_str = col.split('Age ')[1].split(';')[0]
                    if age_str == 'under 1':
                        age = 0 # Treat age under 1 as 0 for mean calculation
                    elif age_str == '100 and over':
                        age = 100 # Use 100 for 100 and over
                    else:
                        age = int(age_str)
                    
                    # Add age * count to sum, handle potential non-numeric data
                    weighted_age_sum += pd.to_numeric(age_df[col], errors='coerce').fillna(0) * age
                except ValueError:
                    pass # Ignore parsing errors
                except Exception:
                    pass # Ignore other column errors

            total_pop_col = 'Age: All categories: Age; measures: Value'
            if total_pop_col in age_df.columns:
                total_population = pd.to_numeric(age_df[total_pop_col], errors='coerce')
                age_df['mean_age_oa'] = weighted_age_sum / total_population.replace(0, pd.NA)
                # print(f"Calculated OA mean age. Min: {age_df['mean_age_oa'].min():.2f}, Max: {age_df['mean_age_oa'].max():.2f}")
            else:
                print(f"Error: Total population column '{total_pop_col}' not found for mean age calc.")
                age_df['mean_age_oa'] = pd.NA
                
            # --- Merge with Lookup --- 
            print("Merging age data with lookup...")
            oa_col_census = 'geography code' # OA code column in census file
            if oa_col_census in age_df.columns and 'mean_age_oa' in age_df.columns:
                # Select only OA code and calculated mean age
                age_oa_df = age_df[[oa_col_census, 'mean_age_oa']].copy()
                # Merge with lookup to get LSOA code
                age_lsoa_df = pd.merge(age_oa_df, lookup_df, left_on=oa_col_census, right_on=oa_col_lookup, how='inner')
                print(f"Shape after merging age OA data with lookup: {age_lsoa_df.shape}")

                # --- Aggregate to LSOA level ---
                print("Aggregating mean age to LSOA level...")
                # Group by LSOA code and calculate the mean of the OA mean ages
                # A population-weighted mean might be more accurate if OA populations are available
                lsoa_agg_age = age_lsoa_df.groupby(lsoa_col_lookup)['mean_age_oa'].mean().reset_index()
                lsoa_agg_age.rename(columns={'mean_age_oa': 'lsoa_mean_age', lsoa_col_lookup: 'LSOA code'}, inplace=True)
                print(f"Aggregated LSOA age data shape: {lsoa_agg_age.shape}")
                print(f"Calculated LSOA mean age. Min: {lsoa_agg_age['lsoa_mean_age'].min():.2f}, Max: {lsoa_agg_age['lsoa_mean_age'].max():.2f}")

                # --- Merge LSOA-level age data ---
                print("Merging LSOA-level mean age data...")
                burglary_df = pd.merge(burglary_df, lsoa_agg_age, on='LSOA code', how='left')
                print(f"Shape after merging LSOA age data: {burglary_df.shape}")
                missing_lsoa_age = burglary_df['lsoa_mean_age'].isnull().sum()
                print(f"Missing LSOA mean_age values after merge: {missing_lsoa_age}")
                if missing_lsoa_age > 0:
                     print(f"Percentage missing LSOA age: {missing_lsoa_age / len(burglary_df) * 100:.2f}%")

                # --- Calculate and Merge Population Density ---
                print("\nCalculating Population Density...")
                # 1. Get LSOA area from IMD data (already loaded as imd_gdf)
                if imd_gdf is not None and 'lsoa11cd' in imd_gdf.columns and 'geometry' in imd_gdf.columns:
                    lsoa_areas = imd_gdf[['lsoa11cd', 'geometry']].copy()
                    # Calculate area in square kilometers (assuming CRS is in meters, like EPSG:27700)
                    lsoa_areas['area_sqkm'] = lsoa_areas['geometry'].area / 1_000_000 
                    lsoa_areas.rename(columns={'lsoa11cd': 'LSOA code'}, inplace=True)
                    print(f"Calculated LSOA areas for {lsoa_areas.shape[0]} LSOAs.")

                    # 2. Aggregate total population to LSOA level from age_df (OA level)
                    total_pop_col = 'Age: All categories: Age; measures: Value'
                    oa_col_census = 'geography code' # OA code column in census age file
                    if total_pop_col in age_df.columns and oa_col_census in age_df.columns:
                        oa_population_df = age_df[[oa_col_census, total_pop_col]].copy()
                        oa_population_df[total_pop_col] = pd.to_numeric(oa_population_df[total_pop_col], errors='coerce').fillna(0)
                        
                        # Merge OA population with lookup to get LSOA code
                        pop_lsoa_df = pd.merge(oa_population_df, lookup_df, left_on=oa_col_census, right_on=oa_col_lookup, how='inner')
                        
                        # Aggregate total population by LSOA
                        lsoa_population = pop_lsoa_df.groupby(lsoa_col_lookup)[total_pop_col].sum().reset_index()
                        lsoa_population.rename(columns={total_pop_col: 'total_population', lsoa_col_lookup: 'LSOA code'}, inplace=True)
                        print(f"Aggregated total population for {lsoa_population.shape[0]} LSOAs.")

                        # 3. Merge LSOA population with LSOA areas
                        lsoa_pop_area = pd.merge(lsoa_population, lsoa_areas[['LSOA code', 'area_sqkm']], on='LSOA code', how='left')

                        # 4. Calculate Population Density
                        # Avoid division by zero for areas that are zero or NaN
                        lsoa_pop_area['population_density'] = lsoa_pop_area['total_population'] / lsoa_pop_area['area_sqkm'].replace(0, pd.NA)
                        print(f"Calculated population density. Min: {lsoa_pop_area['population_density'].min():.2f}, Max: {lsoa_pop_area['population_density'].max():.2f}")
                        
                        # 5. Merge population_density into burglary_df
                        burglary_df = pd.merge(burglary_df, lsoa_pop_area[['LSOA code', 'population_density']], on='LSOA code', how='left')
                        print(f"Shape after merging population density: {burglary_df.shape}")
                        missing_pop_density = burglary_df['population_density'].isnull().sum()
                        print(f"Missing population_density values: {missing_pop_density}")
                        if missing_pop_density > 0:
                            print(f"Percentage missing population_density: {missing_pop_density / len(burglary_df) * 100:.2f}%")
                    else:
                        print("Error: Could not find required population or OA code columns in age_df for density calculation.")
                else:
                    print("Error: IMD GeoDataFrame not available or missing required columns for area calculation.")

                # --- Process Accommodation Data ---
                print("\nLoading and preparing Accommodation data...")
                try:
                    accom_df = pd.read_csv(census_files['accommodation'])
                    print(f"Loaded accommodation data: {accom_df.shape}")
                    print("Accommodation data columns:", accom_df.columns.tolist())
                    print(accom_df.head())

                    # --- Calculate OA-level percentages ---
                    print("Calculating OA accommodation percentages...")
                    oa_col_accom = 'geography code'
                    total_col = 'Dwelling Type: All categories: Accommodation type; measures: Value'
                    detached_col = 'Dwelling Type: Unshared dwelling: Whole house or bungalow: Detached; measures: Value'
                    semi_col = 'Dwelling Type: Unshared dwelling: Whole house or bungalow: Semi-detached; measures: Value'
                    terraced_col = 'Dwelling Type: Unshared dwelling: Whole house or bungalow: Terraced (including end-terrace); measures: Value'
                    flat_col = 'Dwelling Type: Unshared dwelling: Flat, maisonette or apartment: Total; measures: Value'
                    caravan_col = 'Dwelling Type: Unshared dwelling: Caravan or other mobile or temporary structure; measures: Value'

                    cols_to_process = [total_col, detached_col, semi_col, terraced_col, flat_col, caravan_col]
                    
                    # Ensure required columns exist
                    if not all(col in accom_df.columns for col in [oa_col_accom] + cols_to_process):
                        print("Error: Missing required accommodation columns in accom_df.")
                    else:
                        # Convert count columns to numeric
                        for col in cols_to_process:
                            accom_df[col] = pd.to_numeric(accom_df[col], errors='coerce')
                        
                        # Calculate percentages, handle division by zero
                        total_count = accom_df[total_col].replace(0, pd.NA)
                        accom_df['oa_perc_detached'] = (accom_df[detached_col] / total_count * 100).fillna(0)
                        accom_df['oa_perc_semi'] = (accom_df[semi_col] / total_count * 100).fillna(0)
                        accom_df['oa_perc_terraced'] = (accom_df[terraced_col] / total_count * 100).fillna(0)
                        accom_df['oa_perc_flat'] = (accom_df[flat_col] / total_count * 100).fillna(0)
                        accom_df['oa_perc_caravan'] = (accom_df[caravan_col] / total_count * 100).fillna(0)
                        
                        # Select OA code and percentages
                        perc_cols = ['oa_perc_detached', 'oa_perc_semi', 'oa_perc_terraced', 'oa_perc_flat', 'oa_perc_caravan']
                        accom_oa_to_agg = accom_df[[oa_col_accom] + perc_cols].copy()
                        
                        # --- Merge with lookup, aggregate to LSOA, merge into burglary_df ---
                        print("Merging OA percentages with lookup...")
                        accom_lsoa_df = pd.merge(accom_oa_to_agg, lookup_df, left_on=oa_col_accom, right_on=oa_col_lookup, how='inner')
                        
                        print("Aggregating accommodation percentages to LSOA level...")
                        lsoa_agg_accom = accom_lsoa_df.groupby(lsoa_col_lookup)[perc_cols].mean().reset_index()
                        
                        # Rename columns for LSOA level
                        rename_dict = {col: 'lsoa_' + col.split('oa_')[1] for col in perc_cols}
                        rename_dict[lsoa_col_lookup] = 'LSOA code'
                        lsoa_agg_accom.rename(columns=rename_dict, inplace=True)
                        print(f"Aggregated LSOA accommodation data shape: {lsoa_agg_accom.shape}")

                        print("Merging LSOA accommodation percentages...")
                        burglary_df = pd.merge(burglary_df, lsoa_agg_accom, on='LSOA code', how='left')
                        print(f"Shape after merging accommodation data: {burglary_df.shape}")
                        # Check missing for one of the new columns
                        missing_perc_detached = burglary_df['lsoa_perc_detached'].isnull().sum()
                        print(f"Missing lsoa_perc_detached values: {missing_perc_detached}")
                        if missing_perc_detached > 0:
                             print(f"Percentage missing lsoa_perc_detached: {missing_perc_detached / len(burglary_df) * 100:.2f}%")

                except FileNotFoundError:
                    print(f"Error: Census accommodation file not found at {census_files['accommodation']}")
                except Exception as e:
                    print(f"Error processing Census accommodation data: {e}")

                # --- Process Tenure Data ---
                print("\nLoading and preparing Tenure data...")
                try:
                    tenure_df = pd.read_csv(census_files['tenure'])
                    print(f"Loaded tenure data: {tenure_df.shape}")
                    print("Tenure data columns:", tenure_df.columns.tolist())
                    print(tenure_df.head())

                    # --- Calculate OA-level rental percentage ---
                    print("Calculating OA rental percentage...")
                    oa_col_tenure = 'geography code'
                    total_hh_col = 'Tenure: All households; measures: Value'
                    social_rented_col = 'Tenure: Social rented; measures: Value'
                    private_rented_col = 'Tenure: Private rented; measures: Value'

                    tenure_cols_to_process = [total_hh_col, social_rented_col, private_rented_col]
                    
                    # Ensure required columns exist
                    if not all(col in tenure_df.columns for col in [oa_col_tenure] + tenure_cols_to_process):
                         print("Error: Missing required tenure columns in tenure_df.")
                    else:
                        # Convert count columns to numeric
                        for col in tenure_cols_to_process:
                            tenure_df[col] = pd.to_numeric(tenure_df[col], errors='coerce')
                            
                        # Calculate total rented percentage
                        total_households = tenure_df[total_hh_col].replace(0, pd.NA)
                        total_rented = tenure_df[social_rented_col].fillna(0) + tenure_df[private_rented_col].fillna(0)
                        tenure_df['oa_perc_rented'] = (total_rented / total_households * 100).fillna(0)
                        
                        # Select OA code and percentage
                        tenure_oa_to_agg = tenure_df[[oa_col_tenure, 'oa_perc_rented']].copy()
                        
                        # --- Merge with lookup, aggregate to LSOA, merge into burglary_df ---
                        print("Merging OA rental percentage with lookup...")
                        tenure_lsoa_df = pd.merge(tenure_oa_to_agg, lookup_df, left_on=oa_col_tenure, right_on=oa_col_lookup, how='inner')
                        
                        print("Aggregating rental percentage to LSOA level...")
                        lsoa_agg_tenure = tenure_lsoa_df.groupby(lsoa_col_lookup)['oa_perc_rented'].mean().reset_index()
                        lsoa_agg_tenure.rename(columns={'oa_perc_rented': 'lsoa_perc_rented', lsoa_col_lookup: 'LSOA code'}, inplace=True)
                        print(f"Aggregated LSOA rental percentage data shape: {lsoa_agg_tenure.shape}")

                        print("Merging LSOA rental percentage...")
                        burglary_df = pd.merge(burglary_df, lsoa_agg_tenure, on='LSOA code', how='left')
                        print(f"Shape after merging tenure data: {burglary_df.shape}")
                        missing_perc_rented = burglary_df['lsoa_perc_rented'].isnull().sum()
                        print(f"Missing lsoa_perc_rented values: {missing_perc_rented}")
                        if missing_perc_rented > 0:
                             print(f"Percentage missing lsoa_perc_rented: {missing_perc_rented / len(burglary_df) * 100:.2f}%")

                except FileNotFoundError:
                    print(f"Error: Census tenure file not found at {census_files['tenure']}")
                except Exception as e:
                    print(f"Error processing Census tenure data: {e}")

                # --- Process Dwellings Data ---
                print("\nLoading and preparing Dwellings data...")
                try:
                    dwell_df = pd.read_csv(census_files['dwellings'])
                    print(f"Loaded dwellings data: {dwell_df.shape}")
                    print("Dwellings data columns:", dwell_df.columns.tolist())
                    print(dwell_df.head())

                    # --- Calculate OA-level percentage unoccupied ---
                    print("Calculating OA percentage unoccupied...")
                    oa_col_dwell = 'geography code'
                    total_spaces_col = 'Dwelling Type: All categories: Household spaces; measures: Value'
                    unoccupied_col = 'Dwelling Type: Household spaces with no usual residents; measures: Value'
                    
                    dwell_cols_to_process = [total_spaces_col, unoccupied_col]
                    
                    if not all(col in dwell_df.columns for col in [oa_col_dwell] + dwell_cols_to_process):
                        print("Error: Missing required dwelling columns in dwell_df.")
                    else:
                        # Convert count columns to numeric
                        for col in dwell_cols_to_process:
                            dwell_df[col] = pd.to_numeric(dwell_df[col], errors='coerce')
                        
                        # Calculate percentage unoccupied
                        total_spaces = dwell_df[total_spaces_col].replace(0, pd.NA)
                        dwell_df['oa_perc_unoccupied'] = (dwell_df[unoccupied_col] / total_spaces * 100).fillna(0)
                        
                        # Select OA code and percentage
                        dwell_oa_to_agg = dwell_df[[oa_col_dwell, 'oa_perc_unoccupied']].copy()
                        
                        # --- Merge with lookup, aggregate to LSOA, merge into burglary_df ---
                        print("Merging OA unoccupied percentage with lookup...")
                        dwell_lsoa_df = pd.merge(dwell_oa_to_agg, lookup_df, left_on=oa_col_dwell, right_on=oa_col_lookup, how='inner')
                        
                        print("Aggregating unoccupied percentage to LSOA level...")
                        lsoa_agg_dwell = dwell_lsoa_df.groupby(lsoa_col_lookup)['oa_perc_unoccupied'].mean().reset_index()
                        lsoa_agg_dwell.rename(columns={'oa_perc_unoccupied': 'lsoa_perc_unoccupied', lsoa_col_lookup: 'LSOA code'}, inplace=True)
                        print(f"Aggregated LSOA unoccupied percentage data shape: {lsoa_agg_dwell.shape}")

                        print("Merging LSOA unoccupied percentage...")
                        burglary_df = pd.merge(burglary_df, lsoa_agg_dwell, on='LSOA code', how='left')
                        print(f"Shape after merging dwellings data: {burglary_df.shape}")
                        missing_perc_unoccupied = burglary_df['lsoa_perc_unoccupied'].isnull().sum()
                        print(f"Missing lsoa_perc_unoccupied values: {missing_perc_unoccupied}")
                        if missing_perc_unoccupied > 0:
                             print(f"Percentage missing lsoa_perc_unoccupied: {missing_perc_unoccupied / len(burglary_df) * 100:.2f}%")

                except FileNotFoundError:
                    print(f"Error: Census dwellings file not found at {census_files['dwellings']}")
                except Exception as e:
                    print(f"Error processing Census dwellings data: {e}")

                # --- Finished Census Data --- 

            else: # End of check for required dwelling columns
                 print("Skipping Dwellings processing due to missing columns.")

        except FileNotFoundError:
            print(f"Error: Census age file not found at {census_files['age']}")
        except Exception as e:
            print(f"Error processing Census age data: {e}")
    else:
        print("Skipping all Census data processing because OA-LSOA lookup table failed to load.")

    # --- Load LSOA 2001 to 2011 Lookup ---
    print("\nLoading LSOA 2001 to 2011 lookup table...")
    LSOA_CONVERSION_LOOKUP_PATH = os.path.join(CENSUS_DIR, "LSOA2001_LSOA2011_LAD_EW.csv")
    lsoa01_to_lsoa11_lookup_df = None
    # Column names identified from previous output
    lsoa01_col_actual = 'ï»¿LSOA01CD' # Actual name with BOM if utf-8-sig doesn't clean it
    lsoa11_col_actual = 'LSOA11CD'

    try:
        # Try with 'utf-8-sig' to handle BOM
        temp_lookup = pd.read_csv(LSOA_CONVERSION_LOOKUP_PATH, encoding='utf-8-sig')
        print(f"Loaded LSOA01-LSOA11 lookup: {temp_lookup.shape}")
        print("LSOA01-LSOA11 lookup columns (after utf-8-sig):", temp_lookup.columns.tolist())

        # Check if BOM was handled, otherwise use the original name
        if 'LSOA01CD' in temp_lookup.columns:
            lsoa01_col_for_selection = 'LSOA01CD'
        elif lsoa01_col_actual in temp_lookup.columns:
             lsoa01_col_for_selection = lsoa01_col_actual
        else:
            print(f"Error: Column '{lsoa01_col_actual}' or 'LSOA01CD' not found in LSOA01-LSOA11 lookup.")
            lsoa01_col_for_selection = None

        if lsoa11_col_actual in temp_lookup.columns and lsoa01_col_for_selection:
            lsoa01_to_lsoa11_lookup_df = temp_lookup[[lsoa01_col_for_selection, lsoa11_col_actual]].copy()
            lsoa01_to_lsoa11_lookup_df.rename(columns={lsoa01_col_for_selection: 'LSOA01CD_lookup', lsoa11_col_actual: 'LSOA11CD_lookup'}, inplace=True)
            lsoa01_to_lsoa11_lookup_df.drop_duplicates(inplace=True)
            print(f"Prepared LSOA01-LSOA11 lookup. Shape: {lsoa01_to_lsoa11_lookup_df.shape}")
            # print(lsoa01_to_lsoa11_lookup_df.head())
        else:
            print("Error: Could not prepare LSOA01-LSOA11 lookup due to missing key columns.")
            lsoa01_to_lsoa11_lookup_df = None

    except FileNotFoundError:
        print(f"Error: LSOA01-LSOA11 lookup file not found at {LSOA_CONVERSION_LOOKUP_PATH}")
    except Exception as e:
        print(f"Error loading/processing LSOA01-LSOA11 lookup: {e}")


    # --- Load and Merge PTAL Data ---
    print("\nLoading and processing PTAL data...")
    if lsoa01_to_lsoa11_lookup_df is not None:
        try:
            ptal_df = pd.read_csv(PTAL_PATH)
            print(f"Loaded PTAL data: {ptal_df.shape}")
            # print("PTAL data columns:", ptal_df.columns.tolist()) # Already seen

            # Columns from PTAL file: 'LSOA2001', 'PTAI2015'
            ptal_score_col = 'PTAI2015'
            ptal_lsoa01_col = 'LSOA2001'

            if ptal_lsoa01_col not in ptal_df.columns or ptal_score_col not in ptal_df.columns:
                print("Error: Required columns not found in PTAL data.")
            else:
                ptal_to_merge = ptal_df[[ptal_lsoa01_col, ptal_score_col]].copy()

                # Merge PTAL data with the LSOA01-LSOA11 lookup
                print("Merging PTAL data with LSOA01-LSOA11 lookup...")
                merged_ptal = pd.merge(ptal_to_merge, lsoa01_to_lsoa11_lookup_df,
                                       left_on=ptal_lsoa01_col, right_on='LSOA01CD_lookup', how='left')
                print(f"Shape after merging PTAL with LSOA lookup: {merged_ptal.shape}")

                # Aggregate PTAL scores to LSOA11 level (handling potential one-to-many mappings)
                # Taking the mean of PTAI2015 if multiple LSOA01s map to the same LSOA11 or vice versa after the merge
                if 'LSOA11CD_lookup' in merged_ptal.columns:
                    lsoa11_ptal_scores = merged_ptal.groupby('LSOA11CD_lookup')[ptal_score_col].mean().reset_index()
                    lsoa11_ptal_scores.rename(columns={'LSOA11CD_lookup': 'LSOA code', ptal_score_col: 'ptal_score_2015'}, inplace=True)
                    print(f"Aggregated LSOA11 PTAL scores. Shape: {lsoa11_ptal_scores.shape}")
                    # print(lsoa11_ptal_scores.head())

                    # Merge with main burglary_df
                    print("Merging LSOA11 PTAL scores into main DataFrame...")
                    burglary_df = pd.merge(burglary_df, lsoa11_ptal_scores, on='LSOA code', how='left')
                    print(f"Shape after merging PTAL data: {burglary_df.shape}")
                    missing_ptal = burglary_df['ptal_score_2015'].isnull().sum()
                    print(f"Missing ptal_score_2015 values: {missing_ptal}")
                    if missing_ptal > 0:
                        print(f"Percentage missing ptal_score_2015: {missing_ptal / len(burglary_df) * 100:.2f}%")
                else:
                    print("Error: 'LSOA11CD_lookup' not found after merging PTAL with lookup. Cannot aggregate.")
                    
        except FileNotFoundError:
            print(f"Error: PTAL file not found at {PTAL_PATH}")
        except Exception as e:
            print(f"Error processing PTAL data: {e}")
    else:
        print("Skipping PTAL data processing because LSOA01-LSOA11 lookup failed to load or prepare.")

    # --- Handle Missing Values ---
    print("\nHandling missing values by median imputation...")
    
    cols_to_impute = [
        'IMDScore', 'lsoa_mean_age', 'population_density',
        'lsoa_perc_detached', 'lsoa_perc_semi', 'lsoa_perc_terraced',
        'lsoa_perc_flat', 'lsoa_perc_caravan', 'lsoa_perc_rented',
        'lsoa_perc_unoccupied', 'ptal_score_2015'
    ]

    for col in cols_to_impute:
        if col in burglary_df.columns:
            if burglary_df[col].isnull().any():
                median_val = burglary_df[col].median()
                burglary_df[col].fillna(median_val, inplace=True)
                print(f"Imputed missing values in '{col}' with median: {median_val:.2f}")
            else:
                print(f"No missing values to impute in '{col}'.")
        else:
            print(f"Warning: Column '{col}' not found for imputation.")

    # --- Save Merged and Imputed Data --- 
    print("\nSaving merged and imputed data...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
        burglary_df.to_csv(MERGED_OUTPUT_PATH, index=False)
        print(f"Merged data saved to {MERGED_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving merged data: {e}")

except ImportError:
    print("Error: geopandas is required to read shapefiles.")
    print("Please install it: pip install geopandas")
except Exception as e:
    print(f"Error processing IMD data: {e}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Final Check ---
print("\n-------------------------------------")
print("Data Integration Script Completed.")
print(f"Final DataFrame shape: {burglary_df.shape}")
print("Final Columns:", burglary_df.columns.tolist())
print("Missing values per column:")
print(burglary_df.isnull().sum())
print("-------------------------------------")

# print("\nScript finished processing IMD.") # Old message 
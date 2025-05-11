import pandas as pd
import os

def merge_burglary_unemployment_data():
    """
    Reads burglary and unemployment data, merges them by month,
    and saves the combined DataFrame.
    """
    # Define file paths
    # Assuming the script is run from the root of the burglary_analysis project
    unemployment_csv_path = "data/pop_est/claimant_count.csv"
    burglary_csv_path = "data/processed/final_density_data.csv"
    output_csv_path = "data/processed/burglary_with_unemployment_new.csv"

    # --- 1. Read both CSVs into pandas DataFrames ---
    try:
        unemployment_df = pd.read_csv(unemployment_csv_path)
        burglary_df = pd.read_csv(burglary_csv_path)
    except FileNotFoundError as e:
        print(f"Error: One or both input files not found. {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    print("Successfully read both CSV files.")

    # --- 2. Process unemployment data ---
    # Parse the "Date" column into datetime, then format as "YYYY-MM"
    # Using errors='coerce' will turn unparseable dates into NaT (Not a Time)
    unemployment_df['Month_dt'] = pd.to_datetime(
        unemployment_df['Date'], format='%B %Y', errors='coerce'
    )
    unemployment_df['Month_YYYY_MM'] = unemployment_df['Month_dt'].dt.strftime('%Y-%m')

    # Drop rows where Month_YYYY_MM is NaT (due to parsing errors)
    # and the original 'Month' column as we now have 'Month_YYYY_MM'
    unemployment_df.dropna(subset=['Month_YYYY_MM'], inplace=True)
    
    # Rename "Claimant_Rate" to "claimant_rate" for the merge (as per user correction)
    if 'Claimant_Rate' in unemployment_df.columns:
        unemployment_df.rename(columns={'Claimant_Rate': 'claimant_rate'}, inplace=True)
    elif 'claimant_rate' not in unemployment_df.columns:
        print("Warning: 'Claimant_Rate' or 'claimant_rate' column not found in unemployment data.")

    # Select only the necessary columns for merging to avoid duplicate 'Month' columns if named differently
    unemployment_to_merge = unemployment_df[['Month_YYYY_MM', 'claimant_rate']].copy()
    # Drop duplicates in case a month appears multiple times in unemployment data (e.g. different regions if not filtered)
    unemployment_to_merge.drop_duplicates(subset=['Month_YYYY_MM'], keep='first', inplace=True)


    # --- 3. Process/Verify burglary data's month column ---
    if 'Month' not in burglary_df.columns:
        print("Error: 'Month' column not found in burglary data. Please ensure it exists.")
        return

    # Convert 'Month' column to datetime objects, then format to 'YYYY-MM' string
    try:
        burglary_df['Month'] = pd.to_datetime(burglary_df['Month'], errors='coerce').dt.strftime('%Y-%m')
        # Drop rows where 'Month' became NaT/NaN after conversion if any, to avoid merge issues
        burglary_df.dropna(subset=['Month'], inplace=True)
        print(f"Burglary data 'Month' column successfully converted to 'YYYY-MM' format.")
    except Exception as e:
        print(f"Error converting burglary data 'Month' column: {e}")
        return
    
    print(f"Burglary data 'Month' column type after conversion: {burglary_df['Month'].dtype}")


    # --- 4. Merge the DataFrames ---
    # Use a left join to keep all burglary records and add unemployment rates.
    # The burglary data's month column is assumed to be 'Month' (as per typical processed data)
    # The unemployment data's processed month column is 'Month_YYYY_MM'
    merged_df = pd.merge(
        burglary_df,
        unemployment_to_merge,
        left_on='Month',         # Column in burglary_df (should be 'YYYY-MM' string)
        right_on='Month_YYYY_MM', # Column in unemployment_to_merge ('YYYY-MM' string)
        how='left'
    )

    # Drop the redundant 'Month_YYYY_MM' column from the merged DataFrame
    if 'Month_YYYY_MM' in merged_df.columns:
        merged_df.drop(columns=['Month_YYYY_MM'], inplace=True)

    print("DataFrames merged successfully.")

    # --- 5. Post-merge analysis ---
    print("\nFirst five rows of the merged DataFrame:")
    print(merged_df.head())

    missing_unemployment_months = merged_df[merged_df['claimant_rate'].isnull()]['Month'].unique()
    if len(missing_unemployment_months) > 0:
        print(f"\nMonths in burglary data that did not find a matching unemployment rate (nulls in 'claimant_rate'):")
        for month in missing_unemployment_months:
            print(f"- {month}")
    else:
        print("\nAll burglary records found a matching unemployment rate.")

    # --- 6. Save the merged result ---
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_csv_path)
        if output_dir: # Check if output_dir is not an empty string (i.e. file is in root)
            os.makedirs(output_dir, exist_ok=True)
        
        merged_df.to_csv(output_csv_path, index=False)
        print(f"\nMerged DataFrame saved to '{output_csv_path}'")
    except Exception as e:
        print(f"Error saving merged DataFrame to CSV: {e}")


if __name__ == "__main__":
    merge_burglary_unemployment_data()

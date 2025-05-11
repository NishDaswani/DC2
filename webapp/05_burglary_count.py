import pandas as pd
import os

def aggregate_burglary_counts():
    """
    Reads the merged burglary and unemployment data, aggregates burglary counts
    per LSOA per month, and saves the result.
    """
    # Define file paths
    # Assuming the script is run from the root of the burglary_analysis project
    input_csv_path = "data/processed/burglary_with_unemployment_new.csv"
    output_csv_path = "data/01_final_data/burglary_count_added_new.csv"

    # --- 1. Read the CSV file ---
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully read '{input_csv_path}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'. Please ensure the previous script ran successfully.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- 2. Verify required columns exist ---
    # Columns needed for grouping and the ones we want to keep/aggregate
    required_for_grouping = ['LSOA code', 'Month']
    cols_to_aggregate_first = ['LSOA name', 'Year', 'Population', 'area_km2', 'population_density', 'claimant_rate']
    # 'Crime type' is used for counting, but other columns could also be used if 'Crime type' is not reliable
    # or if we are counting rows directly.
    
    all_expected_cols = required_for_grouping + cols_to_aggregate_first + ['Crime type'] # Crime type is used for the count
    
    missing_cols = [col for col in all_expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the input CSV: {', '.join(missing_cols)}")
        return

    # Optional: If you want to be absolutely sure it's only burglaries.
    # Given the input file name, it's assumed to contain only burglaries.
    # If not, you would filter here:
    # df = df[df['Crime type'].str.contains('Burglary', case=False, na=False)]

    # --- 3. Group by LSOA code and Month, then aggregate ---
    print("Aggregating burglary counts per LSOA and Month...")
    
    # Define aggregations
    # For 'burglary_count', we count the occurrences (size of each group).
    # For other columns, we take the first value as they should be consistent within each group.
    agg_functions = {col: 'first' for col in cols_to_aggregate_first}
    # Add the count operation. We can count any column that is non-null for each crime incident.
    # 'Crime type' is a good candidate. The result of this count will be renamed.
    agg_functions['Crime type'] = 'count' 


    try:
        # Perform the grouping and aggregation
        # as_index=False keeps 'LSOA code' and 'Month' as columns
        aggregated_df = df.groupby(required_for_grouping, as_index=False).agg(agg_functions)
        
        # Rename the count column to 'burglary_count'
        aggregated_df.rename(columns={'Crime type': 'burglary_count'}, inplace=True)
        
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return

    print("Aggregation complete.")

    # --- 4. Print first few rows of the aggregated DataFrame ---
    print("\nFirst five rows of the aggregated DataFrame:")
    print(aggregated_df.head())

    # --- 5. Save the aggregated result ---
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_csv_path)
        if output_dir: # Check if output_dir is not an empty string (i.e. file is in root)
            os.makedirs(output_dir, exist_ok=True)
        
        aggregated_df.to_csv(output_csv_path, index=False)
        print(f"\nAggregated DataFrame saved to '{output_csv_path}'")
    except Exception as e:
        print(f"Error saving aggregated DataFrame to CSV: {e}")

if __name__ == "__main__":
    aggregate_burglary_counts()

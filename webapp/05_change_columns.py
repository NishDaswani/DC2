import pandas as pd
import os

def reorder_and_save_data():
    # Define file paths
    input_csv_path = "data/00_new/poi_added_new.csv"  # Updated input file path
    output_dir = "data/00_new"
    output_csv_path = os.path.join(output_dir, "processed_data.csv")

    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully read input file: {input_csv_path}")
        print(f"Input DataFrame columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file '{input_csv_path}': {e}")
        return

    # Columns to rename: {current_name: new_name}
    # Input file already has 'LSOA11CD'. We only need to ensure 'LSOA Name' format.
    rename_map = {}
    if 'LSOA name' in df.columns:
        rename_map['LSOA name'] = 'LSOA Name'
    else:
        print("Warning: Column 'LSOA name' not found, so it cannot be renamed to 'LSOA Name'.")

    if rename_map: # Only rename if there's something to rename
        df.rename(columns=rename_map, inplace=True)
        print(f"Columns after renaming (if any): {df.columns.tolist()}")

    # Define the desired final order of columns
    desired_columns_order = [
        'Month',
        'LSOA11CD',
        'LSOA Name', # This should be the name after potential rename
        'Year',
        'Population',
        'area_km2',
        'population_density',
        'claimant_rate',
        'poi_count',
        'burglary_count'
    ]
    
    # Filter the desired_columns_order to only include columns actually present in the DataFrame
    final_df_columns = []
    all_columns_present = True
    for col in desired_columns_order:
        if col in df.columns:
            final_df_columns.append(col)
        else:
            print(f"Error: Desired column '{col}' not found in the input DataFrame. Cannot proceed with this exact order.")
            all_columns_present = False
            
    if not all_columns_present:
        print("One or more essential columns for the desired order are missing. Please check the input file.")
        return
        
    # Create the new DataFrame with selected and ordered columns
    processed_df = df[final_df_columns].copy()

    # Sort by Month in ascending order
    # Ensure 'Month' column is in a sortable format (e.g., 'YYYY-MM' string or datetime)
    try:
        # If 'Month' is not already datetime, convert for robust sorting, then can remain string if needed
        if not pd.api.types.is_datetime64_any_dtype(processed_df['Month']):
            processed_df['Month_dt'] = pd.to_datetime(processed_df['Month'], errors='coerce')
            processed_df.sort_values(by='Month_dt', ascending=True, inplace=True)
            processed_df.drop(columns=['Month_dt'], inplace=True) # Drop the temporary datetime column
        else:
            processed_df.sort_values(by='Month', ascending=True, inplace=True)
        print("DataFrame sorted by 'Month' ascending.")
    except KeyError:
        print("Error: 'Month' column not found for sorting. Please ensure it exists.")
        return
    except Exception as e:
        print(f"Error sorting DataFrame by 'Month': {e}")
        return

    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return

    # Save the processed DataFrame
    try:
        processed_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully processed data and saved to: {output_csv_path}")
        print(f"Output DataFrame shape: {processed_df.shape}")
        print(f"Output DataFrame columns: {processed_df.columns.tolist()}")
        print("\nFirst 5 rows of the output data:")
        print(processed_df.head())
        print("\nLast 5 rows of the output data:")
        print(processed_df.tail())
    except Exception as e:
        print(f"Error saving processed data to CSV '{output_csv_path}': {e}")
        return

if __name__ == "__main__":
    reorder_and_save_data()

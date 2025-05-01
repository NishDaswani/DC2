'''
Creates the necessary folder structure for the London Burglary Prediction project
and generates a basic README.md file.
'''
import os

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories to create relative to the project root
DIRECTORIES = [
    "data",
    "models",
    "plots",
    "scripts" # Script itself lives here, but ensure it exists
]

# README.md content
README_CONTENT = """
# Predicting Residential Burglary Counts in London

## Description
This project aims to predict monthly residential burglary counts at the Lower Layer Super Output Area (LSOA) level in London, UK. 
It utilizes data from data.police.uk (burglary records), Indices of Multiple Deprivation (IMD), and census data.

The prediction model is built using XGBoost, and SHAP (SHapley Additive exPlanations) is used for model interpretation.

## Folder Structure

- **/data/**: Contains raw and processed datasets.
  - `burglary_data.csv`: Raw burglary records.
  - `imd_data.csv`: Indices of Multiple Deprivation data.
  - `census_data.csv`: Census data relevant to demographics and socio-economic factors.
  - `features_data.csv`: Combined and processed features before splitting.
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: Training and testing data splits.
- **/models/**: Contains trained machine learning models.
  - `xgboost_model.pkl`: Trained XGBoost model saved using pickle or joblib.
- **/plots/**: Contains generated plots and visualizations.
  - `shap_summary.png`: SHAP summary plot.
  - `shap_dependence.png`: Example SHAP dependence plots.
- **/scripts/**: Contains Python scripts for the project workflow.
  - `setup_folders.py`: This script, used for initial folder setup.
  - `data_fetch.py`: Script to download or fetch raw data.
  - `preprocess.py`: Script for data cleaning, feature engineering, and preprocessing.
  - `train_model.py`: Script to train the prediction model.
  - `interpret_shap.py`: Script to generate SHAP interpretations.

## Setup Instructions
The Python virtual environment (`burglary_env`) and package installation are handled by `setup_env.sh` and `install_packages.py`.
The Git repository is initialized by `setup_env.sh` and connected to: https://github.com/NishDaswani/DC2.git

To activate the environment:
```bash
source ../burglary_env/bin/activate # If in scripts directory
# or
source burglary_env/bin/activate    # If in project root
```
"""

# --- Script Logic ---
def create_directories():
    '''Creates the specified project directories.'''
    print("--- Creating project directories ---")
    for directory in DIRECTORIES:
        path = os.path.join(PROJECT_ROOT, directory)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"  Created or verified directory: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}", file=sys.stderr)
    print("Directory setup complete.")

def create_readme():
    '''Creates the README.md file in the project root.'''
    print("--- Generating README.md ---")
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    try:
        with open(readme_path, 'w') as f:
            f.write(README_CONTENT)
        print(f"  README.md created at: {readme_path}")
    except IOError as e:
        print(f"Error writing README.md: {e}", file=sys.stderr)
    print("README.md generation complete.")

if __name__ == "__main__":
    # Ensure the script's own directory exists (needed for relative path calc)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(scripts_dir):
        # This case is unlikely if running the script directly,
        # but good practice if it were called from elsewhere.
        os.makedirs(scripts_dir, exist_ok=True)

    create_directories()
    create_readme()
    print("\nFolder structure and README.md setup finished.")

    # Add placeholder files mentioned in README for clarity (optional)
    print("\n--- Creating placeholder files mentioned in README ---")
    placeholder_files = [
        "data/burglary_data.csv", "data/imd_data.csv", "data/census_data.csv",
        "data/features_data.csv", "data/X_train.csv", "data/X_test.csv",
        "data/y_train.csv", "data/y_test.csv",
        "models/xgboost_model.pkl",
        "plots/shap_summary.png", "plots/shap_dependence.png",
        "scripts/data_fetch.py", "scripts/preprocess.py",
        "scripts/train_model.py", "scripts/interpret_shap.py"
    ]
    for fname in placeholder_files:
        # Only create if it's not this script itself
        if os.path.basename(fname) != "setup_folders.py":
            fpath = os.path.join(PROJECT_ROOT, fname)
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            if not os.path.exists(fpath):
                try:
                    with open(fpath, 'w') as f:
                        f.write(f"# Placeholder for {os.path.basename(fname)}\n") # Add a comment for non-binary files
                    print(f"  Created placeholder: {fpath}")
                except IOError as e:
                    print(f"Error creating placeholder file {fpath}: {e}", file=sys.stderr)
            else:
                 print(f"  Placeholder already exists: {fpath}")
    print("Placeholder file creation complete.") 
'''
Installs required Python packages for the burglary prediction project.
'''
import subprocess
import sys

def install_packages():
    '''Installs pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn, requests.'''
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "shap",
        "matplotlib",
        "seaborn",
        "requests"
    ]
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        print("Successfully installed packages.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    install_packages() 
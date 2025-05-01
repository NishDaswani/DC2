#!/bin/bash

# Bash script to set up the project environment for the London Burglary Prediction project.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define project variables
VENV_NAME="burglary_env"
PYTHON_VERSION="python3.13"
REMOTE_REPO="https://github.com/NishDaswani/DC2.git"
PROJECT_DIR_NAME="DC2" # Assuming the script is run inside the project directory

# --- 1. Create Virtual Environment --- 
echo "--- Creating virtual environment ($VENV_NAME) using $PYTHON_VERSION ---"
if ! command -v $PYTHON_VERSION &> /dev/null
then
    echo "Error: $PYTHON_VERSION could not be found. Please ensure Python 3.13 is installed and accessible as '$PYTHON_VERSION'." 
    exit 1
fi
$PYTHON_VERSION -m venv $VENV_NAME
echo "Virtual environment created."

# --- 2. Activate Virtual Environment and Install Packages --- 
echo "--- Activating virtual environment and installing packages ---"
# Activate the environment *within this script's subshell*
source "$VENV_NAME/bin/activate"

# Install packages using the Python script
if [ -f install_packages.py ]; then
    python install_packages.py
else
    echo "Error: install_packages.py not found in the current directory." 
    # Deactivate venv before exiting
    deactivate
    exit 1
fi

# Deactivate the environment (the user will activate it manually later)
deactivate
echo "Packages installed."

# --- 3. Initialize Git Repository --- 
echo "--- Initializing Git repository ---"
if [ -d ".git" ]; then
    echo "Git repository already initialized."
else
    git init
    echo "Git repository initialized."
fi

# --- 4. Connect to Remote Repository --- 
echo "--- Connecting to remote GitHub repository ($REMOTE_REPO) ---"
if git remote | grep -q origin; then
    echo "Remote 'origin' already exists."
    # Optional: uncomment the next line to update the remote URL if it exists
    # git remote set-url origin $REMOTE_REPO 
else
    git remote add origin $REMOTE_REPO
    echo "Remote repository added."
fi

# --- 5. Create .gitignore --- 
echo "--- Creating .gitignore file ---"
cat <<EOL > .gitignore
# Ignore Python virtual environment
$VENV_NAME/

# Ignore Python cache files
__pycache__/
*.pyc

# Ignore data files (e.g., CSV)
*.csv
EOL
echo ".gitignore created."

# --- 6. Initial Commit and Push --- 
echo "--- Creating initial commit and pushing to main branch ---"
git add .gitignore
# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit (gitignore already tracked or identical)."
else
    git commit -m "Initial commit: Add .gitignore"
    echo "Initial commit created."
    # Push to the main branch (use -u to set upstream)
    # Assumes the default branch name is 'main'. Use 'master' if needed.
    git push -u origin main
    echo "Pushed initial commit to origin/main."
fi

# --- 7. Print Final Instructions --- 
echo "---------------------------------------------------"
echo "Setup complete!"
echo "---------------------------------------------------"
echo "To activate the virtual environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To verify the Git setup, run:"
echo "  git status"
echo "  git remote -v"
echo "---------------------------------------------------"

exit 0 
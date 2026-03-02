#!/usr/bin/env bash

# This script automates the setup of a Python virtual environment,
# installation of dependencies, and execution of a Python script
# with dynamically named output directories for better organization.

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

cd "$(dirname "$0")/arena_predictor"

# Name of the virtual environment directory
VENV_NAME="venv_arena"

# Name of the requirements file
REQUIREMENTS_FILE="requirements.txt"

# --- Dynamic Output Configuration ---
# Base directory for all analysis outputs
BASE_OUTPUT_DIR="analysis_output"

# Persisted CV splits for comparable sweeps
CV_SPLITS_PATH="$BASE_OUTPUT_DIR/cv_splits.json"

# Generate a timestamp for unique run identification (e.g., 20231027_153045)
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')



# Construct the full path to the output directory for this run
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

# Construct the full prefix for all output files (directory + file basename)
OUTPUT_FILE_PREFIX="${RUN_OUTPUT_DIR}/${run}"


# --- Script Logic ---
echo "--- Starting Environment Setup and Script Execution ---"

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed or not in PATH. Please install Python 3."
    exit 1
fi

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating Python virtual environment in './$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
else
    echo "Virtual environment './$VENV_NAME' already exists."
fi

# 2. Activate the virtual environment
# Note: 'source' is a shell built-in, so we reference the venv directly.
echo "Activating the virtual environment..."
source "$VENV_NAME/bin/activate"

# 3. Upgrade pip and install dependencies from requirements.txt
echo "Installing/updating dependencies from $REQUIREMENTS_FILE..."
pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"
echo "Dependencies are up to date."


# 5. Run the main Python script with the dynamic output path
echo "Executing Python script"


# python3 lmsys_predictor5.py \
#   --csv_path benchmarks/clean_combined_all_benches_transformed.csv \
#   --passes 200 \
#   --alpha 0.95 \
#   --selector_cv 5 \
#   --alt_selector_cv 5 \
#   --poly_interactions --poly_include_squares \
#   --alt_top_k_features 10 \
#   --max_workers 8 \
#   --cv_splits_path "$CV_SPLITS_PATH"


python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches.csv \
    --poly_interactions \
    --poly_include_squares \
    --alt_centric_poly --alt_centric_k 4 \
    --cv_repeats_outer 5 \
    --cv_repeats_inner 3 \
    --feature_cv_repeats 1 \
    --alt_cv_repeats 1

# python3 lmsys_predictor_pipeline.py \
#   --csv_path benchmarks/clean_combined_all_benches_transformed.csv \
#   --output_root analysis_output \
#   --selector_cv 5 \
#   --imputer_passes 12 \
#   --imputer_alpha 0.10 \
#   --eval_imputation          # write imputation quality CSVs too

# The virtual environment is automatically deactivated when the script finishes.
echo ""
echo "--- Process complete ---"

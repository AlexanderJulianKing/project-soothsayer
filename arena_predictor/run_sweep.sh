#!/usr/bin/env bash

# Usage: ./run_sweep.sh <num_cores>
# Example: ./run_sweep.sh 8

set -e

CORES="${1:-4}"
CORES_PER_RUN=4
N_JOBS=$((CORES / CORES_PER_RUN))
N_JOBS=$((N_JOBS > 0 ? N_JOBS : 1))  # At least 1 parallel job
N_TRIALS=1000
DB_PATH="sweep_optuna.db"

VENV_NAME="venv_arena"
REQUIREMENTS_FILE="requirements.txt"

# Limit BLAS/OpenMP threads to prevent CPU over-subscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== Optuna Sweep Setup and Execution ==="
echo "  Total cores: $CORES"
echo "  Cores per run: $CORES_PER_RUN"
echo "  Parallel trials: $N_JOBS"
echo ""

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

# 1. Create virtual environment if needed
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating Python virtual environment in './$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
else
    echo "Virtual environment './$VENV_NAME' already exists."
fi

# 2. Activate the virtual environment
echo "Activating the virtual environment..."
source "$VENV_NAME/bin/activate"

# 3. Install base dependencies
echo "Installing/updating dependencies from $REQUIREMENTS_FILE..."
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS_FILE" --quiet

# 4. Install sweep-specific dependencies (optuna + dashboard)
echo "Installing Optuna and dashboard..."
pip install optuna optuna-dashboard --quiet

# 5. Verify critical imports
echo "Verifying imports..."
python3 -c "import optuna; import lightgbm; import xgboost; print('All imports OK')"

echo ""
echo "=== Starting Sweep ==="
echo "  Trials: $N_TRIALS"
echo "  Parallel jobs: $N_JOBS"
echo "  Database: $DB_PATH"
echo ""
echo "To view live dashboard (in another terminal):"
echo "  source $VENV_NAME/bin/activate"
echo "  optuna-dashboard sqlite:///$DB_PATH"
echo ""

# 6. Run the sweep
python3 sweep.py \
    --n_trials "$N_TRIALS" \
    --n_jobs "$N_JOBS" \
    --db_path "$DB_PATH"

echo ""
echo "=== Sweep Complete ==="
echo "Results stored in: $DB_PATH"
echo "View with: optuna-dashboard sqlite:///$DB_PATH"

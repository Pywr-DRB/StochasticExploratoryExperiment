#!/bin/bash
#SBATCH --job-name=extract_vals
#SBATCH --output=./logs/extract_vals.out
#SBATCH --error=./logs/extract_vals.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=12:00:00

# Load modules and environment
module load python/3.11.5
source venv/bin/activate

# Configuration name (determines output directory read from outputs/<CONFIG_NAME>/...)
export CONFIG_NAME=${CONFIG_NAME:-default}

mkdir -p logs

echo "========================================"
echo "Extracting manuscript values (CONFIG_NAME=$CONFIG_NAME)"
echo "========================================"

python3 extract_manuscript_values.py

if [ $? -ne 0 ]; then
    echo "ERROR: extract_manuscript_values.py failed"
    exit 1
fi

echo ""
echo "Manuscript value extraction complete."
echo "Outputs in: StochasticExploratoryExperiment/manuscript/"

#!/bin/bash
#SBATCH --job-name=zone_probs
#SBATCH --output=./logs/zone_probs.out
#SBATCH --error=./logs/zone_probs.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00

module load python/3.11.5
source venv/bin/activate

export CONFIG_NAME=${CONFIG_NAME:-default}

mkdir -p logs

echo "Calculating storage zone probabilities for all datasets..."
python3 si_scripts/SI3_calculate_storage_zone_probabilities.py --all

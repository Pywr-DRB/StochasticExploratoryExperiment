#!/bin/bash
#SBATCH --job-name=SE-DRB
#SBATCH --output=./logs/exploratory.out
#SBATCH --error=./logs/exploratory.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


# Boolean variables to control workflow steps
RUN_BASELINE=false
GENERATE_ENSEMBLE=false
PLOT_DIAGNOSTICS=false
PREP_PYWRDRB=false
RUN_PYWRDRB=false
PLOT_OUTCOMES=true

# Ensemble generation parameters
N_REALIZATIONS=100
N_ENSEMBLE_SETS=5

module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Run baseline simulations on single task
if [ "$RUN_BASELINE" = true ]; then
    echo "Running baseline simulations..."
    time mpirun -np $np python3 00_run_baseline_simulations.py
fi

# Generate stationary ensemble
if [ "$GENERATE_ENSEMBLE" = true ]; then

    echo "Generating stationary ensemble..."
    for i in $(seq 1 $N_ENSEMBLE_SETS); do
        echo "Generating realization set $i"
        start_idx=$((($i - 1) * $N_REALIZATIONS))
        end_idx=$(($start_idx + $N_REALIZATIONS))
        
        output_file="pywrdrb/inputs/stationary_ensemble_${start_idx}_${end_idx}.hdf5"
        echo "Saving to $output_file"
        time mpirun -np $np python3 01_generate_stationary_ensemble_parallel.py $N_REALIZATIONS $output_file
    done

fi

# Plot ensemble diagnostics
if [ "$PLOT_DIAGNOSTICS" = true ]; then
    echo "Plotting ensemble diagnostics..."
    python 02_plot_ensemble_diagnostics.py
fi

# Prepare pywrdrb supporting data
if [ "$PREP_PYWRDRB" = true ]; then
    echo "Preparing pywrdrb supporting data..."
    time mpirun -np $np python3 03_prep_pywrdrb_inputs.py
fi

# Run pywrdrb simulations
if [ "$RUN_PYWRDRB" = true ]; then
    echo "Running pywrdrb simulations with stationary ensemble..."
    time mpirun -np $np python3 03_run_pywrdrb_simulations_parallel.py
fi

# Plot outcome distributions
if [ "$PLOT_OUTCOMES" = true ]; then
    echo "Plotting shortage outcome plots..."
    # python 999_calculate_hashimito_metrics.py
    python 999_plot_pywrdrb_dynamics.py
fi
import numpy as np
from methods.load import load_drb_reconstruction
from methods.moea_generator import Objectives
from methods.constraint_functions import uniform_ks_test

from sglib.methods.nonparametric.kirsch import KirschGenerator


### Load observed data #######################################
# Load DRB flow reconstruction
Q = load_drb_reconstruction()

# Work only with Montague for now
Q = Q[['delMontague']].copy()
Q_monthly = Q.resample('MS').sum()


### Initialize Kirsch generator ##############################
generator = KirschGenerator(Q_monthly)
generator.preprocessing()
generator.fit()

n_years = 20
n_years_with_buffer = n_years + 1

##### Borg Settings #########################################

NFE = 50000
NVARS = int(n_years_with_buffer * 12)  # Each variable represents a month in the n_years period
BOUNDS = [[0.0, int(np.floor(Q_monthly.shape[0]/12))]] * NVARS # limited by Q_monthly size

### Objectives
NOBJS = 4
EPSILONS = [0.01, 1.0, 1.0, 0.01]

### Borg Settings
NCONSTRS = 1
runtime_freq = 1000      # output frequency
islands = 4             # 1 = MW, >1 = MM  # Note the total NFE is islands * nfe
borg_seed = 711

### Objective functions ###############################

obj_func = Objectives(Qh=Q_monthly.squeeze())


### Evaluation function ###################################### 

# function(*vars) -> (objs, constrs)
def evaluate(*vars):
    """
    For this problem, Borg provides an array of indices which are used to resample
    the historic flow data. 
    
    These indices are used in replacement of the random sampling in the Kirsch-Nowak generator. 
    
    Parameters:
    -----------
    *vars : tuple
        A tuple of indices that Borg uses to sample the historic flow data.
    
    Returns:
    --------
    objectives : list
        A list of objective values. In this case, it is empty since there are no objectives defined.
    """
    
    # Reformat so that vars is 2d np.array with shape
    # (n_years, 12) where each row is a year and each column is a month.
    M = np.array(vars).reshape((n_years_with_buffer, 12))

    # Make sure M is integer
    M = np.floor(M).astype(int)
    M = np.clip(M, 0, 79)

    is_uniform = uniform_ks_test(M, alpha=0.1)
    if not is_uniform:
        # If the constraint is not satisfied, return a large penalty
        objectives = [1e6] * NOBJS
        return objectives, [1.0]

    # Generate a single synthetic trace
    Qs_out = generator.generate_single_series(n_years=n_years,
                                              M=M, 
                                              as_array=False)


    # Check if any -np.inf or np.inf values are present in Qs_out
    if np.any(np.isinf(Qs_out)) or np.any(np.isnan(Qs_out)):
        raise ValueError("Generated synthetic flow contains infinite or NaN values.")
    
    # check if any 0.0 values are present in Qs_out
    if np.any(Qs_out == 0.0):
        raise ValueError("Generated synthetic flow contains zero values, which is not allowed.")
    

    try:
        objectives = obj_func.value(Qh=Q_monthly.squeeze(), 
                                    Qs=Qs_out.squeeze())
            
        return objectives, [0.0]

    except Exception as e:
        
        print(f"ERROR with:\n\n")  # Debugging output
        print(f"Qs_out shape: {Qs_out.shape}, Qs_out: {Qs_out[:10]}...\n\n")
        print(f"M shape: {M.shape}, M: {M[:5]}...\n\n")


        raise RuntimeError(f"Error evaluating objectives: {e}")
    

# Pack borg settings into dict
borg_settings = {
    "numberOfVariables": NVARS,
    "numberOfObjectives": NOBJS,
    "numberOfConstraints": NCONSTRS,
    "function": evaluate,
    "epsilons": EPSILONS,
    "bounds": BOUNDS,
    "directions": None,  # default is to minimize all objectives. keep this unchanged.
    "seed": borg_seed
}


if __name__ == "__main__":

    from borg import *
    Configuration.startMPI()

    
    ##### Borg Setup ####################################################################

    borg = Borg(**borg_settings)

    ##### Parallel borg - solvempi #########################################################

    fname_base = f"syn_nfe{NFE}_seed{borg_seed}"
    
    # Runtime
    if islands == 1: # Master slave version
        runtime_filename = f"{fname_base}.runtime"
    else:
        # For MMBorg, the filename should include one %d which gets replaced by the island index
        runtime_filename = f"{fname_base}_%d.runtime"


    solvempi_settings = {
        "islands": islands,
        "maxTime": None,
        "maxEvaluations": NFE,  # Total NFE is islands * maxEvaluations if island > 1
        "initialization": None,
        "runtime": runtime_filename,
        "allEvaluations": None,
        "frequency": runtime_freq,
    }

    result = borg.solveMPI(**solvempi_settings)

    ##### Save results #####################################################################
    if result is not None:
        # The result will only be returned from one node
        with open(f"{fname_base}.csv", "w") as file:
            # You may add header here
            file.write(",".join(
                [f"var{i+1}" for i in range(NVARS)]
                + [f"obj{i+1}" for i in range(NOBJS)]
                + [f"constr{i+1}" for i in range(NCONSTRS)]
                ) + "\n")
            result.display(out=file, separator=",")

        # for MOEAFramework-5.0
        with open(f"{fname_base}.set", "w") as file:
            # You may add header here
            file.write("# Version=5\n")
            file.write(f"# NumberOfVariables={NVARS}\n")
            file.write(f"# NumberOfObjectives={NOBJS}\n")
            file.write(f"# NumberOfConstraints={NCONSTRS}\n")
            for i, bound in enumerate(borg_settings["bounds"]):
                file.write(f"# Variable.{i+1}.Definition=RealVariable({bound[0]},{bound[1]})\n")
            if borg_settings.get("directions") is None:
                for i in range(NOBJS):
                    file.write(f"# Objective.{i+1}.Definition=Minimize\n")
            else:
                for i, direction in enumerate(borg_settings["directions"]):
                    if direction == "min":
                        file.write(f"# Objective.{i+1}.Definition=Minimize\n")
                    elif direction == "max":
                        file.write(f"# Objective.{i+1}.Definition=Maximize\n")
            file.write(f"//NFE={NFE}\n") # if using check point or multi island, the NFE may not be correct.
            result.display(out=file, separator=" ")
            file.write("#\n")
        
        # Write the dictionary to a file in a readable format
        with open(f"{fname_base}.info", 'w') as file:
            file.write("\nBorg settings\n")
            file.write("=================\n")
            for key, value in borg_settings.items():
                file.write(f"{key}: {value}\n")
            file.write("\nBorg solveMPI settings\n")
            file.write("=================\n")
            for key, value in solvempi_settings.items():
                file.write(f"{key}: {value}\n")

        if islands == 1:
            print(f"Master: Completed {fname_base}")
        elif islands > 1:
            print(f"Multi-master controller: Completed {fname_base}")

    ##### End MPI #########################################################################
    Configuration.stopMPI()
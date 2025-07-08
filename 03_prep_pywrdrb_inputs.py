import os
import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers
from pywrdrb.pre import PredictedInflowEnsemblePreprocessor


ensemble_folder = "./pywrdrb/inputs/stationary_ensemble/" 
catchment_inflow_ensemble_fname = f"{ensemble_folder}catchment_inflow_mgd.hdf5"
gage_flow_ensemble_fname = f"{ensemble_folder}gage_flow_mgd.hdf5"

# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
pn_config["flows/stationary_ensemble"] = os.path.abspath(ensemble_folder)
pywrdrb.load_pn_config(pn_config)


if __name__ == "__main__":
    realization_ids = get_hdf5_realization_numbers(catchment_inflow_ensemble_fname)

    # Initialize the preprocessor
    preprocessor = PredictedInflowEnsemblePreprocessor(
        flow_type="stationary_ensemble",
        ensemble_hdf5_file=catchment_inflow_ensemble_fname,
        realization_ids=realization_ids,  
        start_date=None,  # Use full range
        end_date=None,
        modes=('regression_disagg',),
        use_log=True,
        remove_zeros=False,
        use_const=False,
        use_mpi=True
    )

    preprocessor.load()
    preprocessor.process()
    preprocessor.save()
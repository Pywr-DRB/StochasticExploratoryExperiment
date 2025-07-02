import os
import pywrdrb
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict

N_YEARS = 50
N_REALIZATIONS = 10

START_DATE = '1970-01-01'
END_DATE = '2019-12-31'
SITE_SUBSET = ['cannonsville', 'pepacton', 'neversink']

# get directory of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.abspath(f"{ROOT_DIR}/pywrdrb/outputs/")
FIG_DIR = os.path.abspath(f"{ROOT_DIR}/figures/")
ensemble_folder = os.path.abspath(f"{ROOT_DIR}/pywrdrb/inputs/stationary_ensemble/") 
catchment_inflow_ensemble_fname = os.path.abspath(f"{ensemble_folder}/catchment_inflow_mgd.hdf5")
gage_flow_ensemble_fname = os.path.abspath(f"{ensemble_folder}/gage_flow_mgd.hdf5")

RECONSTRUCTION_OUTPUT_FNAME = os.path.abspath(f"{OUTPUT_DIR}/reconstruction.hdf5")
STATIONARY_ENSEMBLE_OUTPUT_FNAME = os.path.abspath(f"{OUTPUT_DIR}/stationary_ensemble.hdf5")

# Dictionary with hierarchy of folders and files
FILES = {
    "pywrdrb" : {
        "inputs": {
            "stationary_ensemble": {
                "catchment_inflow_mgd": catchment_inflow_ensemble_fname,
                "gage_flow_mgd": gage_flow_ensemble_fname,
            },
        },
    }
}


# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
pn_config["flows/stationary_ensemble"] = os.path.abspath(ensemble_folder)


## Pywr-DRB Node Info
pywrdrb_nodes = list(immediate_downstream_nodes_dict.keys())


# A subset of nodes are generated using kirsch. 
# The others will be generated using regression.
pywrdrb_nodes_to_generate = [
    n for n in pywrdrb_nodes if n[0] != '0'
]

pywrdrb_nodes_to_generate.remove('delTrenton')


pywrdrb_nodes_to_regress = [
    n for n in pywrdrb_nodes if n[0] == '0'
]
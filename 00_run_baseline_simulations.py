import sys
import pywrdrb
from methods.mpi_utils import get_comm

import warnings
warnings.filterwarnings("ignore")

# MPI initialization (falls back to serial when mpi4py is unavailable)
comm, rank, size = get_comm()

# Flow scenarios (optionally filtered by command-line args)
model_date_ranges = pywrdrb.utils.dates.model_date_ranges
flowtypes = sys.argv[1:] if len(sys.argv) > 1 else list(model_date_ranges.keys())

# Split flowtypes among MPI ranks
rank_flowtypes = flowtypes[rank::size] if rank < len(flowtypes) else []


if __name__ == "__main__":

    for flow in rank_flowtypes:
        print(f"Rank {rank} running pywrdrb with flow type: {flow}")

        # Get simulation period
        start_date, end_date = model_date_ranges[flow]

        # Filenames
        flow_label = flow if ('pub' not in flow) else "reconstruction"
        model_filename = f"./pywrdrb/models/{flow_label}.json"
        output_filename = f"./pywrdrb/outputs/{flow_label}.hdf5"

        # Create a ModelBuilder instance with inflow data type and time period
        mb = pywrdrb.ModelBuilder(
            inflow_type=flow,
            start_date=start_date,
            end_date=end_date,
            options={"flow_prediction_mode": "perfect_foresight"},
        )
        mb.make_model()
        mb.write_model(model_filename)


        # Load the model from the saved JSON file
        model = pywrdrb.Model.load(model_filename)

        recorder = pywrdrb.OutputRecorder(
            model=model,
            output_filename=output_filename,
        )

        # Run the simulation
        stats = model.run()

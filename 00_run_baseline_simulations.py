import pywrdrb

# Set warning level to error
import warnings
warnings.filterwarnings("ignore")

# Flow scenarios
model_date_ranges = pywrdrb.utils.dates.model_date_ranges
flowtypes = list(model_date_ranges.keys())

if __name__ == "__main__":

    for  flow in flowtypes:

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
            end_date=end_date
        )
        mb.make_model()
        mb.write_model(model_filename)


        # Load the model from the saved JSON file
        model = pywrdrb.Model.load(model_filename)

        recorder = pywrdrb.OutputRecorder(
            model=model,
            output_filename=output_filename,
        )

        # Execute the simulation
        print(f"Running simulation with {flow} inflows...")
        stats = model.run()
        
    print('Done running pywrdrb simulations.')
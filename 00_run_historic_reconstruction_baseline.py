import pywrdrb

inflow_type = "reconstruction"
model_filename = f"./pywrdrb/models/{inflow_type}.json"
output_filename = f"./pywrdrb/outputs/{inflow_type}.hdf5"

if __name__ == "__main__":

    # Create a ModelBuilder instance with inflow data type and time period
    mb = pywrdrb.ModelBuilder(
        inflow_type='pub_nhmv10_BC_withObsScaled',
        start_date="1945-01-01",
        end_date="2023-12-31"
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
    print("Running model with historic reconstruction inflows...")
    stats = model.run()
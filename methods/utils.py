from pywrdrb.load import Output

def get_parameter_subset_to_export(all_parameter_names, results_set_subset):
    output_loader = Output(output_filenames=[]) # empty dataloader to use methods
    keep_keys = []
    for results_set in results_set_subset:
        if results_set == "all":
            continue
        
        keys_subset, _ = output_loader.get_keys_and_column_names_for_results_set(all_parameter_names, 
                                                                                 results_set)
        
        keep_keys.extend(keys_subset)
    return keep_keys
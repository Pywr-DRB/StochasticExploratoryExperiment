import pywrdrb

def get_parameter_subset_to_export(all_parameter_names, results_set_subset):
    output_loader = pywrdrb.load.Output(output_filenames=[]) # empty dataloader to use methods
    keep_keys = []
    for results_set in results_set_subset:
        if results_set == "all":
            continue
        
        keys_subset, _ = output_loader.get_keys_and_column_names_for_results_set(all_parameter_names, 
                                                                                 results_set)
        
        keep_keys.extend(keys_subset)
    return keep_keys

def combine_multiple_ensemble_sets_in_data(data, 
                                           results_sets,
                                           ensemble_type):
    
    for results_set in results_sets:
        all_set_results_data = {}
        full_results_set_dict = getattr(data, results_set)
        for i in range(1, 11):
            set_name = f'{ensemble_type}_ensemble_set{i}'
            set_data = full_results_set_dict[set_name]
            
            # If the keys of the set data are NOT starting at (100 * (set_number -1))
            # then we need to update the keys to be starting at (100*(set_number-1))
            
            start_realization = list(set_data.keys())[0]
            if start_realization != 100*(i-1):
                print(f'Set number {i} has realization ids starting at {start_realization}.\n Modifying to start at {100*(i-1)}.')
                # Update keys to be starting at 100*(i-1)
                # This is to ensure that the realization ids are consistent across sets
                set_data_with_reals = {100*(i-1) + k: v for k, v in set_data.items()}
            else:
                set_data_with_reals = set_data
            all_set_results_data.update(set_data_with_reals)

        full_results_set_dict[f'{ensemble_type}_ensemble'] = all_set_results_data
        setattr(data, results_set, full_results_set_dict)
    return data

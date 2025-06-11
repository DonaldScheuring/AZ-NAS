from tss_general import *

def load_saved_results():
    output_dir = os.path.join(xargs.save_dir, "figs")
    filepath = os.path.join(output_dir,"results_dictionary.npz")
    loaded_npz_file = np.load(filepath)
    logger.log(f"Loaded results from: {filepath}")
    logger.log(f"Keys in results dictionary: {loaded_npz_file.files}")
    
    # --- 3. Re-pack into a standard Python dictionary ---
    reconstructed_dict = {}
    for key in loaded_npz_file.files:
        # Access the array associated with the key and convert it to a regular Python object
        # if it's a scalar or simple type that was saved as a 0-D array.
        # For example, np.array(0.95) will load as np.array(0.95), so [()] extracts the scalar.
        # For strings or other objects saved with allow_pickle=True, [()] is also useful.
        value = loaded_npz_file[key]
        if value.shape == (): # Check if it's a 0-D array (scalar)
            reconstructed_dict[key] = value.item() # .item() extracts the scalar value
        elif value.dtype == 'O': # Check if it's an object array (often for strings, lists, etc.)
            reconstructed_dict[key] = value.item() # .item() extracts the object
        else:
            reconstructed_dict[key] = value # Otherwise, keep it as a NumPy array

    # --- 4. Close the NpzFile object ---
    loaded_npz_file.close()

    # --- 5. Verify the reconstructed dictionary ---
    # print(f"\nReconstructed dictionary:")
    # print(reconstructed_dict)
    print(f"Type of reconstructed object: {type(reconstructed_dict)}")

    return reconstructed_dict


results = load_saved_results()

get_proxy_scatter_plots(results=results)
make_correlation_matrix(results=results)
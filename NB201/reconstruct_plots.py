from tss_general import *
from proxies import *
from aggregators import az_aggregator, tenas_aggregator, geometric_mean

dictionary_dir = "ProxyResultDicts_1000Archs"

def load_saved_results(filename):

    filepath = os.path.join(dictionary_dir,filename)
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

def ensembles(results):
    ensemble_results = {}
    for ensemble_name, proxy_list in EnsembleProxies.items():
        # "zero": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.TRAINABILITY_AZ, Proxy.FLOPS]
        # collect the necessary proxy values for the ensemble
        dictionary = {}
        for proxy in proxy_list:
            dictionary[proxy.value] = results[proxy.value]

        # aggregate in however many ways you like
        #ensemble_results[f"{ensemble_name}_geo"] = geometric_mean(dictionary)
        ensemble_results[f"{ensemble_name}_az"] = az_aggregator(dictionary)

    # Always put in ground_truth and baseline
    ensemble_results["accuracy"] = results["accuracy"]
    ensemble_results["AZ-NAS"] = results["AZ-NAS"]
    print(f"Ensembles: {ensemble_results.keys()}")
    return ensemble_results

def main():

    # Load saved dictionary
    results = load_saved_results("Cifar10_dictionary.npz")

    results = ensembles(results)

    # # NOTE: This is for determining the best arch based on AZ-NAS
    # for proxy_name, proxy_rankings in results.items():

    #     best_idx = np.argmax(proxy_rankings)
    #     best_arch, acc = archs[best_idx], api_valid_accs[best_idx]
    #     if api is not None:
    #         print("{:}".format(api.query_by_arch(best_arch, "200")))

    # Make plots from results
    get_proxy_scatter_plots(results=results)
    make_correlation_matrix(results=results)

if __name__ == "__main__":
    main()
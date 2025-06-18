import numpy as np
import os

def count_invalids_in_npz(npz_path):
    loaded = np.load(npz_path, allow_pickle=True)

    total_nans = total_pos_infs = total_neg_infs = 0

    print(f"Checking file: {npz_path}")
    for key in loaded.files:
        array = loaded[key]

        # Flatten in case of multidimensional arrays
        array = np.array(array).flatten()

        n_nans = np.sum(np.isnan(array))
        n_pos_infs = np.sum(array == np.inf)
        n_neg_infs = np.sum(array == -np.inf)

        total_nans += n_nans
        total_pos_infs += n_pos_infs
        total_neg_infs += n_neg_infs

        print(f"Key: {key}")
        print(f"  NaNs:     {n_nans}")
        print(f"  +Inf:     {n_pos_infs}")
        print(f"  -Inf:     {n_neg_infs}")
        print("-" * 40)

    print("=== TOTALS ===")
    print(f"NaNs:   {total_nans}")
    print(f"+Inf:   {total_pos_infs}")
    print(f"-Inf:   {total_neg_infs}")

# Example usage
count_invalids_in_npz("./results/Experiment_2_Epoch_Test_20250618_152018/Proxy_Scores_Dictionary.npz")

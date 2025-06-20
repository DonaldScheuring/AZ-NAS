import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
from collections import defaultdict
from ensemble_tests import az_aggregator, tenas_aggregator
from proxies import EnsembleProxies

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
from collections import defaultdict

EXPERIMENTS_ROOT = "./results_new/Raw_Data"
SAVE_DIR = "./results_new/Combined_Results"
os.makedirs(SAVE_DIR, exist_ok=True)


# === Utility functions ===

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return {k: data[k].item() if data[k].shape == () or data[k].dtype == 'O' else data[k] for k in data.files}

def merge_architectures(paths):
    merged = []
    seen = set()
    for path in paths:
        data = load_json(path)
        for arch in data:
            key = json.dumps(arch, sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged.append(arch)
    return merged

def merge_dicts(dicts):
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].append(v)
    return merged

def aggregate_summary(summaries):
    keys = summaries[0].keys()
    avg_summary = {}
    for k in keys:
        entries = [s[k] for s in summaries if k in s]
        avg_summary[k] = {
            "avg_time_ms": np.mean([e["avg_time_ms"] for e in entries]),
            "avg_mem_GB": np.mean([e["avg_mem_GB"] for e in entries]),
            "max_mem_GB": np.mean([e["max_mem_GB"] for e in entries]),
        }
    return avg_summary


# === Ensemble Aggregators ===

def compute_ensemble_proxy(proxy_dict, members, method="az"):
    results = {}
    for p in members:
        k = p.value
        if k in proxy_dict:
            results[k] = proxy_dict[k]
    if method == "az":
        return az_aggregator(results)
    elif method == "tenas":
        return tenas_aggregator(results)
    else:
        raise ValueError("Unknown ensemble method")


# === Metrics Computation ===

def compute_metrics_table(proxy_dict, acc_dicts, datasets):
    table = []
    for proxy in proxy_dict:
        row = [proxy]
        for d in datasets:
            proxy_scores = proxy_dict[proxy]
            acc_scores = acc_dicts[d]
            kt = stats.kendalltau(proxy_scores, acc_scores)[0]
            spr = stats.spearmanr(proxy_scores, acc_scores)[0]
            top1_idx = np.argmax(proxy_scores)
            top1_acc = acc_scores[top1_idx]
            row.extend([kt, spr, top1_acc])

        table.append(row)
    return table

def aggregate_tables(tables, datasets):
    num_tables = len(tables)
    final_table = []
    for rows in zip(*tables):
        proxy = rows[0][0]
        kt_vals = np.array([r[1::3] for r in rows])
        spr_vals = np.array([r[2::3] for r in rows])
        acc_vals = np.array([r[3::3] for r in rows])
        row = [proxy]
        for i in range(len(datasets)):
            row.extend([
                f"{kt_vals[:, i].mean():.3f}",
                f"{spr_vals[:, i].mean():.3f}",
                f"{acc_vals[:, i].mean():.3f}",
                f"{acc_vals[:, i].std():.3f}",
            ])
        final_table.append(row)
    return final_table


# === Final Table Generation ===

def generate_metrics_table(tables, datasets, avg_summary, acc_dicts_all):
    header = ["Proxy"] + sum([[f"{d}_KT", f"{d}_SPR", f"{d}_Top1_ACC", f"{d}_STD"] for d in datasets], []) + ["Avg Runtime (ms)", "Avg Memory (GB)", "Max Memory (GB)"]
    final_table = aggregate_tables(tables, datasets)

    for row in final_table:
        proxy = row[0]
        runtime = avg_summary.get(proxy, {}).get("avg_time_ms", 0)
        avg_mem = avg_summary.get(proxy, {}).get("avg_mem_GB", 0)
        max_mem = avg_summary.get(proxy, {}).get("max_mem_GB", 0)
        row.extend([f"{runtime:.1f}", f"{avg_mem:.2f}", f"{max_mem:.2f}"])

    # === Add Ground Truth row ===
    gt_row = ["Ground Truth"]
    for d in datasets:
        top1_accs = [np.max(exp[d]) for exp in acc_dicts_all]
        gt_row.extend([
            "", "",  # KT, SPR not applicable
            f"{np.mean(top1_accs):.3f}",
            f"{np.std(top1_accs):.3f}"
        ])
    gt_row.extend(["", "", ""])  # runtime, mem, max_mem
    final_table.append(gt_row)

    df = pd.DataFrame(final_table, columns=header)
    df.to_csv(os.path.join(SAVE_DIR, "Table1_Merged.csv"), index=False)
    print(df.head())


# === Main ===

def main():
    datasets = ["cifar10", "cifar100", "ImageNet16120"]
    experiment_dirs = [os.path.join(EXPERIMENTS_ROOT, d) for d in os.listdir(EXPERIMENTS_ROOT) if d.startswith("Seed_")]

    arch_paths = [os.path.join(d, "architectures.json") for d in experiment_dirs]
    merged_archs = merge_architectures(arch_paths)
    with open(os.path.join(SAVE_DIR, "merged_architectures.json"), "w") as f:
        json.dump(merged_archs, f)

    acc_dicts_all = []
    proxy_dicts_all = []
    summary_dicts = []
    per_experiment_tables = []

    for exp_dir in experiment_dirs:
        acc_dicts = {}
        proxy_dict = {}

        for dataset in datasets:
            acc_path = os.path.join(exp_dir, f"{dataset}_dictionary.npz")
            acc_data = load_npz(acc_path)

            # Load accuracy data
            acc_dicts[dataset] = acc_data["accuracy"]

            # Overwrite or add FLOPs and Params as dataset-specific proxy features
            if "FLOPs" in acc_data:
                proxy_dict["FLOPs"] = acc_data["FLOPs"]
            if "params" in acc_data:
                proxy_dict["params"] = acc_data["params"]

        # Load the base proxy scores
        proxy_path = os.path.join(exp_dir, "Proxy_Scores_Dictionary.npz")
        proxy_data = load_npz(proxy_path)
        proxy_dict.update(proxy_data)  # Combine base proxies with FLOPs/params

        # Add ensemble proxies
        for name, members in EnsembleProxies.items():
            method = "az" if name == "aznas" else "tenas"
            proxy_dict[name] = compute_ensemble_proxy(proxy_dict, members, method=method)

        proxy_dicts_all.append(proxy_dict)
        acc_dicts_all.append(acc_dicts)

        summary_path = os.path.join(exp_dir, "proxy_performance_summary.json")
        summary_dicts.append(load_json(summary_path))

        table = compute_metrics_table(proxy_dict, acc_dicts, datasets)
        per_experiment_tables.append(table)

    avg_summary = aggregate_summary(summary_dicts)
    with open(os.path.join(SAVE_DIR, "proxy_performance_summary_avg.json"), "w") as f:
        json.dump(avg_summary, f, indent=2)


        # === Final Aggregation of Dataset Dictionaries and Proxy Scores ===

    merged_data_by_dataset = defaultdict(lambda: defaultdict(list))  # dataset -> key -> list of arrays
    merged_proxy_scores = defaultdict(list)  # key -> list of arrays

    for exp_dir in experiment_dirs:
        for dataset in datasets:
            # Load accuracy + metadata (accuracy, FLOPs, params, etc.)
            acc_path = os.path.join(exp_dir, f"{dataset}_dictionary.npz")
            acc_data = load_npz(acc_path)
            for k, v in acc_data.items():
                merged_data_by_dataset[dataset][k].append(v)

        # Load proxy scores (same for all datasets)
        proxy_path = os.path.join(exp_dir, "Proxy_Scores_Dictionary.npz")
        proxy_data = load_npz(proxy_path)
        for k, v in proxy_data.items():
            merged_proxy_scores[k].append(v)

    # === Save merged dataset dictionaries ===

    dataset_file_map = {
        "cifar10": "Cifar10_dictionary.npz",
        "cifar100": "Cifar100_dictionary.npz",
        "ImageNet16120": "ImageNet_dictionary.npz"
    }

    for dataset, file_name in dataset_file_map.items():
        merged = {k: np.concatenate(v_list) for k, v_list in merged_data_by_dataset[dataset].items()}
        np.savez(os.path.join(SAVE_DIR, file_name), **merged)
        print(f"Saved merged data for {dataset} to {file_name}")

    # === Save merged proxy score dictionary ===

    merged_proxies = {k: np.concatenate(v_list) for k, v_list in merged_proxy_scores.items()}
    np.savez(os.path.join(SAVE_DIR, "Proxy_Scores_Dictionary.npz"), **merged_proxies)
    print("Saved merged Proxy_Scores_Dictionary.npz")



    generate_metrics_table(per_experiment_tables, datasets, avg_summary, acc_dicts_all)



if __name__ == "__main__":
    main()

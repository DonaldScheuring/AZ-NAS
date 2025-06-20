import os
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats

# Global paths
EXPERIMENTS_DIR = "./results_new/Combined_Results"
PROXY_FILEPATH = os.path.join(EXPERIMENTS_DIR, "Proxy_Scores_Dictionary.npz")
#SUMMARY_FILEPATH = os.path.join(EXPERIMENTS_DIR, "proxy_performance_summary.json")
SAVE_DIR = os.path.join(EXPERIMENTS_DIR, "FirstOrderProxyResults")


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return {k: data[k].item() if data[k].shape == () or data[k].dtype == 'O' else data[k] for k in data.files}

def rank_and_correlate(predictions, ground_truth):
    x = stats.rankdata(predictions)
    y = stats.rankdata(ground_truth)
    kt = stats.kendalltau(x, y)[0]
    spr = stats.spearmanr(x, y)[0]
    acc = (np.argsort(-predictions)[:1] == np.argsort(-ground_truth)[:1]).astype(int)
    return kt, spr, acc.mean(), acc.std()

def make_scatter_plot(x, y, title, save_path):
    kendall_corr, _ = stats.kendalltau(x, y)
    pearson_corr, _ = stats.pearsonr(x, y)

    # Append correlation values to the title
    full_title = f"{title}\nKendall τ = {kendall_corr:.3f}, Pearson r = {pearson_corr:.3f}"

    plt.figure(figsize=(7, 5))
    plt.grid(True, alpha=0.3)
    plt.scatter(x, y, c=x, cmap='viridis_r', linewidths=0.1)
    plt.xlabel("Predicted Ranking")
    plt.ylabel("Ground Truth Ranking")
    plt.title(full_title)
    plt.colorbar(label="Predicted Rank")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# def make_correlation_matrix(results, title, save_path):
#     keys = list(results.keys())
#     matrix = np.zeros((len(keys), len(keys)))
#     for i, k1 in enumerate(keys):
#         for j, k2 in enumerate(keys):
#             matrix[i, j] = stats.kendalltau(stats.rankdata(results[k1]), stats.rankdata(results[k2]))[0]
#     df = pd.DataFrame(matrix, index=keys, columns=keys)
#     plt.figure(figsize=(10, 10))
#     sn.heatmap(df, annot=True, fmt=".2f", cmap='GnBu', square=True, linewidths=0.5)
#     plt.title(title)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300)
#     plt.close()

def make_correlation_matrix(results, title, save_path):
    keys = list(results.keys())
    matrix = np.zeros((len(keys), len(keys)))

    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            matrix[i, j] = stats.kendalltau(stats.rankdata(results[k1]), stats.rankdata(results[k2]))[0]

    df = pd.DataFrame(matrix, index=keys, columns=keys)

    plt.figure(figsize=(10, 10))
    ax = sn.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap='GnBu',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.85, "label": "Kendall's Tau"}
    )

    # Axis label formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Title formatting
    plt.title(f"Kendall's Tau Correlation Matrix: {title}", fontsize=16, pad=15)

    # Layout and save
    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def make_bar_chart(proxy_names, kendall_scores, pearson_scores, save_path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(proxy_names))
    width = 0.35
    plt.bar(x - width/2, kendall_scores, width, label="Kendall", color='skyblue')
    plt.bar(x + width/2, pearson_scores, width, label="Pearson", color='salmon')
    plt.xticks(x, proxy_names, rotation=45, ha="right")
    plt.ylabel("Correlation Coefficient")
    plt.title("Proxy vs Accuracy Correlation")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# def generate_metrics_table(datasets, proxy_scores, accuracy_dict, perf_summary):
#     rows = []
#     for proxy in proxy_scores:
#         row = [proxy]
#         for dataset in datasets:
#             kt, spr, acc, _ = rank_and_correlate(proxy_scores[proxy], accuracy_dict[dataset])
#             row.extend([f"{kt:.3f}", f"{spr:.3f}", f"{acc:.3f}"])
#         runtime = perf_summary.get(proxy, {}).get("avg_time_ms", 0)
#         avg_mem = perf_summary.get(proxy, {}).get("avg_mem_GB", 0)
#         max_mem = perf_summary.get(proxy, {}).get("max_mem_GB", 0)
#         row.extend([f"{runtime:.1f}", f"{avg_mem:.2f}", f"{max_mem:.2f}"])
#         rows.append(row)

#     cols = ["Proxy"] + sum([[f"{d}_KT", f"{d}_SPR", f"{d}_ACC"] for d in datasets], []) + ["Avg Runtime (ms)", "Avg Memory (GB)", "Max Memory (GB)"]
#     df = pd.DataFrame(rows, columns=cols)
#     df.to_csv(os.path.join(SAVE_DIR, "Table1_Reproduction.csv"), index=False)
#     print(df.head())

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    datasets = ["Cifar10", "Cifar100", "ImageNet"]
    dataset_files = {d: f"{d}_dictionary.npz" for d in datasets}

    accuracy_dict = {d: load_npz(os.path.join(EXPERIMENTS_DIR, f))["accuracy"] for d, f in dataset_files.items()}
    proxy_scores = load_npz(PROXY_FILEPATH)
    #perf_summary = load_json(SUMMARY_FILEPATH)

    # For each dataset
    for d in datasets:
        results = {k: v for k, v in proxy_scores.items() if isinstance(v, np.ndarray)}
        results["accuracy"] = accuracy_dict[d]

        matrix_path = os.path.join(SAVE_DIR, f"{d}_correlation_matrix.png")
        make_correlation_matrix(results, f"{d}", matrix_path)

        # Bar chart
        kts, prs, labels = [], [], []
        for k in proxy_scores:
            kt, _, _, _ = rank_and_correlate(proxy_scores[k], accuracy_dict[d])
            pr = stats.pearsonr(stats.rankdata(proxy_scores[k]), stats.rankdata(accuracy_dict[d]))[0]
            kts.append(kt)
            prs.append(pr)
            labels.append(k)
        bar_path = os.path.join(SAVE_DIR, f"{d}_bar_chart.png")
        make_bar_chart(labels, kts, prs, bar_path)

        # Scatter plots
        for k in proxy_scores:
            x = stats.rankdata(proxy_scores[k])
            y = stats.rankdata(accuracy_dict[d])
            scatter_path = os.path.join(SAVE_DIR, "scatter_plots", f"{d}_{k}.png")
            make_scatter_plot(x, y, f"{d} - {k}", scatter_path)

    # generate_metrics_table(datasets, proxy_scores, accuracy_dict, perf_summary)


if __name__ == "__main__":
    main()

"""
to run for example:
python visualize_proxies.py --logdir AZ_NAS_flops450M-searchbs64-pop500-iter1e5-123


  'complexity','expressivity','flops','gradnorm','naswot',
  'params','progressivity','random','syncflow','trainability','zen','zico'

"""

import os
import re
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


PROXY_ORDER = [
    "complexity","expressivity","flops","gradnorm","naswot",
    "params","progressivity","random","syncflow","trainability","zen","zico"
]

def parse_iteration(fname):
    match = re.findall(r'_(\d+)\.npy$', os.path.basename(fname))
    return int(match[0]) if match else -1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, required=True, help="Directory containing S_{iter}.npy correlation logs.")
    parser.add_argument('--convergence', type=str, default="", help="Optional path to .npy with shape >= (T,) storing a convergence metric.")
    parser.add_argument('--annotate-thresh', type=float, default=0.1, help="Threshold for labeling big correlation jumps in the rate-of-change plot.")
    
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, 'figs'), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, 'figs/correlations'), exist_ok=True)

    s_files = sorted(glob.glob(os.path.join(args.logdir, 'S_*.npy')), key=lambda f: parse_iteration(f))
    if len(s_files) == 0:
        print("No S_*.npy files found in", args.logdir)
        return

    iters = []
    S_list = []
    for sf in s_files:
        it = parse_iteration(sf)
        if it < 0:
            continue
        mat = np.load(sf)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            continue
        iters.append(it)
        S_list.append(mat)
    if len(iters) == 0:
        print("no valid correlation files loaded")
        return

    iters = np.array(iters)
    S_array = np.stack(S_list, axis=0)
    idx_sort = np.argsort(iters)
    iters = iters[idx_sort]
    S_array = S_array[idx_sort]

    p = S_array.shape[1]
    if p > len(PROXY_ORDER):
        print(f"correlation has p={p} > len(PROXY_ORDER)={len(PROXY_ORDER)}.\n")
        extra_count = p - len(PROXY_ORDER)
        proxy_names = PROXY_ORDER + [f"col_{i}" for i in range(extra_count)]
    else:
        proxy_names = PROXY_ORDER[:p]

    print(f"found T={len(iters)} timepoints, p={p} proxies => {proxy_names}")

    converge_vals = None
    if args.convergence and os.path.isfile(args.convergence):
        try:
            converge_vals = np.load(args.convergence)
            print(f"loaded converge_vals of shape {converge_vals.shape}")
        except Exception as e:
            print(f"failed to load convergence file: {e}")

    if len(S_array) > 1:
        S_diff = S_array[1:] - S_array[:-1]
        diff_iters = iters[1:]
    else:
        S_diff = None
        diff_iters = None


    #spearman correlation heatmaps
    s_files = sorted(glob.glob(os.path.join(args.logdir, 'S_*.npy')), key=lambda f: parse_iteration(f))

    for fpath in s_files:
        it = parse_iteration(fpath)
        corr_mat = np.load(fpath)
        if corr_mat.ndim == 0:
            continue

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_mat, vmin=-1, vmax=1, cmap='coolwarm', square=True,
                    cbar_kws={'shrink':0.7})
        plt.title(f'Spearman correlation @ iteration={it}')
        plt.tight_layout()
        outname = os.path.join(args.logdir, 'figs/correlations', f'spearman_heatmap_{it:04d}.png')
        plt.savefig(outname, dpi=120)
        plt.close()

    #fig 1: pairwise correlations
    fig, ax1 = plt.subplots(figsize=(9,6))
    for i in range(p):
        for j in range(i+1, p):
            label = f"{proxy_names[i]} vs {proxy_names[j]}"
            yvals = S_array[:, i, j]
            ax1.plot(iters, yvals, label=label, alpha=0.7)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Spearman Correlation")
    ax1.set_title("Pairwise Proxy Correlations Over Time")

    if converge_vals is not None and len(converge_vals) >= len(iters):
        ax2 = ax1.twinx()
        cvals = converge_vals[:len(iters)]
        ax2.plot(iters, cvals, 'k--', alpha=0.6, label="Convergence")
        ax2.set_ylabel("Convergence measure")

    ax1.legend(loc='upper left', bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=8)
    fig.tight_layout()
    plt.savefig(os.path.join(args.logdir, "figs" ,"pairwise_correlation_lineplot.png"), dpi=120)
    plt.close()
    print("[info] Wrote pairwise_correlation_lineplot.png")

    #fig 2: correlation rate of change over time
    if S_diff is not None:
        fig, ax1 = plt.subplots(figsize=(9,6))
        for i in range(p):
            for j in range(i+1, p):
                label = f"{proxy_names[i]} vs {proxy_names[j]}"
                yvals = S_diff[:, i, j] 
                ax1.plot(diff_iters, yvals, label=label, alpha=0.7)

                for idx in range(len(yvals)):
                    if abs(yvals[idx]) >= args.annotate_thresh:
                        ax1.annotate(f"{yvals[idx]:.2f}",
                                     (diff_iters[idx], yvals[idx]),
                                     textcoords="offset points", xytext=(0,6),
                                     ha='center', fontsize=7, color='red')

        ax1.axhline(0.0, color='gray', linestyle='--')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("ΔCorr (t+1 - t)")
        ax1.set_title("Rate of Change in Pairwise Correlations")

        if converge_vals is not None and len(converge_vals) >= len(diff_iters):
            ax2 = ax1.twinx()
            cvals = converge_vals[:len(diff_iters)]
            ax2.plot(diff_iters, cvals, 'k--', alpha=0.6, label="Convergence")
            ax2.set_ylabel("Convergence measure")

        ax1.legend(loc='upper left', bbox_to_anchor=(1.04,1), fontsize=8)
        fig.tight_layout()
        plt.savefig(os.path.join(args.logdir, "figs", "pairwise_corr_diff_lineplot.png"), dpi=120)
        plt.close()
        print("wrote pairwise_corr_diff_lineplot.png")

    #fig 3: final iteration correlation heatmap
    final_it = iters[-1]
    final_S = S_array[-1]
    fig, ax = plt.subplots(figsize=(p+2, p+1))
    sns.heatmap(final_S, vmin=-1, vmax=1, cmap='coolwarm', square=True,
                annot=True, fmt=".2f", ax=ax,
                xticklabels=proxy_names, yticklabels=proxy_names)
    ax.set_title(f"Final Correlation Matrix (iteration={final_it})")
    plt.tight_layout()
    outname = os.path.join(args.logdir, "figs", f"final_corr_heatmap_{final_it}.png")
    plt.savefig(outname, dpi=120)
    plt.close()
    print(f"[info] Wrote final_corr_heatmap_{final_it}.png")

    #fig4: top 10 pairs with largest correlation diff
    if len(S_array) >= 2:
        first_S = S_array[0]
        last_S = S_array[-1]
        pair_deltas = []
        for i in range(p):
            for j in range(i+1, p):
                start_val = first_S[i, j]
                end_val   = last_S[i, j]
                diff_val  = abs(end_val - start_val)
                pair_deltas.append((diff_val, i, j))
        pair_deltas.sort(key=lambda x: x[0], reverse=True)
        top_10 = pair_deltas[:10]

        fig, ax1 = plt.subplots(figsize=(9,6))
        for (diff_val, i, j) in top_10:
            label = f"{proxy_names[i]} vs {proxy_names[j]} (Δ={diff_val:.2f})"
            ax1.plot(iters, S_array[:, i, j], label=label, alpha=0.9, linewidth=1.8)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Spearman Correlation")
        ax1.set_title("Top 10 Pairs with Largest ΔCorr from Iter0 → IterFinal")

        if converge_vals is not None and len(converge_vals) >= len(iters):
            ax2 = ax1.twinx()
            cvals = converge_vals[:len(iters)]
            ax2.plot(iters, cvals, 'k--', alpha=0.6, label="Convergence")
            ax2.set_ylabel("Convergence measure")

        ax1.legend(loc='upper left', bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=8)
        fig.tight_layout()
        outpath = os.path.join(args.logdir, "figs", "largest_diff_correlations.png")
        plt.savefig(outpath, dpi=120)
        plt.close()
        print("wrote largest_diff_correlations.png")

    #fig 5: mean correlation of each proxy with all others
    mean_corr = np.zeros((len(S_array), p))
    for t in range(len(S_array)):
        for i in range(p):
            row_sum = np.sum(S_array[t, i, :]) - S_array[t, i, i] 
            mean_corr[t, i] = row_sum / (p - 1)

    cmap = cm.get_cmap('tab20', max(p,20))

    fig, ax1 = plt.subplots(figsize=(9,6))
    for i in range(p):
        #skip zico in the final figure; zico doesnt work rn cause no convergence curve as input (since no training rn)
        if proxy_names[i] == "zico" or proxy_names[i] ==  "random":
            continue
        color = cmap(i)
        label = f"{proxy_names[i]}"
        ax1.plot(iters, mean_corr[:, i], label=label, alpha=0.9, linewidth=2.0, color=color)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Corr (with other proxies)")
    ax1.set_title("Mean Correlation of Each Proxy vs. Others Over Time")

    if converge_vals is not None and len(converge_vals) >= len(iters):
        ax2 = ax1.twinx()
        cvals = converge_vals[:len(iters)]
        ax2.plot(iters, cvals, 'k--', alpha=0.6, label="Convergence")
        ax2.set_ylabel("Convergence measure")

    ax1.legend(loc='upper left', bbox_to_anchor=(1.04,1), fontsize=8)
    fig.tight_layout()
    outpath = os.path.join(args.logdir, "figs", "mean_proxy_correlation_over_time.png")
    plt.savefig(outpath, dpi=120)
    plt.close()
    print("wrote mean_proxy_correlation_over_time.png")

    print("done")

if __name__ == "__main__":
    main()



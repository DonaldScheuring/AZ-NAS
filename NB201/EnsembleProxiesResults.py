import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from enum import Enum
import os
from FirstOrderProxiesCorrelations import make_correlation_matrix, make_scatter_plot
from proxies import *

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





# ==== AGGREGATORS ====

def az_aggregator_base(results, base=np.e):
    """
    Aggregates proxy rankings using a log-based approach.
    `base` controls the log base (e.g., 2, 10, np.e).
    """
    rank_agg = None
    for k in results:
        norm_ranks = stats.rankdata(results[k]) / len(results[k])
        log_ranks = np.log(norm_ranks) / np.log(base)  # change-of-base
        rank_agg = log_ranks if rank_agg is None else rank_agg + log_ranks
    return rank_agg

def tenas_aggregator(results):
    rank_agg = None
    for k in results.keys():
        if rank_agg is None:
            rank_agg = stats.rankdata(results[k])
        else:
            rank_agg = rank_agg + stats.rankdata(results[k]) # NOTE: how does adding ranks work?
    return rank_agg

def az_aggregator(results, alpha=None):

    print(f"az_aggregator keys results keys: {results.keys()}")
    rank_agg = None
    for k in results.keys():    
        if rank_agg is None:
            rank_agg = np.log( stats.rankdata(results[k]) / len(results[k]))
        else:
            rank_agg = rank_agg + np.log( stats.rankdata(results[k]) / len(results[k]))
    return rank_agg

def az_aggregator_exp(results, alpha=1.0):
    rank_agg = None
    for k in results:
        norm_ranks = stats.rankdata(results[k]) / len(results[k])
        weighted_ranks = norm_ranks ** alpha
        rank_agg = weighted_ranks if rank_agg is None else rank_agg + weighted_ranks
    return rank_agg

def az_aggregator_log(results, alpha=1.0):
    rank_agg = None
    for k in results:
        norm_ranks = stats.rankdata(results[k]) / len(results[k])
        weighted_log = np.log(norm_ranks) * alpha
        rank_agg = weighted_log if rank_agg is None else rank_agg + weighted_log
    return rank_agg

def geometric_mean_param(results: dict, alpha: float = 1.0):
    n_proxies = len(results)
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        norm = (m - stats.rankdata(results[k])) / m
        norm = norm ** alpha
        rank_agg = norm if rank_agg is None else rank_agg * norm
    rank_agg = rank_agg ** (1 / n_proxies)
    return -rank_agg

def geometric_mean_param_rank_sensitive(results: dict, alpha: float = 1.0):
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        norm = (m - stats.rankdata(results[k])) / m  # Higher is better
        rank_agg = norm if rank_agg is None else rank_agg * norm
    return -rank_agg ** alpha  # Raise entire product to alpha (outside), NOT just normalization

def softmax_aggregator(results: dict, alpha: float = 1.0):
    """
    Softmax-style transformation over normalized ranks.
    Lower rank (better) → higher weight.
    Alpha controls sharpness (higher alpha = stricter).
    """
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        # Normalize to [0, 1]
        norm_rank = stats.rankdata(results[k]) / m
        # Invert: better ranks get larger values
        transformed = np.exp(-alpha * norm_rank)
        rank_agg = transformed if rank_agg is None else rank_agg + transformed
    return -rank_agg  # Lower is better

def power_aggregator(results: dict, alpha: float = 1.0):
    """
    Raise inverted normalized rank to a power.
    Encourages sharp penalty on bad ranks as alpha ↑.
    """
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        inverted_rank = (m - stats.rankdata(results[k])) / m
        transformed = inverted_rank ** alpha
        rank_agg = transformed if rank_agg is None else rank_agg + transformed
    return -rank_agg

def sigmoid_aggregator(results, alpha=10):
    """
    Applies sigmoid to inverted normalized ranks.
    Alpha controls steepness of the cutoff.
    """
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        norm = (m - stats.rankdata(results[k])) / m  # Higher = better
        transformed = 1 / (1 + np.exp(-alpha * (norm - 0.5)))  # Midpoint at 0.5
        rank_agg = transformed if rank_agg is None else rank_agg + transformed
    return -rank_agg

def exp_decay_aggregator(results, alpha=5.0):
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        norm = stats.rankdata(results[k]) / m  # Lower = better, verbose=False

def rectifier_aggregator(results, threshold=0.2):
    """
    Assigns 1 to architectures above a rank threshold, 0 otherwise.
    You can sweep threshold from 0.05 to 0.5.
    """
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        norm = (m - stats.rankdata(results[k])) / m
        transformed = (norm > threshold).astype(float)
        rank_agg = transformed if rank_agg is None else rank_agg + transformed
    return -rank_agg

def entropy_aggregator(results, alpha=1.0):
    """
    Uses entropy-normalized weights from ranks.
    Lower entropy → more focused aggregation.
    """
    m = len(next(iter(results.values())))
    rank_agg = None
    for k in results:
        norm = stats.rankdata(results[k]) / m
        weights = np.exp(-alpha * norm)
        weights /= np.sum(weights)
        rank_agg = weights if rank_agg is None else rank_agg + weights
    return -rank_agg


from sklearn.model_selection import cross_val_predict, KFold
import lightgbm as lgb
import numpy as np

import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
import numpy as np
import os

def lgbm_regression_aggregator(
    title: str,
    save_path: str,
    results: dict,
    target: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Trains a LightGBM regressor using k-fold CV and returns out-of-fold predictions.
    Optionally saves a feature importance plot.

    Parameters:
        results: dict of proxy score arrays
        target: ground truth (e.g., accuracy) to predict
        n_splits: number of CV folds
        random_state: for reproducibility
        save_path: optional path to save feature importance plot
        
    Returns:
        preds: out-of-fold predictions (same length as target)
    """

    feature_names = sorted(results.keys())
    X = np.column_stack([results[k] for k in feature_names])
    y = target

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    importances = np.zeros((n_splits, len(feature_names)))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        model = lgb.LGBMRegressor(
            num_leaves=15,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state + fold
        )
        model.fit(X[train_idx], y[train_idx])
        preds[valid_idx] = model.predict(X[valid_idx])
        importances[fold] = model.feature_importances_

    avg_importance = np.mean(importances, axis=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        sorted_idx = np.argsort(avg_importance)[::-1]
        plt.barh(
            [feature_names[i] for i in sorted_idx],
            avg_importance[sorted_idx],
            color='teal'
        )
        plt.xlabel("Average Feature Importance")
        plt.title(f"{title}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    return preds



from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

data_dir = "./results_new/Combined_Results"
datasets = ["Cifar10", "Cifar100", "ImageNet"]

output_dir = os.path.join(data_dir, "EnsembleProxiesResults")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_files = {name: np.load(f"{os.path.join(data_dir, name)}_dictionary.npz") for name in datasets}
proxy_scores = np.load(os.path.join(data_dir, "Proxy_Scores_Dictionary.npz"))

def run_experiment1():
    aggregators = {
        "AZNAS": az_aggregator,
        "LGB": None,
    }

    for dataset_name in datasets:
        gt_acc = data_files[dataset_name]["accuracy"]
        gt_FLOPs = data_files[dataset_name]["FLOPs"]
        gt_PARAMS = data_files[dataset_name]["params"]

        for agg_name, agg_fn in aggregators.items():
            plt.figure(figsize=(10, 6))
            correlation_matrix_results = {}

            for ensemble_name, proxy_list in EnsembleProxies.items():
                # Collect proxy results
                results = {
                    proxy.value: proxy_scores[proxy.value]
                    for proxy in proxy_list
                    if proxy not in {Proxy.FLOPS, Proxy.PARAMS}
                }
                if Proxy.FLOPS in proxy_list:
                    results["FLOPs"] = gt_FLOPs
                if Proxy.PARAMS in proxy_list:
                    results["params"] = gt_PARAMS

                # Aggregate scores
                if agg_name == "LGB":
                    title = f"Dataset: {dataset_name}, Aggregator: {agg_name}, Ensemble: {ensemble_name}"
                    path = os.path.join(output_dir, "LGB", f"Dataset: {dataset_name}, Aggregator: {agg_name}, Ensemble: {ensemble_name}.png")
                    agg_score = lgbm_regression_aggregator(title, path, results, gt_acc, )
                else:
                    agg_score = agg_fn(results, None)

                # Save for heatmap
                correlation_matrix_results[ensemble_name] = agg_score

                # === SCATTER PLOT ===
                pred_rank = stats.rankdata(agg_score)
                true_rank = stats.rankdata(gt_acc)
                scatter_title = f"Dataset: {dataset_name}, Aggregator: {agg_name}, Ensemble: {ensemble_name}"
                scatter_path = os.path.join(output_dir, "scatter_plots", f"Dataset: {dataset_name}, Aggregator: {agg_name}, Ensemble: {ensemble_name}.png")
                make_scatter_plot(pred_rank, true_rank, scatter_title, scatter_path)

            # Add GT for correlation matrix
            correlation_matrix_results["accuracy"] = gt_acc
            heatmap_title = f"{dataset_name} - {agg_name} Correlation Matrix"
            heatmap_path = os.path.join(output_dir, "CorrelationMatrics", f"Dataset: {dataset_name}, Aggregator: {agg_name}.png")
            make_correlation_matrix(correlation_matrix_results, heatmap_title, heatmap_path)



def run_experiment2():
    aggregators = {
        "LOG": az_aggregator,
        "EXP": az_aggregator_exp,
        "SOFTMAX": softmax_aggregator,
        "POWER": power_aggregator,
        "SIGMOID": sigmoid_aggregator,
        "THRESHOLD": rectifier_aggregator,
    }

    alphas = np.linspace(0.001, 10, 100)  # Sweep from forgiving to strict

    for dataset_name in datasets:
        gt_acc = data_files[dataset_name]["accuracy"]
        gt_FLOPs = data_files[dataset_name]["FLOPs"]
        gt_PARAMS = data_files[dataset_name]["params"]

        for agg_name, agg_fn in aggregators.items():
            plt.figure(figsize=(10, 6))

            for ensemble_name, proxy_list in EnsembleProxies.items():
                correlations = []
                for alpha in alphas:
                    # Collect relevant proxies
                    results = {
                        proxy.value: proxy_scores[proxy.value]
                        for proxy in proxy_list if proxy not in {Proxy.FLOPS, Proxy.PARAMS}
                    }
                    if Proxy.FLOPS in proxy_list:
                        results["FLOPs"] = gt_FLOPs
                    if Proxy.PARAMS in proxy_list:
                        results["params"] = gt_PARAMS

                    # Aggregate scores and compute Kendall Tau
                    agg_score = agg_fn(results, alpha)
                    corr, _ = stats.kendalltau(agg_score, gt_acc)
                    correlations.append(corr)

                # Plot for this ensemble
                plt.plot(alphas, correlations, label=ensemble_name)

            plt.title(f"{dataset_name} - {agg_name} Aggregator", fontsize=14)
            plt.xlabel("Strictness (alpha)", fontsize=12)
            plt.ylabel("Kendall's Tau vs True Accuracy", fontsize=12)
            plt.grid(True)
            plt.legend(title="Ensemble Proxy", fontsize=10)
            plt.tight_layout()
            save_file = os.path.join(output_dir, "sweeps", f"{dataset_name}_{agg_name}_correlation_vs_alpha.png")
            plt.savefig(save_file, dpi=300)
            plt.close()



if __name__ == "__main__":
    #run_experiment1()
    run_experiment2()

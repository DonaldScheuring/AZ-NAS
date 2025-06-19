import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from enum import Enum
import os
from reconstruct_plots import make_correlation_matrix
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


def az_aggregator(results, alpha=None):
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

def lgbm_regression_aggregator(results: dict, target: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """
    Trains a LightGBM regressor using k-fold CV and returns out-of-fold predictions
    (i.e., valid for generalization evaluation).
    
    Parameters:
        results: dict of proxy score arrays
        target: ground truth (e.g., accuracy) to predict
        n_splits: number of CV folds
        random_state: for reproducibility
        
    Returns:
        preds: out-of-fold predictions (same length as target)
    """
    # Stack proxy features
    X = np.column_stack([results[k] for k in sorted(results.keys())])
    y = target

    model = lgb.LGBMRegressor(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=100,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )

    # Cross-validated predictions: each prediction is made on data not seen during training
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)  # n_jobs=-1 for parallelism

    return preds


from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer



# ==== MAIN EXPERIMENT ====
def run_experiment():
    data_dir = "./results/June18th_1000_Samples_Rand_Seed_1"
    
    output_dir = os.path.join(data_dir, "ensemble_tests")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datasets = ["cifar10", "cifar100", "ImageNet16120"]
    data_files = {name: np.load(f"{os.path.join(data_dir, name)}_dictionary.npz") for name in datasets}
    proxy_scores = np.load(os.path.join(data_dir, "Proxy_Scores_Dictionary.npz"))

    aggregators = {
        "AX_REG": az_aggregator,
        "AZ_EXP": az_aggregator_exp,
        # "AZ_LOG": az_aggregator_log,
        "GEO_MEAN": geometric_mean_param,
        "GEO_MEAN_SENSE":geometric_mean_param_rank_sensitive,
        "SOFTMAX": softmax_aggregator,
        "POWER": power_aggregator,
        "SIGMOID": sigmoid_aggregator,
        "EXP": exp_decay_aggregator,
        "THRESHOLD": rectifier_aggregator,
        "BASE": az_aggregator_base,
        "LGB_REGRESSION": None,
    }

    #alphas = np.linspace(0.001, 10, 100)  # sweep from forgiving to strict
    alphas = [1]

    for dataset_name in datasets:
        gt_acc = data_files[dataset_name]["accuracy"]
        gt_FLOPs = data_files[dataset_name]["FLOPs"]
        gt_PARAMS = data_files[dataset_name]["params"]
        
        
        for agg_name, agg_fn in aggregators.items():
            plt.figure(figsize=(10, 6))
            correlation_matrix_results = {}
            for ensemble_name, proxy_list in EnsembleProxies.items():
                correlations = []
                for alpha in alphas:
                    # Build results dict
                    results = {
                        proxy.value: proxy_scores[proxy.value] for proxy in proxy_list if proxy != Proxy.FLOPS and proxy != Proxy.PARAMS
                    }

                    if Proxy.FLOPS in proxy_list:
                        results["FLOPs"] = gt_FLOPs
                    if Proxy.PARAMS in proxy_list:
                        results["params"] = gt_PARAMS

                    # Aggregate ranks

                    if agg_name == "LGB_REGRESSION":
                        agg_score = lgbm_regression_aggregator(results, gt_acc)
                    else:
                        agg_score = agg_fn(results, alpha)

                    # Use to make 2D correlation later
                    correlation_matrix_results[ensemble_name] = agg_score

                    # Correlation to ground truth
                    corr, _ = stats.kendalltau(agg_score, gt_acc)
                    correlations.append(corr)

                plt.plot(alphas, correlations, label=ensemble_name)
                plt.title(f"{dataset_name} - {agg_name} Aggregator")
                plt.xlabel("Strictness (alpha)")
                plt.ylabel("Kendall Tau with True Accuracy")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir,f"{dataset_name}_{agg_name}_correlation_vs_alpha.png"))
                plt.close()
            
            correlation_matrix_results["accuracy"] = gt_acc
            print(correlation_matrix_results.keys())
            make_correlation_matrix(correlation_matrix_results, f"{dataset_name}_{agg_name}", os.path.join(output_dir,f"{dataset_name}_{agg_name}.png"))


        # correlation_matrix_results = {}
        # plt.figure(figsize=(10, 6))
        # for ensemble_name, proxy_list in EnsembleProxies.items():
        #     correlations = []

        #     # Build results dict
        #     results = {
        #         proxy.value: proxy_scores[proxy.value] for proxy in proxy_list if proxy not in [Proxy.FLOPS, Proxy.PARAMS]
        #     }

        #     if Proxy.FLOPS in proxy_list:
        #         results["FLOPs"] = gt_FLOPs
        #     if Proxy.PARAMS in proxy_list:
        #         results["params"]  = gt_PARAMS

        #     # Aggregate ranks
        #     agg_score = az_aggregator(results)

        #     # Use to make 2D correlation later
        #     correlation_matrix_results[ensemble_name] = agg_score

        



if __name__ == "__main__":
    run_experiment()

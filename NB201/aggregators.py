import numpy as np
import scipy.stats as stats

def az_aggregator(results):
    rank_agg = None
    for k in results.keys():    
        if rank_agg is None:
            rank_agg = np.log( stats.rankdata(results[k]) / len(results[k]))
        else:
            rank_agg = rank_agg + np.log( stats.rankdata(results[k]) / len(results[k]))
    return rank_agg


# def tenas_aggregator(results):
#     rank_agg = None
#     for k in results.keys():
#         if rank_agg is None:
#             rank_agg = stats.rankdata(results[k])
#         else:
#             rank_agg = rank_agg + stats.rankdata(results[k]) # NOTE: how does adding ranks work?
#     return rank_agg

def az_aggregator_exp(results, alpha=1.0):
    rank_agg = None
    for k in results.keys():
        norm_ranks = stats.rankdata(results[k]) / len(results[k])
        weighted_ranks = norm_ranks ** alpha
        if rank_agg is None:
            rank_agg = weighted_ranks
        else:
            rank_agg += weighted_ranks
    return rank_agg

def az_aggregator_log(results, alpha=1.0):
    rank_agg = None
    for k in results.keys():
        norm_ranks = stats.rankdata(results[k]) / len(results[k])
        weighted_log = np.log(norm_ranks) * alpha
        if rank_agg is None:
            rank_agg = weighted_log
        else:
            rank_agg += weighted_log
    return rank_agg

def geometric_mean_param(results: dict, alpha: float = 1.0):
    """
    Parameterized geometric mean of normalized ranks:
    S(i) = - (prod_j ((m - Rank_j(i)) / m)^alpha)^(1/n_proxies)
         = - (prod_j (m - Rank_j(i)) / m)^(alpha / n_proxies)

    Lower score is better. Alpha > 1 increases strictness, < 1 makes it more forgiving.
    """
    n_proxies = len(results)
    m = len(next(iter(results.values())))

    rank_agg = None
    for k in results.keys():
        norm = (m - stats.rankdata(results[k])) / m  # higher is better
        norm = norm ** alpha  # apply strictness control
        if rank_agg is None:
            rank_agg = norm
        else:
            rank_agg *= norm

    rank_agg = rank_agg ** (1 / n_proxies)
    return -rank_agg  # Lower is better








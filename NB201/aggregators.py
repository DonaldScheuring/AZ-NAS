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

def tenas_aggregator(results):
    rank_agg = None
    for k in results.keys():
        if rank_agg is None:
            rank_agg = stats.rankdata(results[k])
        else:
            rank_agg = rank_agg + stats.rankdata(results[k]) # NOTE: how does adding ranks work?
    return rank_agg

def geometric_mean(results: dict):
    """
    Element-wise geometric mean of normalized ranks:
    S(i) = (prod_j (m - Rank_j(i)) / m) ^ (1/n_proxies)
    Penalizes low ranks, but less harshly than AZ-NAS.
    """
    n_proxies = len(results)
    m = len(next(iter(results.values())))  # number of architectures

    rank_agg = None
    for k in results.keys():
        # Normalize: higher rank (worse) -> closer to 0
        normalized = (m - stats.rankdata(results[k])) / m  # shape: (m,)
        if rank_agg is None:
            rank_agg = normalized
        else:
            rank_agg *= normalized  # elementwise product

    rank_agg = rank_agg ** (1 / n_proxies)  # elementwise root
    return -rank_agg  # shape: (m,)


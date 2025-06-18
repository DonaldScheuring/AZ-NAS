import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import time
import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
import json
import datetime

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import copy

# XAutoDL 
from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_search_spaces

# API
from nats_bench import create

# custom modules
from custom.tss_model import TinyNetwork
from xautodl.models.cell_searchs.genotypes import Structure
from ZeroShotProxy import *
from proxies import EnsembleProxies
from aggregators import *
from collections import defaultdict


parser = argparse.ArgumentParser("Training-free NAS on NAS-Bench-201 (NATS-Bench-TSS)")
parser.add_argument("--data_path", type=str, default='./cifar.python', help="The path to dataset")
parser.add_argument("--dataset", type=str, default='cifar10',choices=["cifar10", "cifar100", "ImageNet16-120"], help="Choose between Cifar10/100 and ImageNet-16.")

# channels and number-of-cells
parser.add_argument("--search_space", type=str, default='tss', help="The search space name.")
parser.add_argument("--config_path", type=str, default='./configs/nas-benchmark/algos/weight-sharing.config', help="The path to the configuration.")
parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
parser.add_argument("--affine", type=int, default=1, choices=[0, 1], help="Whether use affine=True or False in the BN layer.")
parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1], help="Whether use track_running_stats or not in the BN layer.")

# log
parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")

# custom
parser.add_argument("--gpu", type=int, default=0, help="To enable GPU set to 0, to disable set to None")
parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
parser.add_argument("--api_data_path", type=str, default="./api_data/NATS-tss-v1_0-3ffb9-simple/", help="")

desc = "2_Epoch_Test"
experiment_name = f"Experiment_{desc}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
default_save_dir = os.path.join("./results", experiment_name)
if not  os.path.exists(default_save_dir):
    os.makedirs(default_save_dir)

parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Folder to save results to")
#parser.add_argument("--save_checkpoints_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")


parser.add_argument("--n_samples", type=int, default=20, help="Number of architectures to evaluate from NB201")


parser.add_argument(
    '--proxies',
    nargs='+',
    #default=['aznas', 'zen', 'gradnorm', 'naswot', 'synflow','zico'],
    default=['aznas','zico','zen','gradnorm','naswot','synflow','snip','grasp','gradsign_rev', 'tenas'],
    help="A list of proxy names to include in the analysis. "
         "Provide multiple names separated by spaces (e.g., --proxies aznas zen tenas)."
)


parser.add_argument("--rand_seed", type=int, default=1, help="manual seed (we use 1-to-5)")
args = parser.parse_args(args=[])



if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)

print(args.rand_seed)
print(args)
xargs=args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = prepare_logger(args)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(xargs.workers)
prepare_seed(xargs.rand_seed)
logger = prepare_logger(args)



# Let system decide which to useNB201
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0) # Get name of the first GPU
    logger.log(f"PyTorch: GPU is available! Using: {gpu_name}")
    gpu = torch.cuda.current_device()
    logger.log(f"gpu variable: {gpu}")
    device = torch.device('cuda:{}'.format(xargs.gpu))
    logger.log(f"device variable: {device}")
else:
    logger.log("PyTorch: No GPU found, using CPU.")
    gpu = None
    device = "cpu"


real_input_metrics = ['zico', 'snip', 'grasp', 'tenas', 'gradsign_rev']

# dataloaders
train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
config = load_config(xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger)
search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data,
                                                                valid_data,
                                                                xargs.dataset,
                                                                "./configs/nas-benchmark/",
                                                                (config.batch_size, config.test_batch_size),
                                                                xargs.workers,)
logger.log("||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))



def get_nasbench201_api():
    api = create(xargs.api_data_path, xargs.search_space, fast_mode=True, verbose=False)
    logger.log("Create API = {:} done".format(api))
    return api

## model
def get_search_space(logger, xargs):
    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    logger.log("search space : {:}".format(search_space))
    return search_space

def random_genotype(max_nodes, op_names):
    genotypes = []
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            op_name = random.choice(op_names)
            xlist.append((op_name, j))
        genotypes.append(tuple(xlist))
    arch = Structure(genotypes)
    return arch


def search_find_best(xargs, xloader, search_space, n_samples=None, archs=None):

    input_, target_ = next(iter(xloader))
    resolution = input_.size(2)
    batch_size = input_.size(0)
    arch_list = []

    proxies = xargs.proxies
    logger.log(f"Proxy list: {proxies}")

    proxy_stats = {proxy: {"times": [], "mem_alloc": [], "mem_reserved": []} for proxy in proxies}
    zero_shot_score_dict = defaultdict(list)  # Score values
    arch_list = []

    logger.log(f"GPU: {gpu}")
    logger.log(f"Device: {device}")

    if gpu is not None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    for i in tqdm.tqdm(range(n_samples)):
        if gpu is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        arch = random_genotype(xargs.max_nodes, search_space)
        network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num).to(device)
        network.train()

        scores_dict = {}
        for proxy in proxies:

            #logger.log(f"Processing proxy: {proxy}...")

            if proxy in real_input_metrics:
                trainloader = train_loader
            else:
                trainloader = None

            score_fn_name = f"compute_{proxy.lower()}_score"
            score_fn = globals().get(score_fn_name)

            if gpu is not None:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start.record()

            score_dict = score_fn.compute_nas_score(
                network, gpu, trainloader=trainloader,
                resolution=resolution, batch_size=batch_size
            )

            if gpu is not None:
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                mem_reserved = torch.cuda.max_memory_reserved()
                mem_alloc = torch.cuda.max_memory_allocated()
                
                # logger.log("Appending time, memory...")
                proxy_stats[proxy]["times"].append(elapsed_time)
                proxy_stats[proxy]["mem_reserved"].append(mem_reserved)
                proxy_stats[proxy]["mem_alloc"].append(mem_alloc)

            scores_dict.update(score_dict)

        arch_list.append(arch)
        # Some proxies return more than one key value pair
        for key, value in scores_dict.items():
            zero_shot_score_dict[key].append(value)

    logger.log(f"proxy_stats: {proxy_stats}")
    # Compile performance summary
    proxy_perf_summary = {}
    for proxy, stats in proxy_stats.items():
        proxy_perf_summary[proxy] = {
            "avg_time_ms": float(np.mean(stats["times"])) if stats["times"] else 0.0,
            "avg_mem_GB": float(np.mean(stats["mem_reserved"])) / 1e9 if stats["mem_reserved"] else 0.0,
            "max_mem_GB": float(np.max(stats["mem_reserved"])) / 1e9 if stats["mem_reserved"] else 0.0
        }
    return arch_list, zero_shot_score_dict, proxy_perf_summary


def get_results_from_api(api, arch, dataset='cifar10'):
    dataset_candidates = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    assert dataset in dataset_candidates
    index = api.query_index_by_arch(arch)
    api._prepare_info(index)
    archresult = api.arch2infos_dict[index]['200']
    
    if dataset == 'cifar10-valid':
        acc = archresult.get_metrics(dataset, 'x-valid', iepoch=None, is_random=False)['accuracy']
    elif dataset == 'cifar10':
        acc = archresult.get_metrics(dataset, 'ori-test', iepoch=None, is_random=False)['accuracy']
    else:
        acc = archresult.get_metrics(dataset, 'x-test', iepoch=None, is_random=False)['accuracy']
    flops = archresult.get_compute_costs(dataset)['flops']
    params = archresult.get_compute_costs(dataset)['params']
    
    return acc, flops, params


def main():
    api = get_nasbench201_api()
    search_space = get_search_space(logger, xargs)

    # Get archs and proxy scores (raw)
    archs, proxy_scores, proxy_perf_summary = search_find_best(
        xargs, train_loader, search_space, xargs.n_samples
    )

    # Save proxy data
    save_path = os.path.join(xargs.save_dir, f"Proxy_Scores_Dictionary.npz")
    np.savez(save_path, **proxy_scores)
    logger.log(f"Saved Proxy_Scores_Dictionary to {save_path}")
    logger.log(f"Proxy dictionary: {proxy_scores}")

    # Get and save dataset specific data
    datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
    dataset_results = {}

    for dataset in datasets:
        accs, flops, params = [], [], []

        for arch in archs:
            acc, f, p = get_results_from_api(api, arch, dataset)
            accs.append(acc)
            flops.append(f)
            params.append(p)

        results = {}
        results["accuracy"] = accs
        results["FLOPs"] = flops
        results["params"] = params
        dataset_results[dataset] = results

    for dataset in datasets:
        results = dataset_results[dataset]

        # Save dictionary
        save_path = os.path.join(xargs.save_dir, f"{dataset.replace('-', '')}_dictionary.npz")
        np.savez(save_path, **results)
        logger.log(f"Saved {dataset} results to {save_path}")

    # Save runtime/memory stats to JSON
    with open(os.path.join(xargs.save_dir, "proxy_performance_summary.json"), "w") as f:
        json.dump(proxy_perf_summary, f, indent=4)
    logger.log(f"Saved proxy performance summary to proxy_performance_summary.json")

    # Save architecture list
    with open(os.path.join(xargs.save_dir, "architectures.json"), "w") as f:
        json.dump([str(a) for a in archs], f, indent=2)
    logger.log(f"Saved architecture list to architectures.json")


if __name__ == "__main__":
    main()
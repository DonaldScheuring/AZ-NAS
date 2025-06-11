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

parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save results to")
#parser.add_argument("--save_checkpoints_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")


#parser.add_argument('--zero_shot_score', type=str, default='az_nas', choices=['az_nas','zico','zen','gradnorm','naswot','synflow','snip','grasp','te_nas','gradsign'])
parser.add_argument("--n_samples", type=int, default=10, help="Number of architectures to evaluate from NB201")
parser.add_argument(
    '--proxies',
    nargs='+',
    default=['aznas', 'zen', 'gradnorm', 'naswot', 'synflow', 'snip', 'grasp', 'gradsign', 'tenas', 'zico'],
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



# Let system decide which to use
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0) # Get name of the first GPU
    print(f"PyTorch: GPU is available! Using: {gpu_name}")
    gpu = torch.cuda.current_device()
    print(f"gpu variable: {gpu}")
    device = torch.device('cuda:{}'.format(xargs.gpu))
    print(f"device variable: {device}")
else:
    print("PyTorch: No GPU found, using CPU.")
    gpu = None
    device = "cpu"


real_input_metrics = ['zico', 'snip', 'grasp', 'tenas', 'gradsign']

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


    
def search_find_best(xargs, xloader, search_space, n_samples = None, archs = None):
    
    input_, target_ = next(iter(xloader))
    resolution = input_.size(2)
    batch_size = input_.size(0)
    zero_shot_score_dict = None 
    arch_list = []
        
    if archs is None and n_samples is not None:
        all_time = []
        all_mem = []
        if gpu:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        for i in tqdm.tqdm(range(n_samples)):
            if gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            # random sampling
            arch = random_genotype(xargs.max_nodes, search_space)
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            if gpu:
                start.record()


            # Each proxy returns a dictionary with the name and value of the proxy
            scores_dict = {}
            for proxy in xargs.proxies:
                
                if proxy in real_input_metrics:
                    print('Use real images as inputs')
                    trainloader = train_loader
                else:
                    print('Use random inputs')
                    trainloader = None

                score_fn_name = "compute_{}_score".format(proxy.lower())
                score_fn = globals().get(score_fn_name)


                score_dict = score_fn.compute_nas_score(network, gpu, trainloader=trainloader, resolution=resolution, batch_size=batch_size)
                scores_dict.update(score_dict)

            if gpu:
                end.record()
                torch.cuda.synchronize()
                all_time.append(start.elapsed_time(end))
                all_mem.append(torch.cuda.max_memory_reserved())
                all_mem.append(torch.cuda.max_memory_allocated())

            arch_list.append(arch)
            if zero_shot_score_dict is None: # initialize dict
                zero_shot_score_dict = dict()
                for k in scores_dict.keys():
                    zero_shot_score_dict[k] = []
            for k, v in scores_dict.items():
                zero_shot_score_dict[k].append(v)

        if gpu:
            # NOTE: Logger saves everything to the log files that are created on each run. Nice. 
            logger.log("------Runtime------")
            logger.log("All: {:.5f} ms".format(np.mean(all_time)))
            logger.log("------Avg Mem------")
            logger.log("All: {:.5f} GB".format(np.mean(all_mem)/1e9))
            logger.log("------Max Mem------")
            logger.log("All: {:.5f} GB".format(np.max(all_mem)/1e9))
    
    # Note: the following code is for running through all archs
    elif archs is not None and n_samples is None:
        all_time = []
        all_mem = []
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)
        for arch in tqdm.tqdm(archs):
            #torch.cuda.empty_cache()
            #torch.cuda.reset_peak_memory_stats()
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            #start.record()

            info_dict = score_fn.compute_nas_score(network, gpu, trainloader=trainloader, resolution=resolution, batch_size=batch_size)

            #end.record()
            # torch.cuda.synchronize()
            #all_time.append(start.elapsed_time(end))
#             all_mem.append(torch.cuda.max_memory_reserved())
            #all_mem.append(torch.cuda.max_memory_allocated())

            arch_list.append(arch)
            if zero_shot_score_dict is None: # initialize dict
                zero_shot_score_dict = dict()
                for k in info_dict.keys():
                    zero_shot_score_dict[k] = []
            for k, v in info_dict.items():
                zero_shot_score_dict[k].append(v)

        # logger.log("------Runtime------")
        # logger.log("All: {:.5f} ms".format(np.mean(all_time)))
        # logger.log("------Avg Mem------")
        # logger.log("All: {:.5f} GB".format(np.mean(all_mem)/1e9))
        # logger.log("------Max Mem------")
        # logger.log("All: {:.5f} GB".format(np.max(all_mem)/1e9))

    print(f"Zero shot score dict: {zero_shot_score_dict}")    
    return arch_list, zero_shot_score_dict


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


def make_correlation_matrix(results: dict):

    # Prepare metrics
    metrics = copy.deepcopy(results)

    keys = key_names = list(metrics.keys())

    # Build correlation matrix
    matrix = np.zeros((len(keys), len(keys)))
    for i in range(len(keys)):
        for j in range(len(keys)):
            x = stats.rankdata(metrics[keys[i]])
            y = stats.rankdata(metrics[keys[j]])
            kendalltau = stats.kendalltau(x, y)[0]
            matrix[i, j] = kendalltau

    # Convert to DataFrame
    df_cm = pd.DataFrame(matrix, index=key_names, columns=key_names)

    # Plot
    plt.figure(figsize=(10, 10))
    ax = sn.heatmap(df_cm,
                     annot=True,
                     fmt=".2f",
                     cmap='GnBu',
                     cbar_kws={"shrink": 0.8},
                     square=True,
                     linewidths=0.5,
                     linecolor='gray',
                     annot_kws={"size": 10})

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title("Kendall Tau Correlation Matrix", fontsize=14, pad=20)
    plt.tight_layout()

    filepath = os.path.join(xargs.save_dir, "figs")
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    plt.savefig(os.path.join(filepath,"correlation_matrix.png"), dpi=300)

#### Visualize scatter plots
def visualize_proxy_cmap(x, y, title, save_name, ref_rank=None):
    if ref_rank is None:
        ref_rank = x
    plt.figure(figsize=(4.5*1.5,3*1.5))
    plt.grid(True, alpha=0.3)
    plt.scatter(x,y, linewidths=0.1, c=ref_rank, cmap='viridis_r')
    plt.xlabel("Predicted network ranking", fontsize=12)
    plt.ylabel("Ground-truth network ranking", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar()
    plt.title(title, fontsize=20)

    filepath = os.path.join(xargs.save_dir, "figs/scatter_plots")
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    plt.savefig(os.path.join(filepath,'{}.png'.format(save_name)), dpi=300)

def get_proxy_scatter_plots(results):
    
    proxy_names = []
    kendall_correlations = []
    pearson_correlations = []    

    # --------- Proxy vs Proxy Correlations ----------------
    for k in results.keys(): 
        print(f"Processing proxy: {k}")
        x = stats.rankdata(results[k])
        y = stats.rankdata(results["accuracy"]) # Assuming api_valid_accs is the true accuracy
        
        kendalltau = stats.kendalltau(x, y)
        spearmanr = stats.spearmanr(x, y)
        pearsonr = stats.pearsonr(x, y)
        
        # Visualize the scatter plot for each proxy
        visualize_proxy_cmap(x, y, r"{0} ($\tau$={1:.3f}, $\rho$={2:.3f})".format(k, kendalltau[0], spearmanr[0]), k)
        
        # Store correlations for the bar chart
        proxy_names.append(k)
        kendall_correlations.append(kendalltau[0])
        pearson_correlations.append(pearsonr[0])    

    # NOTE: This is for determining the best arch based on AZ-NAS
    # best_idx = np.argmax(rank_agg)
    # best_arch, acc = archs[best_idx], api_valid_accs[best_idx]
    # if api is not None:
    #     print("{:}".format(api.query_by_arch(best_arch, "200")))


    # ----------- Bar Chart for Proxy vs True accuracy -----------
    
    plt.figure(figsize=(10, 6))
    
    # Set the positions for the bars
    x_positions = np.arange(len(proxy_names))
    width = 0.35 # Width of the bars
    
    # Plot Kendall's Tau bars
    plt.bar(x_positions - width/2, kendall_correlations, width, label="Kendall's Tau", color='skyblue')
    
    # Plot Pearson correlation bars
    plt.bar(x_positions + width/2, pearson_correlations, width, label="Pearson Correlation", color='lightcoral')
    
    plt.xlabel("Proxy", fontsize=12)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.title("Correlation of Proxies with True Accuracy", fontsize=16)
    plt.xticks(x_positions, proxy_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([-1, 1]) # Correlation coefficients range from -1 to 1
    plt.axhline(0, color='gray', linewidth=0.8) # Add a line at y=0 for reference
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    filepath = os.path.join(xargs.save_dir, "figs")
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    plt.savefig(os.path.join(filepath,"proxy_correlations_bar_chart.png"), dpi=300)

def add_ensembles_to_results(results, api_valid_accs, api_flops):
    results.update({'FLOPs':api_flops})
    # Add ground_truth accuracy to results
    results.update({'accuracy':api_valid_accs})

    # Also put TE-NAS proxy in dictionary
    rank_agg = None
    for k in results.keys():
        if "_tenas" in k:
            print(k)
            if rank_agg is None:
                rank_agg = stats.rankdata(results[k])
            else:
                rank_agg = rank_agg + stats.rankdata(results[k]) # NOTE: how does adding ranks work?
    results.update({"TE-NAS":rank_agg})

    # Also put AZ-NAS proxy in dictionary
    rank_agg = None
    l = len(api_flops)
    rank_agg = np.log(stats.rankdata(api_flops) / l)
    for k in results.keys():    
        if "_az" in k or k=="FLOPs": # NOTE: need to only extract out AZ_nas proxies, not all of them
            print(k)
            if rank_agg is None:
                rank_agg = np.log( stats.rankdata(results[k]) / l)
            else:
                rank_agg = rank_agg + np.log( stats.rankdata(results[k]) / l)
    results.update({"AZ-NAS":rank_agg})

def main():

    api = get_nasbench201_api()

    search_space = get_search_space(logger, xargs)

    ######### search across random N archs #########
    archs, results = search_find_best(xargs, train_loader, search_space, xargs.n_samples)


    api_valid_accs, api_flops, api_params = [], [], []
    for a in archs:
        valid_acc, flops, params = get_results_from_api(api, a, 'cifar10')
        #valid_acc, flops, params = get_results    plt.show()_from_api(api, a, 'cifar100')
        #valid_acc, flops, params = get_results_from_api(api, a, 'ImageNet16-120')
        api_valid_accs.append(valid_acc)
        api_flops.append(flops)
        api_params.append(params)
        
    print("Maximum acc: {}% \n Info".format(np.max(api_valid_accs)))
    best_idx = np.argmax(api_valid_accs)
    best_arch = archs[best_idx]
    if api is not None:
        print("{:}".format(api.query_by_arch(best_arch, "200")))


    # NOTE: get_proxy_scatter_plots modifies results and adds the aggregate proxies (AZ-NAS, TE-NAS) and api_valid_accs and api_flops

    # Put ensemble metrics in the results
    add_ensembles_to_results(results, api_valid_accs, api_flops)

    # Make visualisations
    get_proxy_scatter_plots(results)
    make_correlation_matrix(results)

    # After updating results, lets save all the whole dictionary (maybe json or picke file)
    output_dir = os.path.join(xargs.save_dir, "figs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir,"results_dictionary.npz")
    np.savez(filepath, **results)
    logger.log(f"Results dictionary saved to: {filepath}")

    # TODO: should also save experiment settings from xargs


if __name__ == "__main__":
    main()
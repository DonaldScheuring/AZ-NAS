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
parser.add_argument("--gpu", type=int, default=0, help="")
parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
parser.add_argument("--api_data_path", type=str, default="./api_data/NATS-tss-v1_0-3ffb9-simple/", help="")
parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")
parser.add_argument('--zero_shot_score', type=str, default='az_nas', choices=['az_nas','zico','zen','gradnorm','naswot','synflow','snip','grasp','te_nas','gradsign'])
parser.add_argument("--rand_seed", type=int, default=1, help="manual seed (we use 1-to-5)")
args = parser.parse_args(args=[])

if args.rand_seed is None or args.rand_seed < 0:
    args.rand_seed = random.randint(1, 100000)

print(args.rand_seed)
print(args)
xargs=args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = prepare_logger(args)



assert torch.cuda.is_available(), "CUDA is not available."
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(xargs.workers)
prepare_seed(xargs.rand_seed)
logger = prepare_logger(args)

device = torch.device('cuda:{}'.format(xargs.gpu))

real_input_metrics = ['zico', 'snip', 'grasp', 'te_nas', 'gradsign']

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
    logger.log("Searching with {}".format(xargs.zero_shot_score.lower()))
    score_fn_name = "compute_{}_score".format(xargs.zero_shot_score.lower())
    score_fn = globals().get(score_fn_name)
    input_, target_ = next(iter(xloader))
    resolution = input_.size(2)
    batch_size = input_.size(0)
    zero_shot_score_dict = None
    arch_list = []
    if xargs.zero_shot_score.lower() in real_input_metrics:
        print('Use real images as inputs')
        trainloader = train_loader
    else:
        print('Use random inputs')
        trainloader = None
        
    if archs is None and n_samples is not None:
        all_time = []
        all_mem = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in tqdm.tqdm(range(n_samples)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # random sampling
            arch = random_genotype(xargs.max_nodes, search_space)
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            start.record()
            

            info_dict = score_fn.compute_nas_score(network, gpu=xargs.gpu, trainloader=trainloader, resolution=resolution, batch_size=batch_size)

            end.record()
            torch.cuda.synchronize()
            all_time.append(start.elapsed_time(end))
#             all_mem.append(torch.cuda.max_memory_reserved())
            all_mem.append(torch.cuda.max_memory_allocated())

            arch_list.append(arch)
            if zero_shot_score_dict is None: # initialize dict
                zero_shot_score_dict = dict()
                for k in info_dict.keys():
                    zero_shot_score_dict[k] = []
            for k, v in info_dict.items():
                zero_shot_score_dict[k].append(v)

        logger.log("------Runtime------")
        logger.log("All: {:.5f} ms".format(np.mean(all_time)))
        logger.log("------Avg Mem------")
        logger.log("All: {:.5f} GB".format(np.mean(all_mem)/1e9))
        logger.log("------Max Mem------")
        logger.log("All: {:.5f} GB".format(np.max(all_mem)/1e9))
        
    elif archs is not None and n_samples is None:
        all_time = []
        all_mem = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for arch in tqdm.tqdm(archs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            start.record()
            

            info_dict = score_fn.compute_nas_score(network, gpu=xargs.gpu, trainloader=trainloader, resolution=resolution, batch_size=batch_size)

            end.record()
            torch.cuda.synchronize()
            all_time.append(start.elapsed_time(end))
#             all_mem.append(torch.cuda.max_memory_reserved())
            all_mem.append(torch.cuda.max_memory_allocated())

            arch_list.append(arch)
            if zero_shot_score_dict is None: # initialize dict
                zero_shot_score_dict = dict()
                for k in info_dict.keys():
                    zero_shot_score_dict[k] = []
            for k, v in info_dict.items():
                zero_shot_score_dict[k].append(v)

        logger.log("------Runtime------")
        logger.log("All: {:.5f} ms".format(np.mean(all_time)))
        logger.log("------Avg Mem------")
        logger.log("All: {:.5f} GB".format(np.mean(all_mem)/1e9))
        logger.log("------Max Mem------")
        logger.log("All: {:.5f} GB".format(np.max(all_mem)/1e9))
        
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


def main():

    api = get_nasbench201_api()

    search_space = get_search_space(logger, xargs)

    ######### search across random N archs #########
    archs, results = search_find_best(xargs, train_loader, search_space, n_samples=3000)


    api_valid_accs, api_flops, api_params = [], [], []
    for a in archs:
        valid_acc, flops, params = get_results_from_api(api, a, 'cifar10')
    #     valid_acc, flops, params = get_results_from_api(api, a, 'cifar100')
    #     valid_acc, flops, params = get_results_from_api(api, a, 'ImageNet16-120')
        api_valid_accs.append(valid_acc)
        api_flops.append(flops)
        api_params.append(params)
        
    print("Maximum acc: {}% \n Info".format(np.max(api_valid_accs)))
    best_idx = np.argmax(api_valid_accs)
    best_arch = archs[best_idx]
    if api is not None:
        print("{:}".format(api.query_by_arch(best_arch, "200")))


if __name__ == "__main__":
    main()
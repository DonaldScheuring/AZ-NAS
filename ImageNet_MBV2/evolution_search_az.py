'''
Code modified from
ZenNAS: 'https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/evolution_search.py'
ZiCo : 'https://github.com/SLDGroup/ZiCo/blob/3eeb517d51cd447685099c8a4351edee8e31e999/evolution_search.py'
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np
import global_utils
import Masternet
import PlainNet
from xautodl import datasets
import time
from tqdm import tqdm



from ZeroShotProxy import *
import benchmark_network_latency

import scipy.stats as stats
from sklearn.metrics import mutual_info_score
import pandas as pd
import pickle
import warnings

# 1) Force newly-created tensors to default float32 (not float64).
torch.set_default_dtype(torch.float32)

# We'll compute correlation every N steps only:
CORR_INTERVAL = 50

working_dir = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default='az_nas')
    parser.add_argument('--search_space', type=str, default=None)
    parser.add_argument('--evolution_max_iter', type=int, default=100000)
    parser.add_argument('--budget_model_size', type=float, default=None)
    parser.add_argument('--budget_flops', type=float, default=None)
    parser.add_argument('--budget_latency', type=float, default=None)
    parser.add_argument('--max_layers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_image_size', type=int, default=224)
    parser.add_argument('--population_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--maxbatch', type=int, default=2)
    parser.add_argument('--rand_input', type=str2bool, default=True)
    parser.add_argument('--search_no_res', type=str2bool, default=False)
    parser.add_argument('--seed', type=none_or_int, default=None)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def get_new_random_structure_str(AnyPlainNet, structure_str, num_classes, get_search_space_func,
                                 num_replaces=1):
    """Mutate or create a new architecture string from the search space."""
    the_net = AnyPlainNet(num_classes=num_classes,
                          plainnet_struct=structure_str,
                          no_create=True)
    assert isinstance(the_net, PlainNet.PlainNet)

    selected_random_id_set = set()
    for _ in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        to_search_blocks_list_list = get_search_space_func(the_net.block_list, random_id)

        block_candidates = [x for sublist in to_search_blocks_list_list for x in sublist]
        block_candidates = sorted(block_candidates)
        new_student_block_str = random.choice(block_candidates)

        if len(new_student_block_str) > 0:
            new_block = PlainNet.create_netblock_list_from_str(new_student_block_str, no_create=True)
            assert len(new_block) == 1
            new_block = new_block[0]
            # fix in_channels if not first block
            if random_id > 0:
                last_block_outc = the_net.block_list[random_id - 1].out_channels
                new_block.set_in_channels(last_block_outc)
            the_net.block_list[random_id] = new_block
        else:
            # empty
            the_net.block_list[random_id] = None

    # remove empty + fix channels
    tmp_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for blk in tmp_list[1:]:
        blk.set_in_channels(last_channels)
        last_channels = blk.out_channels
    the_net.block_list = tmp_list

    new_str = the_net.split(split_layer_threshold=6)
    return new_str

def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str,
                          no_create=True)
    assert hasattr(the_net, 'split')
    splitted_str = the_net.split(split_layer_threshold=6)
    return splitted_str

def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    """Compute hardware latency if budget_latency is not None."""
    the_model = AnyPlainNet(num_classes=args.num_classes,
                            plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    lat = benchmark_network_latency.get_model_latency(
        model=the_model, batch_size=args.batch_size,
        resolution=args.input_image_size, in_channels=3,
        gpu=gpu, repeat_times=1, fp16=True
    )
    del the_model
    torch.cuda.empty_cache()
    return lat

def _force_fp32_after_forward(net, gpu, input_size):
    """
    Some blocks create double precision weights on-the-fly in forward().
    So we do a dummy forward pass, then forcibly cast all leftover double
    params/buffers to float32.
    """
    dummy_x = torch.zeros(1, 3, input_size, input_size, device=f'cuda:{gpu}')
    with torch.no_grad():
        try:
            net(dummy_x)  # triggers lazy param creation
        except Exception as e:
            print(f"[warn] dummy forward encountered error: {e}")

    # forcibly cast
    for p in net.parameters():
        if p.dtype == torch.float64:
            p.data = p.data.float()
    for b in net.buffers():
        if b.dtype == torch.float64:
            b.data = b.data.float()

# Looks like this one takes architectures in one at a time (not batch processed)
def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args,
                      trainloader=None, lossfunc=None):
    """Compute the standard AZ-NAS proxies."""
    the_model = AnyPlainNet(num_classes=args.num_classes,
                            plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=args.search_no_res)
    the_model = the_model.cuda(gpu)

    # 1) run a dummy forward pass & forcibly cast leftover doubles
    _force_fp32_after_forward(the_model, gpu, args.input_image_size)

    if args.zero_shot_score.lower() != 'az_nas':
        raise NotImplementedError("Use 'evolutionary_search_others.py' for other metrics")

    score_fn_name = f"compute_{args.zero_shot_score.lower()}_score"
    score_fn = globals().get(score_fn_name)
    info = score_fn.compute_nas_score(
        model=the_model, gpu=gpu, trainloader=trainloader,
        resolution=args.input_image_size, batch_size=args.batch_size
    )
    del the_model
    torch.cuda.empty_cache()
    return info

# This is the code Donald added. This only also takes one architecture in at a time
def compute_all_proxies(AnyPlainNet, random_structure_str, gpu, args, trainloader=None, lossfunc=None):
    net = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str, no_create=False, no_reslink=args.search_no_res)
    net = net.cuda(gpu)

    #dummy forward
    _force_fp32_after_forward(net, gpu, args.input_image_size)

    #az
    info = compute_az_nas_score.compute_nas_score(model=net, gpu=gpu, trainloader=trainloader, resolution=args.input_image_size, batch_size=args.batch_size)

    #proxies
    info['zen'] = compute_zen_score.compute_nas_score(gpu, net, args.gamma, args.input_image_size, args.batch_size, repeat=1)['avg_nas_score']
    info['gradnorm'] = compute_gradnorm_score.compute_nas_score(gpu, net, args.input_image_size, args.batch_size)
    info['syncflow'] = compute_syncflow_score.do_compute_nas_score(gpu, net, args.input_image_size, args.batch_size)

    try:
        info['naswot'] = compute_NASWOT_score.compute_nas_score(gpu, net, args.input_image_size, args.batch_size)
    except RuntimeError as e:
        print(f"[warn] NASWOT failed: {e}")
        info['naswot'] = np.nan

#    try:
#        info['te_nas'] = compute_te_nas_score.compute_NTK_score(gpu, net, args.input_image_size, args.batch_size)
#    except RuntimeError as e:
#        print(f"[warn] TE-NAS failed: {e}")
#        info['te_nas'] = np.nan

    if trainloader is not None and lossfunc is not None:
        info['zico'] = compute_zico.getzico(net, trainloader, lossfunc)
    else:
        info['zico'] = np.nan

    info['flops']  = net.get_FLOPs(args.input_image_size)
    info['params'] = net.get_model_size()
    info['random'] = np.random.randn()

    del net
    torch.cuda.empty_cache()
    return info



def getmisc(args):
    if args.dataset == "cifar10":
        root = args.datapath
        imgsize = 32
    elif args.dataset == "cifar100":
        root = args.datapath
        imgsize = 32
    # NOTE: interesting didn't know imagenet-1k was in here
    elif args.dataset.startswith("imagenet-1k"):
        root = args.datapath
        imgsize = 224
    elif args.dataset.startswith("ImageNet16"):
        root = args.datapath
        imgsize = 16
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # NOTE: for GB proxies may not be able to just change datasets easily
    train_data, test_data, xshape, class_num = datasets.get_datasets(
        args.dataset, root, 0
    )
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_worker
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_worker
    )
    return trainloader, testloader, xshape, class_num

def main(args, argv):
    gpu = args.gpu
    if gpu is not None:
        print(torch.cuda.device_count())
        torch.cuda.set_device('cuda:{}'.format(gpu))
    print(args)

    trainloader, testloader, xshape, class_num = getmisc(args)

    if args.rand_input:
        print("Use random input")
        trainbatches = None
    else:
        print("Use real input")
        trainbatches = []
        for batchid, batch in enumerate(trainloader):
            if batchid == args.maxbatch:
                break
            datax, datay = batch
            datax = datax.cuda(non_blocking=True)
            datay = datay.cuda(non_blocking=True)
            trainbatches.append([datax, datay])

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt'new)
    if os.path.isfile(best_structure_txt):
        print(f'skip {best_structure_txt}')
        return None

    select_search_space = global_utils.load_py_module_from_path(args.search_space)
    AnyPlainNet = Masternet.MasterNet

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    initial_structure_str = str(masternet)

    popu_structure_list = []
    popu_zero_shot_score_dict = None
    popu_latency_list = []
    popu_zero_shot_score_list = None

    start_timer = time.time()
    lossfunc = nn.CrossEntropyLoss().cuda()

    for loop_count in tqdm(range(0, args.evolution_max_iter), desc="Evolution iterations"):
        # mutate or new struct
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet, initial_structure_str, args.num_classes,
                select_search_space.gen_search_space, num_replaces=1
            )
        elif len(popu_structure_list) < args.population_size - 1:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet, popu_structure_list[tmp_idx], args.num_classes,
                select_search_space.gen_search_space, num_replaces=2
            )
        else:
            # mutate top half
            if popu_zero_shot_score_list is not None:
                top_half_index = np.argsort(popu_zero_shot_score_list, axis=0)[-args.population_size+1:]
                tmp_idx = np.random.choice(top_half_index)
                base_str = popu_structure_list[tmp_idx]
            else:
                base_str = initial_structure_str
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet, base_str, args.num_classes,
                select_search_space.gen_search_space, num_replaces=2
            )

        random_structure_str = get_splitted_structure_str(AnyPlainNet,
                                                          random_structure_str,
                                                          args.num_classes)

        # measure zero-shot proxies
        t0 = time.time()
        # NOTE: here we score the new single new structure across all proxies (not batched.)
        the_nas_core = compute_all_proxies(AnyPlainNet, random_structure_str,
                                           gpu, args, trainbatches, lossfunc)
        dt = time.time() - t0

        # store
        if popu_zero_shot_score_dict is None:
            popu_zero_shot_score_dict = {}
            for k in the_nas_core.keys():
                popu_zero_shot_score_dict[k] = []

        for k, v in the_nas_core.items():
            popu_zero_shot_score_dict[k].append(v)

        popu_structure_list.append(random_structure_str)
        popu_latency_list.append(np.inf)  # or store real latency if you wish

        # re-rank using AZ proxies
        az_keys = ['expressivity','progressivity','trainability','complexity']
        popu_zero_shot_score_list = None
        for key in az_keys:
            arr = popu_zero_shot_score_dict[key]
            N = len(arr)
            arr_rank = stats.rankdata(arr)
            if popu_zero_shot_score_list is not None:
                popu_zero_shot_score_list += np.log(arr_rank / N)
            else:
                popu_zero_shot_score_list = np.log(arr_rank / N)
        popu_zero_shot_score_list = popu_zero_shot_score_list.tolist()


        if len(popu_structure_list) > args.population_size:
            sorted_idx = np.argsort(popu_zero_shot_score_list)
            top_idx = sorted_idx[-args.population_size:]
            popu_structure_list  = [ popu_structure_list[i]  for i in top_idx ]
            popu_latency_list = [ popu_latency_list[i]    for i in top_idx ]
            popu_zero_shot_score_list = [ popu_zero_shot_score_list[i] for i in top_idx ]
            for k in popu_zero_shot_score_dict.keys():
                old_list = popu_zero_shot_score_dict[k]
                popu_zero_shot_score_dict[k] = [ old_list[i] for i in top_idx ]

        # only do correlation every CORR_INTERVAL steps if we have >=2 arch
        if loop_count>0 and (loop_count % CORR_INTERVAL==0):
            scores = np.stack([
                popu_zero_shot_score_dict[k]
                for k in sorted(popu_zero_shot_score_dict.keys())
            ], 1)

            if scores.shape[0]<2:
                # skip
                print(f"[info] skip correlation at loop={loop_count}, only {scores.shape[0]} arch so far.")
            else:
                # handle NaN in scores
                scores = np.nan_to_num(scores, nan=0.0)

                # Spearman with warnings suppressed
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    result = stats.spearmanr(scores, axis=0)
                S = result.correlation
                if np.ndim(S)==0:
                    # only 1 column => fallback
                    dim = scores.shape[1]
                    S = np.eye(dim, dtype=np.float32)

                # Clean up S in case it has inf or NaN
                S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0)
                # Force symmetry
                S = 0.5*(S + S.T)
                # Bump diagonal if <=0
                for i in range(S.shape[0]):
                    if S[i,i] < 1e-12:
                        S[i,i] = 1e-12

                # Save S
                np.save(os.path.join(args.save_dir, f"S_{loop_count}.npy"), S)

                # Kendall
                dims = scores.shape[1]
                K = np.zeros((dims,dims), dtype=np.float32)
                for i in range(dims):
                    for j in range(dims):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            ktau = stats.kendalltau(scores[:, i], scores[:, j]).correlation
                        if np.isnan(ktau):
                            ktau=0.0
                        K[i,j]=ktau
                np.save(os.path.join(args.save_dir, f"K_{loop_count}.npy"), K)

                # mutual info
                bin_list=[]
                for j in range(dims):
                    col_j = scores[:, j]
                    try:
                        col_bins = pd.qcut(col_j, q=10, labels=False, duplicates='drop')
                    except ValueError:
                        col_bins = np.zeros_like(col_j, dtype=int)
                    col_bins = np.nan_to_num(col_bins, nan=-1)
                    bin_list.append(col_bins)

                M = np.zeros_like(S, dtype=np.float32)
                for i in range(dims):
                    for j in range(dims):
                        try:
                            M[i,j] = mutual_info_score(bin_list[i], bin_list[j])
                        except ValueError:
                            M[i,j] = 0.0
                np.save(os.path.join(args.save_dir, f"M_{loop_count}.npy"), M)

                # eigenvalues
                try:
                    lamb = np.linalg.eigvalsh(S)[::-1]
                except np.linalg.LinAlgError:
                    print(f"[warn] eigenvalues did not converge at loop={loop_count}, fallback to zeros")
                    lamb = np.zeros(S.shape[0], dtype=np.float32)
                np.save(os.path.join(args.save_dir, f"lambda_{loop_count}.npy"), lamb)

                # median abs corr
                upper = np.triu_indices_from(S, k=1)
                r_g = np.median(np.abs(S[upper])) if len(upper[0])>0 else 0.0
                np.save(os.path.join(args.save_dir, f"r_{loop_count}.npy"), r_g)

        loop_count+=1

    # done
    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list


if __name__=='__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    if args.seed is not None:
        logging.info(f"The seed number is set to {args.seed}")
        logging.info("This is a test from Andrew")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        os.environ["PYTHONHASHSEED"]=str(args.seed)

    info = main(args, sys.argv)
    if info is None:
        exit()

    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info
    if popu_zero_shot_score_list is None or len(popu_zero_shot_score_list)<1:
        print("[done] no valid arch found?")
        exit()

    best_score = max(popu_zero_shot_score_list)
    best_idx   = popu_zero_shot_score_list.index(best_score)
    best_arch  = popu_structure_list[best_idx]
    best_txt   = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_txt)
    with open(best_txt, 'w') as fid:
        fid.write(best_arch)
    print(f"[done] best arch = {best_arch}")


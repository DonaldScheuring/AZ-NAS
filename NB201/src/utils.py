import os
import torch
from nats_bench import create
from xautodl.models import get_search_spaces

from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)

def prepare_environment(args):
    """Prepares the environment, including logger, CUDA, and random seed."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = prepare_logger(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    prepare_seed(args.rand_seed)
    logger.log(f"Environment prepared with seed: {args.rand_seed}")
    return logger

def get_nasbench201_api(api_data_path, search_space, logger):
    """Initializes and returns the NAS-Bench-201 API."""
    api = create(api_data_path, search_space, fast_mode=True, verbose=False)
    logger.log(f"NAS-Bench-201 API created: {api_data_path}")
    return api

def get_search_space_natsbench(logger, args):
    """Returns the defined search space."""
    search_space = get_search_spaces(args.search_space, "nats-bench")
    logger.log(f"Search space: {search_space}")
    return search_space
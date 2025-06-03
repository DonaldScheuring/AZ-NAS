import torch
import numpy as np
import tqdm

from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.config_utils import load_config
from custom.tss_model import TinyNetwork
from ZeroShotProxy import * # Assuming this module defines compute_*_score functions

class ArchEvaluator:
    """Handles the evaluation of neural architectures using zero-cost proxies and NAS-Bench-201 API."""

    def __init__(self, device, logger, args, class_num=None, real_input_metrics=None):
        self.device = device
        self.logger = logger
        self.args = args
        self.class_num = class_num
        self.real_input_metrics = real_input_metrics if real_input_metrics is not None else []

        self._setup_data_loaders()
        score_fn_name = "compute_{}_score".format(self.args.zero_shot_score.lower())
        self.score_fn = globals().get(score_fn_name)
        if not self.score_fn:
            raise ValueError(f"Zero-shot score function '{self.args.zero_shot_score}' not found in ZeroShotProxy.py")

    def _setup_data_loaders(self):
        """Sets up the necessary data loaders."""
        train_data, valid_data, xshape, class_num = get_datasets(self.args.dataset, self.args.data_path, -1)
        self.class_num = class_num
        config = load_config(self.args.config_path, {"class_num": class_num, "xshape": xshape}, self.logger)
        self.search_loader, self.train_loader, self.valid_loader = get_nas_search_loaders(
            train_data, valid_data, self.args.dataset, "./configs/nas-benchmark/",
            (self.args.batch_size, config.test_batch_size), self.args.workers
        )
        self.input_, self.target_ = next(iter(self.train_loader))
        self.resolution = self.input_.size(2)

    def compute_zero_cost_score(self, arch):
        """Computes the zero-cost proxy score for a given architecture."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        network = TinyNetwork(self.args.channel, self.args.num_cells, arch, self.class_num)
        network = network.to(self.device)
        network.train()

        trainloader = self.train_loader if self.args.zero_shot_score.lower() in self.real_input_metrics else None

        info_dict = self.score_fn.compute_nas_score(
            network,
            gpu=self.device.index,
            trainloader=trainloader,
            resolution=self.resolution,
            batch_size=self.args.batch_size
        )
        return info_dict[self.args.zero_shot_score.lower()]

    def get_accuracy_from_api(self, api, arch):
        """Retrieves validation accuracy from NAS-Bench-201 API."""
        dataset_map = {
            'cifar10': 'cifar10',
            'cifar100': 'cifar100',
            'ImageNet16-120': 'ImageNet16-120'
        }
        api_dataset_name = dataset_map.get(self.args.dataset, 'cifar10') # Default to cifar10 if not found

        index = api.query_index_by_arch(arch)
        api._prepare_info(index)
        archresult = api.arch2infos_dict[index]['200']

        if api_dataset_name == 'cifar10':
            acc = archresult.get_metrics(api_dataset_name, 'ori-test', iepoch=None, is_random=False)['accuracy']
        elif api_dataset_name == 'cifar100' or api_dataset_name == 'ImageNet16-120':
            acc = archresult.get_metrics(api_dataset_name, 'x-test', iepoch=None, is_random=False)['accuracy']
        else: # For 'cifar10-valid' if it were an option, though not in your original map.
            acc = archresult.get_metrics(api_dataset_name, 'x-valid', iepoch=None, is_random=False)['accuracy']

        return acc
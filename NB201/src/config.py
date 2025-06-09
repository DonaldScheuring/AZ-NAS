import argparse
import random

def get_args():
    parser = argparse.ArgumentParser("Training-free NAS on NAS-Bench-201 (NATS-Bench-TSS)")

    # Dataset and Search Space
    parser.add_argument("--data_path", type=str, default='./cifar.python', help="The path to dataset")
    parser.add_argument("--dataset", type=str, default='cifar10', choices=["cifar10", "cifar100", "ImageNet16-120"], help="Choose between Cifar10/100 and ImageNet-16.")
    parser.add_argument("--search_space", type=str, default='tss', help="The search space name.")
    parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
    parser.add_argument("--affine", type=int, default=1, choices=[0, 1], help="Whether use affine=True or False in the BN layer.")
    parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1], help="Whether use track_running_stats or not in the BN layer.")
    parser.add_argument("--config_path", type=str, default='./configs/nas-benchmark/algos/weight-sharing.config', help="The path to the configuration.")

    # Zero-cost Proxy
    parser.add_argument('--zero_shot_score', type=str, default='az_nas', choices=['az_nas','zico','zen','gradnorm','naswot','synflow','snip','grasp','te_nas','gradsign'], help="Zero-cost proxy score to use.")

    # Hardware and Logging
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id.")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--api_data_path", type=str, default="./api_data/NATS-tss-v1_0-3ffb9-simple/", help="Path to NATS-Bench-201 API data.")
    parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")
    parser.add_argument("--rand_seed", type=int, default=1, help="Manual seed (we use 1-to-5).")
    parser.add_argument("--print_freq", type=int, default=200, help="Print frequency (default: 200).")

    # Search Algorithm specific arguments
    parser.add_argument("--search_algorithm", type=str, default='random', choices=['random', 'evolutionary'], help="Choose the NAS search algorithm.")

    # Random Search specific
    parser.add_argument("--n_samples", type=int, default=3000, help="Number of architectures to sample for random search.")

    # Evolutionary Algorithm specific
    parser.add_argument("--population_size", type=int, default=50, help="Population size for evolutionary algorithm.")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations for evolutionary algorithm.")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for evolutionary algorithm.")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover rate for evolutionary algorithm.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for data loaders.") # Added this as it was implicitly used before

    args = parser.parse_args() # For Jupyter compatibility, remove `args=[]` for command line

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)

    print(args) # Print arguments for verification
    return args
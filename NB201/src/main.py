import os
import torch
from src.config import get_args
from src.utils import prepare_environment, get_nasbench201_api, get_search_space_natsbench
from src.evaluation import ArchEvaluator
from src.search_algorithms import RandomSearch, EvolutionarySearch # Will add EvolutionarySearch later


def main():
    args = get_args()
    logger = prepare_environment(args)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    api = get_nasbench201_api(args.api_data_path, args.search_space, logger)
    search_space = get_search_space_natsbench(logger, args)

    evaluator = ArchEvaluator(
        device=device,
        logger=logger,
        args = args,
        class_num=None, # Will be set internally by ArchEvaluator
        real_input_metrics=['zico', 'snip', 'grasp', 'te_nas', 'gradsign']
    )

    logger.log(f"Starting {args.search_algorithm} search...")

    if args.search_algorithm == 'random':
        searcher = RandomSearch(
            api=api,
            evaluator=evaluator,
            search_space=search_space,
            max_nodes=args.max_nodes,
            n_samples=args.n_samples, # Assuming n_samples is now in args
            dataset=args.dataset,
            logger=logger
        )
        best_arch, best_acc = searcher.run()
    elif args.search_algorithm == 'evolutionary':
        searcher = EvolutionarySearch(
            api=api,
            evaluator=evaluator,
            search_space=search_space,
            max_nodes=args.max_nodes,
            dataset=args.dataset,
            logger=logger,
            population_size=args.population_size, # New arg for EA
            generations=args.generations,         # New arg for EA
            mutation_rate=args.mutation_rate,     # New arg for EA
            crossover_rate=args.crossover_rate    # New arg for EA
        )
        best_arch, best_acc = searcher.run()
    else:
        raise ValueError(f"Unknown search algorithm: {args.search_algorithm}")

    logger.log("-" * 50)
    logger.log(f"Search Finished! Best architecture found (validation accuracy): {best_acc:.2f}%")
    if api is not None:
        logger.log(f"Best architecture info from API: {api.query_by_arch(best_arch, '200')}")
    logger.log("-" * 50)


if __name__ == "__main__":
    main()
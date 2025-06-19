import random
import numpy as np
import tqdm
import wandb

from src.arch_sampler import ArchSampler


# ----------------------------------------------------------------------
# Base class shared by all strategies
# ----------------------------------------------------------------------
class BaseSearcher:
    def __init__(self, api, evaluator, search_space, max_nodes,
                 dataset, logger):
        self.api          = api
        self.evaluator    = evaluator
        self.search_space = search_space
        self.max_nodes    = max_nodes
        self.op_names     = search_space
        self.dataset      = dataset
        self.logger       = logger

        self.best_arch = None       # by *true* accuracy (for reference)
        self.best_acc  = -1.0

    def _update_best(self, arch, accuracy):
        if accuracy > self.best_acc:
            self.best_acc  = accuracy
            self.best_arch = arch
            self.logger.log(f"New best (oracle) accuracy = {accuracy:.2f} %")


# ======================================================================
# 1. RANDOM BASELINE – score = accuracy of **one** random architecture
# ======================================================================
class RandomSearch(BaseSearcher):
    """
    Draw a single architecture uniformly at random and use its validation
    accuracy as the run's metric.  The extra `n_samples` argument is kept
    for API compatibility but only the *first* sample is used.
    """
    def __init__(self, api, evaluator, search_space, max_nodes,
                 n_samples, dataset, logger):
        super().__init__(api, evaluator, search_space, max_nodes,
                         dataset, logger)
        self.n_samples = n_samples

    # --------------------------------------------------------------
    def run(self):
        self.logger.log("Random baseline: selecting ONE uniform architecture")

        # one‑shot sample
        arch = ArchSampler.random_genotype(self.max_nodes, self.op_names)
        acc  = self.evaluator.get_accuracy_from_api(self.api, arch)
        wandb.log({"random_acc": acc})

        self._update_best(arch, acc)          # only for record keeping
        return arch, acc


# ======================================================================
# 2. PROXY‑DRIVEN EVOLUTION – unchanged from previous version
# ======================================================================
class EvolutionarySearch(BaseSearcher):
    def __init__(self, api, evaluator, search_space, max_nodes,
                 dataset, logger,
                 population_size, generations,
                 mutation_rate, crossover_rate):
        super().__init__(api, evaluator, search_space, max_nodes,
                         dataset, logger)
        self.population_size = population_size
        self.generations     = generations
        self.mutation_rate   = mutation_rate
        self.crossover_rate  = crossover_rate

    def run(self):
        self.logger.log(f"EA: {self.generations} gens, pop={self.population_size}")

        population = [ArchSampler.random_genotype(self.max_nodes,
                                                  self.op_names)
                      for _ in range(self.population_size)]

        last_fitness   = None
        last_population = None

        for gen in range(self.generations):
            self.logger.log(f"Generation {gen+1}/{self.generations}")

            fitness = []
            for arch in tqdm.tqdm(population, desc="Evaluating population"):
                score = self.evaluator.compute_zero_cost_score(arch)
                fitness.append(score)

            fitness = np.nan_to_num(np.asarray(fitness, np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
            if fitness.min() < 0:
                fitness -= fitness.min()

            last_fitness   = fitness.copy()
            last_population = population.copy()

            probs = (fitness / fitness.sum()
                     if fitness.sum() > 0 else
                     np.ones_like(fitness) / len(fitness))

            # oracle monitoring
            arch_oracle = population[int(fitness.argmax())]
            acc_oracle  = self.evaluator.get_accuracy_from_api(self.api,
                                                               arch_oracle)
            self._update_best(arch_oracle, acc_oracle)

            parents = random.choices(population, weights=probs,
                                     k=self.population_size)
            children = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[i+1] if i+1 < len(parents) else random.choice(population)

                if random.random() < self.crossover_rate:
                    c1, c2 = ArchSampler.crossover_archs(p1, p2)
                else:
                    c1, c2 = p1, p2

                if random.random() < self.mutation_rate:
                    c1 = ArchSampler.mutate_arch(c1, self.op_names, self.max_nodes)
                if random.random() < self.mutation_rate:
                    c2 = ArchSampler.mutate_arch(c2, self.op_names, self.max_nodes)
                children.extend([c1, c2])

            population = children[:self.population_size]

        # score = accuracy of the arch with highest proxy in final gen
        idx_best = int(last_fitness.argmax())
        final_arch = last_population[idx_best]
        final_acc  = self.evaluator.get_accuracy_from_api(self.api,
                                                          final_arch)
        self.logger.log(f"EA result — proxy‑best arch accuracy = "
                        f"{final_acc:.2f} %")
        return final_arch, final_acc


# ======================================================================
# 3. ORACLE EVOLUTION – unchanged
# ======================================================================
class EvolutionarySearchOracle(EvolutionarySearch):
    """EA whose fitness is true accuracy (upper‑bound baseline)."""
    def run(self):
        self.logger.log(f"[ORACLE EA] {self.generations} gens, pop={self.population_size}")

        population = [ArchSampler.random_genotype(self.max_nodes,
                                                  self.op_names)
                      for _ in range(self.population_size)]
        last_fitness   = None
        last_population = None

        for gen in range(self.generations):
            accs = []
            for arch in tqdm.tqdm(population, desc=f"Gen {gen+1} eval"):
                acc = self.evaluator.get_accuracy_from_api(self.api, arch)
                accs.append(acc)
                wandb.log({"true_acc": acc, "gen": gen})

            accs = np.asarray(accs, np.float64)
            probs = accs/accs.sum() if accs.sum() > 0 else np.ones_like(accs)/len(accs)

            last_fitness    = accs.copy()
            last_population = population.copy()

            self._update_best(population[int(accs.argmax())], float(accs.max()))

            parents = random.choices(population, weights=probs,
                                     k=self.population_size)
            children = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[i+1] if i+1 < len(parents) else random.choice(population)

                if random.random() < self.crossover_rate:
                    c1, c2 = ArchSampler.crossover_archs(p1, p2)
                else:
                    c1, c2 = p1, p2

                if random.random() < self.mutation_rate:
                    c1 = ArchSampler.mutate_arch(c1, self.op_names, self.max_nodes)
                if random.random() < self.mutation_rate:
                    c2 = ArchSampler.mutate_arch(c2, self.op_names, self.max_nodes)
                children.extend([c1, c2])

            population = children[:self.population_size]

        idx = int(last_fitness.argmax())
        arch_final = last_population[idx]
        acc_final  = float(last_fitness[idx])

        self.logger.log(f"[ORACLE EA] final score = {acc_final:.2f} %")
        return arch_final, acc_final


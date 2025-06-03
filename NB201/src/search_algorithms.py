import numpy as np
import tqdm
import random
from src.arch_sampler import ArchSampler

class BaseSearcher:
    """Base class for NAS search algorithms."""
    def __init__(self, api, evaluator, search_space, max_nodes, dataset, logger):
        self.api = api
        self.evaluator = evaluator
        self.search_space = search_space
        self.max_nodes = max_nodes
        self.op_names = search_space
        self.dataset = dataset
        self.logger = logger
        self.best_arch = None
        self.best_acc = -1.0

    def run(self):
        raise NotImplementedError

    def _update_best(self, arch, accuracy):
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_arch = arch
            self.logger.log(f"New best found: Accuracy = {accuracy:.2f}%")

class RandomSearch(BaseSearcher):
    """Implements the Random Search algorithm."""
    def __init__(self, api, evaluator, search_space, max_nodes, n_samples, dataset, logger):
        super().__init__(api, evaluator, search_space, max_nodes, dataset, logger)
        self.n_samples = n_samples

    def run(self):
        self.logger.log(f"Starting Random Search for {self.n_samples} samples.")
        for i in tqdm.tqdm(range(self.n_samples)):
            arch = ArchSampler.random_genotype(self.max_nodes, self.op_names)
            accuracy = self.evaluator.get_accuracy_from_api(self.api, arch)
            self._update_best(arch, accuracy)
        return self.best_arch, self.best_acc


class EvolutionarySearch(BaseSearcher):
    """Implements an Evolutionary Algorithm for NAS."""
    def __init__(self, api, evaluator, search_space, max_nodes, dataset, logger,
                 population_size, generations, mutation_rate, crossover_rate):
        super().__init__(api, evaluator, search_space, max_nodes, dataset, logger)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def run(self):
        self.logger.log(f"Starting Evolutionary Search for {self.generations} generations with population size {self.population_size}.")

        # 1. Initialize Population
        population = []
        for _ in range(self.population_size):
            arch = ArchSampler.random_genotype(self.max_nodes, self.op_names)
            population.append(arch)

        for generation in range(self.generations):
            self.logger.log(f"Generation {generation+1}/{self.generations}")

            # 2. Evaluate Population (using zero-cost proxy for fitness)
            fitness_scores = []
            for arch in tqdm.tqdm(population, desc="Evaluating population"):
                score = self.evaluator.compute_zero_cost_score(arch)
                fitness_scores.append(score)

            fitness_scores = np.array(fitness_scores, dtype=np.float64)
            fitness_scores = np.nan_to_num(fitness_scores, nan=0.0, posinf=0.0, neginf=0.0)

            min_score = np.min(fitness_scores)
            if min_score < 0:
                fitness_scores = fitness_scores - min_score

            total_score = np.sum(fitness_scores)
            if total_score <= 0:
                selection_probs = np.ones_like(fitness_scores) / len(fitness_scores)
            else:
                selection_probs = fitness_scores / total_score
            
            self.logger.log(f"Selection probs: {selection_probs}")

            # Update best architecture based on actual validation accuracy (optional, but good for tracking)
            # For real-world use, you might evaluate only the best architecture from the population with full training
            current_best_arch_gen = population[np.argmax(fitness_scores)]
            current_best_acc_gen = self.evaluator.get_accuracy_from_api(self.api, current_best_arch_gen)
            self._update_best(current_best_arch_gen, current_best_acc_gen)

            # 3. Selection (Roulette Wheel Selection)
            selected_parents = random.choices(population, weights=selection_probs, k=self.population_size)

            # 4. Crossover and Mutation to create next generation
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i+1] if i+1 < self.population_size else random.choice(population) # Handle odd population size

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = ArchSampler.crossover_archs(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = ArchSampler.mutate_arch(child1, self.op_names, self.max_nodes)
                if random.random() < self.mutation_rate:
                    child2 = ArchSampler.mutate_arch(child2, self.op_names, self.max_nodes)

                next_population.extend([child1, child2])

            population = next_population[:self.population_size] # Ensure population size is maintained

        # Final evaluation of the best architecture found
        final_best_acc = self.evaluator.get_accuracy_from_api(self.api, self.best_arch)
        self.logger.log(f"Final best architecture from Evolutionary Search has validation accuracy: {final_best_acc:.2f}%")

        return self.best_arch, final_best_acc
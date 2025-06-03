import random
from xautodl.models.cell_searchs.genotypes import Structure

class ArchSampler:
    """Provides methods for sampling neural architectures."""

    @staticmethod
    def random_genotype(max_nodes, op_names):
        """Generates a random architecture genotype."""
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

    @staticmethod
    def mutate_arch(arch, op_names, max_nodes):
        """Applies a mutation to an architecture genotype."""
        new_genotypes = []
        for i in range(len(arch.genotypes)):
            node_ops = list(arch.genotypes[i])
            # Randomly select a connection to mutate
            idx_to_mutate = random.randint(0, len(node_ops) - 1)
            # Change the operation for the selected connection
            original_op, original_node = node_ops[idx_to_mutate]
            new_op = random.choice(op_names)
            while new_op == original_op: # Ensure a different operation is chosen
                new_op = random.choice(op_names)
            node_ops[idx_to_mutate] = (new_op, original_node)
            new_genotypes.append(tuple(node_ops))
        return Structure(new_genotypes)

    @staticmethod
    def crossover_archs(parent1_arch, parent2_arch):
        """Performs crossover between two architecture genotypes."""
        # Simple one-point crossover of the genotype list
        genotypes1 = list(parent1_arch.genotypes)
        genotypes2 = list(parent2_arch.genotypes)

        min_len = min(len(genotypes1), len(genotypes2))
        if min_len < 2: # Cannot perform crossover if genotypes are too short
            return parent1_arch, parent2_arch # Return original parents

        crossover_point = random.randint(1, min_len - 1)

        child1_genotypes = genotypes1[:crossover_point] + genotypes2[crossover_point:]
        child2_genotypes = genotypes2[:crossover_point] + genotypes1[crossover_point:]

        return Structure(child1_genotypes), Structure(child2_genotypes)
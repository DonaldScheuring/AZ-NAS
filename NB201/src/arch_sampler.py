import random
from xautodl.models.cell_searchs.genotypes import Structure # Assuming this is the correct path for the Structure class

class ArchSampler:
    """Provides methods for sampling, mutating, and crossing over neural architectures.
       Architectures are represented as xautodl.models.cell_searchs.genotypes.Structure objects.
    """

    @staticmethod
    def random_genotype(max_nodes, op_names):
        """Generates a random architecture genotype (Structure object)."""
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        arch = Structure(genotypes)
        return arch

    @staticmethod
    def mutate_arch(arch: Structure, op_names, max_nodes) -> Structure:
        """Applies a mutation to an architecture genotype (Structure object)."""
        # Convert Structure to its string representation for easier manipulation
        arch_str = arch.tostr()
        nodes_str = arch_str.split('+') # Each element is a node's connections, e.g., "|op1~0|op2~1|"

        # Select a random node to mutate
        node_idx_to_mutate = random.randint(0, len(nodes_str) - 1)
        target_node_str = nodes_str[node_idx_to_mutate]

        # Extract individual operations and their connections from the target node
        connections = [conn.split('~') for conn in target_node_str.strip('|').split('|') if conn]
        if not connections: # Handle cases where a node might be empty (shouldn't happen with current random_genotype)
            return arch # No connections to mutate

        # Select a random connection within the chosen node to mutate
        conn_idx_to_mutate = random.randint(0, len(connections) - 1)
        original_op = connections[conn_idx_to_mutate][0]
        original_input_node = connections[conn_idx_to_mutate][1]

        # Choose a new operation
        new_op = random.choice(op_names)
        while new_op == original_op: # Ensure a different operation is chosen
            new_op = random.choice(op_names)

        # Update the connection
        connections[conn_idx_to_mutate] = [new_op, original_input_node]

        # Reconstruct the target node string
        new_target_node_str_parts = ["{}~{}".format(op, idx) for op, idx in connections]
        new_target_node_str = "|{}|".format("|".join(new_target_node_str_parts))

        # Update the main architecture string
        nodes_str[node_idx_to_mutate] = new_target_node_str
        new_arch_str = "+".join(nodes_str)

        # Convert back to Structure object
        return Structure.str2structure(new_arch_str)

    @staticmethod
    def crossover_archs(parent1_arch: Structure, parent2_arch: Structure) -> tuple[Structure, Structure]:
        """Performs crossover between two architecture genotypes (Structure objects)."""
        parent1_str = parent1_arch.tostr()
        parent2_str = parent2_arch.tostr()

        nodes1 = parent1_str.split('+')
        nodes2 = parent2_str.split('+')

        min_nodes = min(len(nodes1), len(nodes2))
        if min_nodes < 2: # Cannot perform meaningful crossover if not enough nodes
            return parent1_arch, parent2_arch # Return original parents

        crossover_point = random.randint(1, min_nodes - 1) # Crossover point between nodes

        # Perform crossover on the list of node strings
        child1_nodes = nodes1[:crossover_point] + nodes2[crossover_point:]
        child2_nodes = nodes2[:crossover_point] + nodes1[crossover_point:]

        # Reconstruct full architecture strings
        child1_str = "+".join(child1_nodes)
        child2_str = "+".join(child2_nodes)

        # Convert back to Structure objects
        child1_arch = Structure.str2structure(child1_str)
        child2_arch = Structure.str2structure(child2_str)

        return child1_arch, child2_arch
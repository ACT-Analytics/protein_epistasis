import pandas as pd


def construct_mutations_to_index_map(mutants: list[list[str]] | pd.Series) -> dict[frozenset[str], int]:
    """Constructs a mapping from each mutant frozenset to its row index."""
    return {frozenset(mutant): i for i, mutant in enumerate(mutants)}


def construct_index_to_mutations_map(mutations_to_index_map: dict[frozenset[str], int]) -> dict[int, frozenset[str]]:
    """Constructs a mapping from each index to its corresponding mutant frozenset."""
    return {index: mutant for mutant, index in mutations_to_index_map.items()}


def construct_mutation_number_map(mutations: list[list[str]] | pd.Series) -> dict[int, dict[frozenset[str], int]]:
    """Groups mutants by mutation count -> {num_mutations: {frozenset: index}}."""
    mutation_number_map = {}
    for i, mutant in enumerate(mutations): 
        f_mutant = frozenset(mutant)
        num_mutations = len(f_mutant)
        mutation_number_map.setdefault(num_mutations, {})[f_mutant] = i
    return mutation_number_map


def is_parent_sequence(parent: frozenset[str], child: frozenset[str]) -> bool:
    """Checks if 'parent' is a subset of 'child' with exactly 1 fewer mutation."""
    return parent.issubset(child) and len(parent) + 1 == len(child)


def find_parent_sequences(
    sequence: frozenset[str], 
    mutation_number_map: dict[int, dict[frozenset[str], int]]
) -> tuple[frozenset[str], frozenset[str]] | None:
    """
    Finds the pair of single-mutant parents for a double mutant.
    Returns (parent1, parent2) ONLY if BOTH single mutants exist in the dataset.
    """
    num_mutations = len(sequence)
    potential_parents = mutation_number_map.get(num_mutations - 1, {})

    if not potential_parents:
        return None

    for parent1 in potential_parents:
        if is_parent_sequence(parent1, sequence):
            parent2 = sequence - parent1
            # Verify BOTH single-mutant parents exist in the dataset
            if parent2 in potential_parents:
                return (parent1, parent2)

    return None


def construct_child_to_parent_map(
    mutation_number_map: dict[int,dict[frozenset[str], int]], 
    level: int = 2
) -> dict[frozenset[str], tuple[frozenset[str], frozenset[str]]]: 
    """Creates a mapping from child double mutant to its two single-mutant parents."""
    children = mutation_number_map.get(level, {})
    double_to_single = {}

    for child in children:
        parents = find_parent_sequences(child, mutation_number_map)
        if parents:  # Safe check: skips double mutants with missing single parents
            double_to_single[child] = parents

    return double_to_single


def get_single_mutants(sequence: frozenset, mutation_number_map:dict): 
    return [frozenset(single_mutant) for single_mutant in sequence
            if frozenset(single_mutant) in mutation_number_map[1] ]


def construct_mutation_to_single_mutation_map(
    mutation_number_map: dict[int, dict[frozenset[str], int]], 
    level: int = 2
    ) -> dict[frozenset[str], tuple[frozenset[str], frozenset[str]]]: 
    """
    Finds all single mutants that contribute to mutant
    Returns tuple with frozen sets
    """

    mutations = mutation_number_map.get(level, {})
    mutation_to_single_mutants = {}

    for mutant in mutations:
        single_mutants = get_single_mutants(mutant, mutation_number_map)
        if len(single_mutants) == len(mutant):  # Safe check: skips over mutants that don't have all single mutants in measured set. 
            mutation_to_single_mutants[mutant] = single_mutants

    return mutation_to_single_mutants
    

if __name__ == '__main__':
    def test_is_parent_sequence():
        parent = frozenset(['A1T', 'T2Y'])
        child = frozenset(['A1T', 'T2Y', 'L10K'])

        # Correct assertions
        assert is_parent_sequence(parent, child) == True
        assert is_parent_sequence(child, parent) == False
        print("is_parent_sequence assertions passed!")

    test_is_parent_sequence()
# Code for generating the epistasis graph and looking up mutants""

import pandas as pd

def construct_mutations_to_index_map(mutants: list[list[str]]| pd.Series[list[str]] ):
    """
    Constructs a mapping from each mutant to its index in the list of mutants.

    Args:
        mutants (list): A list of mutants.

    Returns:
        dict: A dictionary mapping each mutant to its index.
    """
    return {frozenset(mutant): i for i, mutant in enumerate(mutants)}


def construct_index_to_mutations_map(mutations_to_index_map: dict[frozenset, int]):
    """
    Constructs a mapping from each index to its corresponding mutant.

    Args:
        mutations_to_index_map (dict): A dictionary mapping each mutant to its index.   
    """

    return {index: mutant for mutant, index in mutations_to_index_map.items()}


def construct_mutation_number_map(mutations: list[list[str]]| pd.Series[list[str]]):
    """
    Constructs a mapping from each mutant to its number of mutations.

    Args:
        mutations (list): A list of mutants.

    Returns:
        dict: A dictionary with the number of mutations as keys and the corresponding mutants as values.
    """
    mutation_number_map = {}
    i = 0
    for mutant in mutations: 
        num_mutations = len(mutant)
        mutation_number_map.setdefault(num_mutations, {})
        mutation_number_map[num_mutations][frozenset(mutant)] = i
        i += 1
    return mutation_number_map


def is_parent_sequence(sequence: frozenset[str], other_sequence: frozenset[str]) -> bool:
    """
    Checks if a given sequence is a parent sequence of another sequence.

    Args:
        sequence (frozenset[str]): The sequence to check.
        other_sequence (frozenset[str]): The potential child sequence.
    Returns: 
        bool: True if the sequence is a parent of the other_sequence, False otherwise.
    """
    return sequence.issubset(other_sequence) and len(sequence) + 1 == len(other_sequence)


def find_parent_sequences(sequence: frozenset[str], mutation_number_map: dict[int, dict[frozenset[str], int]]) -> list[frozenset[str]]:
    """
    Finds all parent sequences of a given sequence.

    Args:
        sequence (frozenset[str]): The sequence to find parents for.
        mutation_number_map (dict): A dictionary mapping the number of mutations to the corresponding mutants.

    Returns:
        list: A list of parent sequences.
    """

    num_mutations = len(sequence)

    potential_parents = mutation_number_map.get(num_mutations - 1, {})
    if len(potential_parents):
        return [[parent, sequence-parent] for parent in potential_parents if is_parent_sequence(parent, sequence)]

    return []


def construct_child_to_parent_map(num_mutations_per_mutant_map: dict[int,dict[frozenset, int]], level:int = 2): 
    """ Create a mapping from child to parents"""

    children = num_mutations_per_mutant_map[level]

    double_to_single = {child:find_parent_sequences(child, num_mutations_per_mutant_map)[0] for child in children}

    return double_to_single
    
    


if __name__ == 'main':


    def test_is_parent_sequence():
        parent = frozenset(['A1T', 'T2Y'])
        child = frozenset(['A1T', 'T2Y', 'L10K'])

        assert is_parent_sequence(child, parent)

        assert is_parent_sequence(parent, child) == False

    print(test_is_parent_sequence())


    parent = frozenset(['A1T', 'T2Y'])
    child = frozenset(['A1T', 'T2Y', 'L10K'])

    print(child - parent)
    print('yo')


    



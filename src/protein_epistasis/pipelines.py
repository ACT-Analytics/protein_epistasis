## Pipelines: You know, for kids!

import pandas as pd

from protein_epistasis.lattice.graph import (
    construct_child_to_parent_map,
    # construct_index_to_mutations_map,
    construct_mutation_number_map,
    construct_mutation_to_single_mutation_map,
    # construct_mutations_to_index_map,
    # find_parent_sequences,
)
from protein_epistasis.metrics.scores import (
    construct_child_parent_dataframe,
    construct_sequence_to_score_map,
)


def construct_graph_dicts(
    df: pd.DataFrame,
    value_cols: list[str] | None = None,
    mutation_col: str = "mutant",
    linkage_type: str = "singles",
):
    """Constructs the dictionaries for creating the lowlevel graph representation

    Args:
        df: The dataframe with the data
        value_cols: the cols with the values of interest.
        mutation_col: The column with the mutation information
        linkage_type: how to construct graph possible are 'single_mutant', 'parents'

    Returns:
        [dict, dict]
    """
    if not value_cols: 
        value_cols = ['delta TM', 'KD FC','EC50 FI']

    num_mutations_per_mutant_map = construct_mutation_number_map(df[mutation_col])

    # should we do preprocessing here? consider it.
    mutation_map = {}
    max_n_mutants = df[mutation_col].apply(lambda x: len(x)).max()
    print(max_n_mutants)

    if linkage_type == "singles":
        for i in range(1, max_n_mutants+1):
            mutation_map.update(
                construct_mutation_to_single_mutation_map(
                    num_mutations_per_mutant_map, i
                )
            )

    if linkage_type == "parents":
        for i in range(1, max_n_mutants):
            mutation_map.update(
                construct_child_to_parent_map(num_mutations_per_mutant_map, i)
            )

    scores_dict = {}
    for val in value_cols:

        sequence_to_score = construct_sequence_to_score_map(df, score_col = val)
        scores_dict[val] = construct_child_parent_dataframe(mutation_map, sequence_to_score)

    return num_mutations_per_mutant_map, mutation_map, scores_dict


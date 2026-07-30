import pandas as pd
import re

def construct_sequence_to_score_map(
    df: pd.DataFrame,
    mutant_col: str = 'mutant',
    score_col: str = 'DMS_score',
    sep: str = ','
) -> dict[frozenset, float]: 
    """Creates a dict mapping frozenset mutant representations to scores.""" 
    score_map = {}
    for mutant, score in zip(df[mutant_col], df[score_col]):
        # Handle raw delimited strings vs. pre-split lists/tuples
        if isinstance(mutant, str):
            mutant_set = frozenset(m.strip() for m in mutant.split(sep) if m.strip())
        else:
            mutant_set = frozenset(mutant)
            
        score_map[mutant_set] = float(score)
        
    return score_map


def _frozen_to_string(frozen: frozenset, sep: str = ',') -> str:
    """Converts a frozenset of mutations into a position-sorted string."""
    def _extract_pos(mut_str: str) -> int:
        match = re.search(r'\d+', mut_str)
        return int(match.group()) if match else 0

    sorted_mutants = sorted(frozen, key=_extract_pos)
    return sep.join(sorted_mutants)


def construct_child_parent_dataframe(
    child_to_parent_map: dict[frozenset, tuple[frozenset, frozenset]], 
    score_map: dict[frozenset, float]
) -> pd.DataFrame:
    """Constructs a DataFrame pairing child double mutants with their parent single mutants."""
    rows = []

    for child, (parent1, parent2) in child_to_parent_map.items():  # Fixed .items()
        # Safely verify all required variants exist in score_map
        if child in score_map and parent1 in score_map and parent2 in score_map:
            rows.append({
                'Child Mutant': _frozen_to_string(child),
                'Parent 1': _frozen_to_string(parent1),
                'Parent 2': _frozen_to_string(parent2),
                'Y': score_map[child],
                'x1': score_map[parent1],
                'x2': score_map[parent2]
            })

    return pd.DataFrame(rows)


# Unit Tests
def test_frozen_to_string(): 
    test1 = frozenset({'T275A', 'Q228I'})
    assert _frozen_to_string(test1) == 'Q228I,T275A'

    test2 = frozenset({'T275A', 'Q228I', 'A102K'})
    assert _frozen_to_string(test2) == 'A102K,Q228I,T275A'
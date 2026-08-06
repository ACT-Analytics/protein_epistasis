import pandas as pd

from protein_epistasis.utils import _frozen_to_string


def construct_sequence_to_score_map(
    df: pd.DataFrame,
    mutant_col: str = 'mutant',
    score_col: str = 'DMS_score',
    sep: str = ',',
    use_frozen_set_for_keys: bool= True
) -> dict[frozenset, float]: 
    """Creates a dict mapping frozenset mutant representations to scores.
    
    TODO: support replicate-aware aggregation for duplicate mutation keys;
    current behavior keeps the last observed score per key by design
    for quick-turn analysis.

    TODO: skip invalid mutant values (e.g., NaN/None/non-iterable) and
    report a skipped-record count for data-quality visibility.
    """ 
    score_map = {}
    for mutant, score in zip(df[mutant_col], df[score_col]):
        # Handle raw delimited strings vs. pre-split lists/tuples
        
        if use_frozen_set_for_keys:
            if isinstance(mutant, str):
                mutant_set = frozenset(m.strip() for m in mutant.split(sep) if m.strip())
            else:
                mutant_set = frozenset(mutant)
                
            score_map[mutant_set] = float(score)
        else:
            score_map[mutant] = score
    return score_map


def _verify_keys_in_map(keys:list, map:dict): 
    """Verify every key in list in map"""

    for key in keys: 
        if key not in map:
            return False
    return True


def construct_child_parent_dataframe(
    child_to_parent_map: dict[frozenset, tuple[frozenset, frozenset]], 
    score_map: dict[frozenset, float], 
    parent_prefix:str = 'Parent'
) -> pd.DataFrame:
    """Constructs a DataFrame pairing child double mutants with their parent single mutants."""
    rows = []

    for child, parents in child_to_parent_map.items():  # Fixed .items()
        # Safely verify all required variants exist in score_map
        if child in score_map and _verify_keys_in_map(parents, score_map):
            values = {
                'Child Mutant': _frozen_to_string(child),
                'Y': score_map[child],
            }

            for i, parent in enumerate(parents): 
                values[f'{parent_prefix} {i+1}'] = _frozen_to_string(parent)
                values[f'x{i+1}'] = score_map[parent]
            rows.append(values)

    df =  pd.DataFrame(rows)
    col_order = (['Child Mutant'] + 
                 [col for col in df.columns if col.startswith(parent_prefix)] + 
                 ['Y'] + 
                 [col for col in df.columns if col.startswith('x')] )
    if len(df):
        return df[col_order]

    # NOTE: Returning None on empty output is intentionally accepted for now.
    # TODO: standardize this to return an empty DataFrame with stable columns.


# Unit Tests
def test_frozen_to_string(): 
    test1 = frozenset({'T275A', 'Q228I'})
    assert _frozen_to_string(test1) == 'Q228I,T275A'

    test2 = frozenset({'T275A', 'Q228I', 'A102K'})
    assert _frozen_to_string(test2) == 'A102K,Q228I,T275A'
    print('_frozen_to_string works!')


def test__verify_keys_in_map():

    keys1 = ['apple', 'bannana']
    keys2 = ['apple', 'bannana', 'pineapple']

    fruit_dict = {'apple':1, 'bannana':2, 'cherry':3}
    assert _verify_keys_in_map(keys1, fruit_dict)
    assert not _verify_keys_in_map(keys2, fruit_dict)
    print('_verify_keys_in_string works!')


if __name__ == '__main__':
    test__verify_keys_in_map()
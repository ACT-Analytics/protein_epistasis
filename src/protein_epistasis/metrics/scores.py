# For scoring
import pandas as pd

def construct_sequence_to_score_map(
    df: pd.DataFrame,
    mutant_col:str = 'mutant',
    score_col:str = 'DMS_score'
    ) -> dict[frozenset, float]: 
    
    """Creates a dict with mapping the mutants to scores.""" 
    
    return {frozenset(mutant): score for mutant, score in zip(df[mutant_col], df[score_col])}


def __frozen_to_string__(frozen: frozenset, sep:str = ',')->str:
    frozen_list = list(frozen)
    frozen_list.sort(key= lambda x: int(x[1:-1]))
    return f'{sep}'.join(frozen_list)


def construct_child_parent_dataframe(child_to_parent_map: dict, score_map:dict)-> pd.DataFrame:
    """construct the dataframe for doing actual plotting and data manipulation"""

    rows = []

    for child, (parent1, parent2) in child_to_parent_map.items(): 
        
        values = {'Child Mutant': __frozen_to_string__(child),
                  'Parent 1': __frozen_to_string__(parent1),
                  'Parent 2': __frozen_to_string__(parent2),
                  'Y': score_map[child],
                  'x1': score_map[parent1],
                  'x2': score_map[parent2]
        }
        rows.append(values)

    return pd.DataFrame.from_records(rows)

                  
        
        




        

def test__frozen_to_string__(): 
    test1 = frozenset({ 'T275A','Q228I'})
    assert __frozen_to_string__(test1) == 'Q228I,T275A'

    test2 = frozenset({ 'T275A','Q228I', 'A102K'})
    assert __frozen_to_string__(test2) == 'A102K,Q228I,T275A'
    

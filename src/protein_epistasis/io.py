# code for parsing inputs.abs

from pathlib import Path

import pandas as pd


def load_and_process_protein_gym_file(path: Path|str)->pd.DataFrame: 
    df = pd.read_csv(path)
    df['mutant'] = df['mutant'].str.split(':')
    df['n_mutations'] = df['mutant'].str.len()
    return df
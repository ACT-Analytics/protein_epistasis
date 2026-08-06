"""
Data Preprocessing Utilities for Mutation and Melting Temperature (Tm) Analysis.

This module provides clean, robust, and vectorized functions to preprocess 
dataframe records containing protein/gene mutation data, calculate parental averages, 
and derive delta Tm metrics.
"""

import numpy as np
import pandas as pd


def process_mutations(
    df: pd.DataFrame, 
    mutant_col: str = 'mutations', 
    sep: str = ',',
    replace_colon: bool= True, 
) -> pd.DataFrame:
    """
    Splits mutation strings into lists and counts the number of mutations per entry.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing sequence or variant information.
    mutant_col : str, default='mutations'
        Column name containing delimited mutation strings (e.g., "A12V,G34S").
    sep : str, default=','
        Delimiter string used to split the mutations.

    Returns:
    --------
    pd.DataFrame
        A copy of the DataFrame with new columns:
        - 'mutant': List of mutation strings.
        - 'num_mutations': Integer count of mutations.
    """
    df = df.copy()
    
    # Helper to handle missing values and empty string cases correctly
    def parse_mutations(val):
        if pd.isna(val) or not str(val).strip():
            return []
        # Split by `sep` (fixes previous bug where ',' was hardcoded)
        return [m.strip() for m in str(val).split(sep) if m.strip()]

    if replace_colon: 
        df[mutant_col] = df[mutant_col].str.replace(':', '_')

    df['mutant'] = df[mutant_col].apply(parse_mutations)
    df['num_mutations'] = df['mutant'].apply(len)
    
    return df


def average_numeric_columns_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages numeric columns and preserves categorical/non-numeric values from the first row.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to summarize.

    Returns:
    --------
    pd.DataFrame
        A single-row DataFrame with numeric columns averaged.
    """
    if df.empty:
        return df.copy()

    # Create a 1-row DataFrame base using metadata from the first row
    result = df.iloc[[0]].copy()

    # Calculate means for numeric columns
    numeric_means = df.mean(numeric_only=True)

    # Assign calculated mean values back to numeric columns safely
    for col in numeric_means.index:
        result[col] = numeric_means[col]

    return result


def average_parental_rows(
    df: pd.DataFrame, 
    seq_type_col: str = 'seq_type', 
    parental_val: str = 'parental'
) -> pd.DataFrame:
    """
    Averages numeric values for parental control rows while retaining all non-parental rows.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing control and variant samples.
    seq_type_col : str, default='seq_type'
        Column denoting sequence classification or group.
    parental_val : str, default='parental'
        Identifier value for parental baseline samples.

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with parental rows consolidated into a single averaged row.
    """
    df = df.copy()
    parental_mask = df[seq_type_col] == parental_val
    parental_rows = df.loc[parental_mask]
    non_parental_rows = df.loc[~parental_mask]

    if parental_rows.empty:
        return df

    avg_parental = average_numeric_columns_in_dataframe(parental_rows)
    return pd.concat([avg_parental, non_parental_rows], ignore_index=True)


def coerce_numeric_columns(
    df: pd.DataFrame, 
    cols: list[str] | str,
    strict_strings_to_nan: bool = False
) -> pd.DataFrame:
    """
    Converts specified target columns in a DataFrame to numeric types (float/int),
    converting string entries to NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    cols : str or list of str
        Column name or list of column names to process.
    strict_strings_to_nan : bool, default=False
        - If False (default): Converts text strings like 'N/A' or 'control' to NaN, 
          while parsing stringified numbers like '42.5' into floats/ints.
        - If True: Replaces ALL string instances (even numeric strings like '42.5') 
          strictly with NaN.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the specified columns converted to numeric values.
    """
    if isinstance(cols, str):
        cols = [cols]

    updates = {}
    for col in cols:
        if col not in df.columns:
            continue

        series = df[col]
        if strict_strings_to_nan:
            # Replace any Python str or StringDtype instance with NaN
            is_str = series.apply(lambda val: isinstance(val, str))
            series = series.mask(is_str, np.nan)

        # Convert remaining parseable values to numeric (float/int), coerce failures to NaN
        updates[col] = pd.to_numeric(series, errors='coerce')

    return df.assign(**updates)



def compute_tm_delta(
    df: pd.DataFrame, 
    tm_col: str = 'TM (°C)', 
    seq_type_col: str = 'seq_type', 
    parental_val: str = 'parental',
    output_name: str = 'delta TM',
    baseline_tm: float| None = None
) -> pd.DataFrame:
    """
    Computes delta Tm relative to the parental reference melting temperature.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Tm values and sequence type tags.
    tm_col : str, default='TM (°C)'
        Column containing raw melting temperatures.
    seq_type_col : str, default='seq_type'
        Column identifying sequence type.
    parental_val : str, default='parental'
        Tag value for parental sequence control.
    output_name : str, default='delta TM'
        Name of the newly calculated delta Tm column.
    baseline_tm : float, optional
        Optional explicit baseline scalar Tm. If provided, overrides lookup.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an added column for delta Tm values (Sample Tm - Baseline Tm).

    Raises:
    -------
    ValueError
        If no parental sequence rows are found in the DataFrame.
    """
    df = df.copy()
    
    if baseline_tm is not None:
        parental_tm = float(baseline_tm)
    else:
        parental_mask = df[seq_type_col] == parental_val
        parental_subset = df.loc[parental_mask, tm_col]
        
        if parental_subset.empty:
            raise ValueError(f"No rows found with {seq_type_col} == '{parental_val}' to calculate baseline Tm.")
        
        # Extract scalar value safely whether there's 1 averaged row or multiple parental rows
        parental_tm = float(parental_subset.iloc[0] if len(parental_subset) == 1 else parental_subset.mean())
    
    # Delta Tm = Sample Tm - Baseline Tm (Subtracts baseline scalar float from each row's Tm)
    df[output_name] = df[tm_col] - parental_tm
    return df


def assign_tm_vectorized(
    df: pd.DataFrame, 
    ip1_col: str = 'tm.IP1', 
    ip2_col: str = 'tm.IP2', 
    output_name: str = 'TM (°C)'
) -> pd.Series:
    """
    Vectorized selection of melting temperature: prefers IP2 if present, otherwise falls back to IP1.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing IP1 and IP2 columns.
    ip1_col : str, default='tm.IP1'
        Primary fallback Tm column name.
    ip2_col : str, default='tm.IP2'
        Preferred Tm column name.

    Returns:
    --------
    pd.Series
        Series of selected Tm values.
    """
    df = df.copy()
    df[output_name] = df[ip2_col].fillna(df[ip1_col])
    return df


def assign_tm(row: pd.Series) -> float:
    """
    Row-level helper: returns IP2 temperature if available, otherwise IP1.

    Parameters:
    -----------
    row : pd.Series
        Single row from DataFrame.

    Returns:
    --------
    float
        Selected melting temperature value.
    """
    ip1 = row.get('tm.IP1', np.nan)
    ip2 = row.get('tm.IP2', np.nan)

    # Bug Fix: Replaced deprecated `np.float` type hint with builtin `float`
    if pd.notna(ip2):
        return float(ip2)
    return float(ip1)


def preprocess_pipeline(df, assign_tm:bool = True):

    df = df.copy()

    ### assign the tm_vectorized
    if assign_tm: 
        df = assign_tm_vectorized(df)

    # Step 2: Parse mutation strings into lists & count length
    df = process_mutations(df)

    # Step 3: Consolidate
    df = average_parental_rows(df)

    # Step 4: Compute delta TM relative to parental baseline (25.0°C)
    df = compute_tm_delta(df)
    
    return df


test_df = pd.DataFrame({
    'mutations' : ['', '', 'L:K12L','H:K32L'],
    'seq_type': ['parental', 'parental', 'variant', 'variant'], 
    'tm.IP1': [ 30, 20, 35, 40],
    'tm.IP2': [ np.nan,np.nan , 60, np.nan]
    }
)


if __name__ =='__main__':

    # 1. User Input DataFrame
    test_df = pd.DataFrame(
        {
            'mutations': ['', '', 'L:K12L', 'H:K32L', 'H:T15Y', 'L:K12L,H:K32L' ],
            'seq_type': ['parental', 'parental', 'variant', 'variant', 'variant','variant'],
            'tm.IP1': [30, 20, 35, 40, 55, 55],
            'tm.IP2': [np.nan, np.nan, 60, np.nan, 50, 65,],
        }
    )

    # Step 1: Resolve selected TM (Prefers IP2 = 60, falls back to IP1 = 30, 20, 40)
    df1 = assign_tm_vectorized(test_df)

    # Step 2: Parse mutation strings into lists & count length
    df2 = process_mutations(df1)

    # Step 3: Consolidate duplicate parental rows into 1 averaged row (Parental TM mean = (30 + 20) / 2 = 25.0)
    df3 = average_parental_rows(df2)

    # Step 4: Compute delta TM relative to parental baseline (25.0°C)
    processed_df = compute_tm_delta(df3)
    print(processed_df)









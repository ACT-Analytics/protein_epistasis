import re
from pathlib import Path

import pandas as pd


def ls(
    path: Path,
    infixes: str | list[str] | None = None,
    prefixes: str | list[str] | None = None,
    suffixes: str | list[str] | None = None,
    names_only: bool = False,
) -> list[Path] | list[str]:
    """List out the objects in a path filtered by infixes, prefixes, or suffixes."""

    def normalize(val: str | list[str] | None) -> tuple[str, ...]:
        if not val:
            return ()
        return (val,) if isinstance(val, str) else tuple(val)

    def keep_path(p: Path) -> bool:
        name = p.name
        inf, pre, suf = (
            normalize(infixes),
            normalize(prefixes),
            normalize(suffixes),
        )

        # Check infixes (contains at least one)
        if inf and not any(i in name for i in inf):
            return False
        # Check prefixes (starts with at least one)
        if pre and not name.startswith(pre):
            return False
        # Check suffixes (ends with at least one)
        return not (suf and not name.endswith(suf))

    results = [p for p in path.iterdir() if keep_path(p)]

    if names_only:
        return [p.name for p in results]
    return results


def _frozen_to_string(frozen: frozenset, sep: str = ",") -> str:
    """Converts a frozenset of mutations into a position-sorted string."""

    def _extract_pos(mut_str: str) -> int:
        match = re.search(r"\d+", mut_str)
        return int(match.group()) if match else 0

    sorted_mutants = sorted(frozen, key=_extract_pos)
    return sep.join(sorted_mutants)


def replace_ones_numpy_mul(df: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    """
    FASTEST (Method 1): For numeric/boolean binary columns.
    
    Uses direct NumPy array multiplication.
    Since 1 * float = float and 0 * float = 0, array broadcasting 
    multiplies columns in C-level vectorized speed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with metadata/string columns and binary numeric columns.
    s : pd.Series
        Series whose index matches the binary column names and values are floats.
        
    Returns:
    --------
    pd.DataFrame
        New DataFrame with 1s replaced by series floats.
    """
    cols = s.index
    df_out = df.copy()
    
    # .to_numpy() bypasses Pandas indexing overhead; NumPy automatically broadcasts 
    # a 1D array of shape (num_cols,) across 2D array of shape (num_rows, num_cols).
    df_out[cols] = df_out[cols].to_numpy() * s.to_numpy()
    return df_out


def replace_ones_pandas_mul(df: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    """
    FAST (Method 2): Pure Pandas multiplication via .mul().
    
    Slightly slower than NumPy array broadcasting due to index alignment checks,
    but concise and handles column order automatically.
    """
    cols = s.index
    df_out = df.copy()
    df_out[cols] = df_out[cols].mul(s.T.values, axis=1)
    return df_out
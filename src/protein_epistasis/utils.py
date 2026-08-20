import re
from pathlib import Path

import pandas as pd
from itables import show, to_html_datatable


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


def show_table(df:pd.DataFrame, custom_filename:str='my_file', maxBytes: str ='10MB'):
    """Shows a dataframe with itables

    Notes: 
        1. table-layout: auto allows columns to visually fluidly resize
        2. display: inline-table stops the block container from claiming 100% space
        3. Interactive column dropdown toggles
        4. forces text onto one line so columns shrink as tight as possible
        5. Individual column search filters enabled natively
    """
    if str(custom_filename).endswith('.csv'):
        custom_filename = str(custom_filename)[:-4]

    show(
        df,
        autoWidth=False,
        style="table-layout: auto; display: inline-table; width: auto; margin: 0;", 
        columnDefs=[{"className": "dt-nowrap", "targets": "_all"}],
        maxBytes = maxBytes, 
        # Correct itables argument: options are "header" or "footer"
        column_filters="footer",
        
        # Grid layout allocation
        layout={
            "topStart": {
                "buttons": [
                    {
                        "extend": "csvHtml5",
                        "title": str(custom_filename)
                    },
                    "colvis"
                ]
            },
            "topEnd": "search",       
            "bottomStart": "info",    
            "bottomEnd": "pageLength" 
        }
    )


def save_table_html(df, custom_filename='my_file', output_path='interactive_table.html'):
    """Generates an online standalone HTML file using public CDNs for itables configuration.

    Notes: 
        1. table-layout: auto allows columns to visually fluidly resize
        2. display: inline-table stops the block container from claiming 100% space
        3. Interactive column dropdown toggles
        4. forces text onto one line so columns shrink as tight as possible
        5. Individual column search filters enabled natively
        6. Uses cloud CDNs to fetch table interactivity scripts on load
    """
    # 1. Sanitize the user's custom file export download name
    if str(custom_filename).endswith('.csv'):
        custom_filename = str(custom_filename)[:-4]

    # 2. Generate the table HTML structure with online mode enabled
    # When connected=True, itables automatically attaches script tags pointing to its CDN
    table_body = to_html_datatable(
        df,
        connected=True,  # Pulls JS/CSS dependencies from public CDNs online
        autoWidth=False,
        style="table-layout: auto; display: inline-table; width: auto; margin: 0;", 
        columnDefs=[{"className": "dt-nowrap", "targets": "_all"}],
        column_filters="footer",
        layout={
            "topStart": {
                "buttons": [
                    {
                        "extend": "csvHtml5",
                        "title": str(custom_filename)
                    },
                    "colvis"
                ]
            },
            "topEnd": "search",       
            "bottomStart": "info",    
            "bottomEnd": "pageLength" 
        }
    )

    # 3. Construct the HTML document (The header no longer needs manual offline script injection)
    complete_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{custom_filename} Export</title>
</head>
<body>
    <div style="padding: 20px;">
        {table_body}
    </div>
</body>
</html>"""

    # 4. Handle all file-saving logic internally
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(complete_html)
        
    print(f"Success: Online-dependent table saved to '{output_path}'")

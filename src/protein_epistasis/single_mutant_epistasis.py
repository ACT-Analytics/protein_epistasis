"""Module for computing higher-order epistasis relative to constituent single mutants."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Thermodynamic constant: R in kcal/(mol·K)
R_KCAL: float = 1.987204259e-3
DEFAULT_TEMP_K: float = 298.15  # 25 °C


def construct_mutant_to_single_mutant_table(
    df: pd.DataFrame,
    mutation_col: str = "mutations",
    number_mutations_col: str = "num_mutations",
    analyzer: Callable[[str], list[str]] | None = None,
) -> pd.DataFrame:
    """Constructs a binary indicator matrix (rows = variants, cols = unique single mutants).

    Preserves the original DataFrame index.
    """
    mutations_clean = df[mutation_col].fillna("").astype(str).str.replace(":", "_")

    # Extract unique single mutant identifiers
    mask = df[number_mutations_col] == 1
    single_mutants = pd.unique(mutations_clean.loc[mask])

    if len(single_mutants) == 0:
        raise ValueError(
            f"No single mutants ({number_mutations_col} == 1) found in dataset."
        )

    # Default parser assumes comma-separated mutation strings
    if analyzer is None:
        analyzer = lambda x: [m.strip() for m in x.split(",")] if x else []

    vectorizer = CountVectorizer(
        vocabulary=single_mutants,
        binary=True,
        lowercase=False,
        analyzer=analyzer,
    )

    X_sparse = vectorizer.transform(mutations_clean)

    return pd.DataFrame(
        X_sparse.toarray(),
        index=df.index,
        columns=vectorizer.get_feature_names_out(),
    )


def replace_ones_with_series(df: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Scales binary indicator columns by the corresponding single-mutant score."""
    df_out = df.copy()
    valid_cols = [c for c in df.columns if c in scores.index]
    df_out[valid_cols] = df_out[valid_cols].mul(scores[valid_cols], axis=1)
    return df_out


def compute_epistasis(
    df: pd.DataFrame,
    mutant_score_col: str = "Y",
    metric: str = "delta TM",
    temp_k: float = DEFAULT_TEMP_K,
) -> pd.DataFrame:
    """Computes additive or multiplicative epistatic coupling residuals."""
    sm_cols = [col for col in df.columns if col != mutant_score_col]

    # 1. Thermal Shift: Additive Model
    if metric == "delta TM":
        sum_sm = df[sm_cols].fillna(0.0).sum(axis=1)
        epistasis = df[mutant_score_col] - sum_sm
        return df.assign(
            Sum_Single_Mutant_delta_TM=sum_sm,
            epistasis_delta_TM=epistasis,
            positive_epistasis_delta_TM=epistasis > 0,
        )

    # 2. KD Fold Change (Input = WT / Mutant, i.e., Fold Improvement)
    # DDG = RT * ln(KD_mut / KD_wt) = RT * ln(1 / Y)
    if metric == "KD FC":
        RT = R_KCAL * temp_k
        prod_sm = df[sm_cols].fillna(1.0).replace(0, 1.0).prod(axis=1)
        ddg_mut = RT * np.log(1.0 / df[mutant_score_col])
        ddg_sum = RT * np.log(1.0 / prod_sm)
        ddg_epistasis = ddg_mut - ddg_sum
        fc_epistasis = df[mutant_score_col] / prod_sm

        return df.assign(
            KD_FC_mutant=df[mutant_score_col],
            KD_FC_sm_prod=prod_sm,
            epistasis_KD_FC=fc_epistasis,
            DDG_mutant=ddg_mut,
            DDG_sum_sm=ddg_sum,
            epistasis_DDG=ddg_epistasis,  # < 0 means tighter binding than expected
            synergy_kd=ddg_epistasis < 0,
        )

    # 3. EC50 Fold Improvement (Input = WT / Mutant)
    # Log Fold-Improvement Epistasis = ln(Y_observed) - sum(ln(x_single_mutants))
    if metric == "EC50 FI":
        prod_sm = df[sm_cols].fillna(1.0).replace(0, 1.0).prod(axis=1)
        log_fc_mut = np.log(df[mutant_score_col])
        log_fc_sum = np.log(prod_sm)
        log_epistasis = log_fc_mut - log_fc_sum
        fc_epistasis = df[mutant_score_col] / prod_sm

        return df.assign(
            **{
                "EC50(wt/mut)": df[mutant_score_col],
                "EC50(prod(wt/sm))": prod_sm,
                "epistasis_EC50_FI": fc_epistasis,
                "Log_EC50_FC_mut": log_fc_mut,
                "Log_EC50_FC_prod_sm": log_fc_sum,
                "epistasis_Log_EC50": log_epistasis,  # > 0 means synergistic potency gain
                "positive_epistasis_EC50": fc_epistasis > 1.0,
            }
        )

    raise ValueError(f"Unsupported metric: {metric}")


def construct_epistasis_table(
    df: pd.DataFrame,
    score_col: str = "delta TM",
    mutation_col: str = "mutations",
    number_mutations_col: str = "num_mutations",
    new_score_name: str = "Y",
    remove_single_mutant_cols: bool = False,
    temp_k: float = DEFAULT_TEMP_K,
) -> pd.DataFrame:
    """End-to-end pipeline to compute epistasis while preserving row indices and metadata."""
    clean_series = df[mutation_col].fillna("").astype(str).str.replace(":", "_")

    # 1. Binary indicator matrix
    single_mutant_df = construct_mutant_to_single_mutant_table(
        df=df,
        mutation_col=mutation_col,
        number_mutations_col=number_mutations_col,
    )

    # 2. Extract and average single-mutant scores across replicates if present
    single_mask = df[number_mutations_col] == 1
    single_mutant_scores = (
        pd.Series(
            df.loc[single_mask, score_col].values,
            index=clean_series.loc[single_mask],
        )
        .groupby(level=0)
        .mean()
    )

    # 3. Scale indicators by single mutant scores
    table = replace_ones_with_series(single_mutant_df, single_mutant_scores)

    # 4. Attach observed score using the original row index
    table[new_score_name] = df[score_col]

    # 5. Compute epistasis metrics
    table = compute_epistasis(
        table,
        mutant_score_col=new_score_name,
        metric=score_col,
        temp_k=temp_k,
    )

    if remove_single_mutant_cols:
        table = table.drop(columns=single_mutant_scores.index)

    return table
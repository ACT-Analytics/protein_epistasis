# For computing epistasis relative to single mutants


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Thermodynamic constant: R in kcal/(mol·K)
R_KCAL = 1.987204259e-3
DEFAULT_TEMP_K = 298.15  # 25 °C


def replace_ones_with_series(df: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Replaces binary 1s in a DataFrame with matching float values from a Series."""
    cols = scores.index
    df_out = df.copy()
    df_out[cols] = df_out[cols].mul(scores.T.values, axis=1)
    return df_out


def construct_mutant_to_single_mutant_table(
    df: pd.DataFrame,
    mutation_col: str = "mutations",
    number_mutations_col: str = "num_mutations",
    analyzer: callable | None = None,
) -> pd.DataFrame:
    # Standardize colon replacement on the series
    mutations = df[mutation_col].fillna("").astype(str).str.replace(":", "_")

    # Extract UNIQUE single mutant identifiers for vocabulary
    mask = df[number_mutations_col] == 1
    single_mutants = pd.unique(mutations.loc[mask])

    if len(single_mutants) == 0:
        raise ValueError("No single mutants (num_mutations == 1) found in dataset.")

    # Default analyzer assumes comma-separated mutation strings
    if analyzer is None:
        analyzer = lambda x: [m.strip() for m in x.split(",")] if x else []

    vectorizer = CountVectorizer(
        vocabulary=single_mutants,
        binary=True,
        lowercase=False,  # Preserves case (e.g., A10V vs a10v)
        analyzer=analyzer,
    )

    X_sparse = vectorizer.transform(mutations)

    return pd.DataFrame(
        X_sparse.toarray(),
        index=mutations,
        columns=vectorizer.get_feature_names_out(),
    )


def compute_inverse_prod(
    df: pd.DataFrame, cols: list[str], output_name: str = "InverseProd"
) -> pd.DataFrame:
    cleaned_cols = df[cols].replace(0, 1)
    inv_prod = 1.0 / cleaned_cols.prod(axis=1)
    return df.assign(**{output_name: inv_prod})


def compute_epistasis(
    df: pd.DataFrame,
    mutant_score_col: str = "Y",
    metric: str = "delta TM",
    temp_k: float = 298.15,
) -> pd.DataFrame:
    sm_cols = [col for col in df.columns if col != mutant_score_col]

    # 1. Thermal Shift: Additive Model
    if metric == "delta TM":
        sum_sm = df[sm_cols].fillna(0.0).sum(axis=1)
        epistasis = df[mutant_score_col] - sum_sm
        return df.assign(
            Sum_Single_Mutant_delta_TM=sum_sm,
            epistasis_delta_TM=epistasis,
            positive_epistasis_delta_TM = epistasis >0 
        )

    # 2. KD Fold Change (Input = WT / Mutant)
    # DDG = RT * ln(KD_mut / KD_wt) = RT * ln(1 / Y)
    if metric == "KD FC":
        RT = R_KCAL * temp_k
        prod_sm = df[sm_cols].fillna(1.0).replace(0, 1.0).prod(axis=1)
        
        # DDG = RT * ln(KD_mut / KD_wt) = RT * ln(1 / FC)
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
            epistasis_DDG=ddg_epistasis,  # < 0 indicates thermodynamic synergy (tighter binding)
            synergy_kd=ddg_epistasis < 0,
        )

    # 3. EC50 Fold Improvement (Input = WT / Mutant)
    # Log Improvement Epistasis = ln(Y_observed) - sum(ln(x_single_mutants))
    if metric == "EC50 FI":
        prod_sm = df[sm_cols].fillna(1.0).replace(0, 1.0).prod(axis=1)
        
        # Direct log fold-improvement excess: > 0 means synergistic potency gain
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
    remove_single_mutant_cols: bool = False
) -> pd.DataFrame:
    # Standardize mutation string formatting
    clean_df = df.assign(
        clean_mutations=df[mutation_col].str.replace(":", "_")
    )

    # Construct binary matrix (rows = mutants, cols = single mutants)
    single_mutant_df = construct_mutant_to_single_mutant_table(
        clean_df, "clean_mutations", number_mutations_col
    )

    # Extract single-mutant scores as a Series
    single_mutant_scores = (
        clean_df.loc[clean_df[number_mutations_col] == 1]
        .set_index("clean_mutations")[score_col]
    )

    # Replace 1s with corresponding single mutant score values
    table = replace_ones_with_series(single_mutant_df, single_mutant_scores)

    # Merge observed mutant score column
    table = table.merge(
        clean_df.set_index("clean_mutations")[score_col].rename(new_score_name),
        left_index=True,
        right_index=True,
    )

    # Compute epistasis non-mutatively
    table = compute_epistasis(table, mutant_score_col=new_score_name, metric=score_col)
    if remove_single_mutant_cols:
        table = table.drop(columns=single_mutant_scores.index)
    
    return table


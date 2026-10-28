"""Epistatic Interaction & Regularized Free Energy Modeling Pipeline.

Decomposes combinatorial deep mutational scanning (DMS) variants into
additive first-order effects and regularized pairwise epistatic couplings
using sparse design matrices and ElasticNet regression.

__author__ = "Christopher Rivera"
__credits__ = ["Christopher Rivera", "Gemini"]
"""

import numpy as np
import pandas as pd
import patsy
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

from protein_epistasis.single_mutant_epistasis import (
    construct_mutant_to_single_mutant_table,
)


def construct_categorical_design_matrix(
    df: pd.DataFrame,
    mutation_col: str = "mutations",
    target_cols: list[str] | str | None = None,
    merge_dfs: bool = False,
    min_mutations: int = 1,
):
    """Constructs an initial design matrix and target vectors for regression."""
    if target_cols is None:
        target_cols = ["delta TM", "hasIP2", "KD FC"]
    elif isinstance(target_cols, str):
        target_cols = [target_cols]

    # Filter dataframe first so X and y always share the identical index/shape
    filtered_df = df.query("num_mutations >= @min_mutations")

    X = construct_mutant_to_single_mutant_table(filtered_df, mutation_col)
    y = filtered_df.set_index(mutation_col)[target_cols]

    if merge_dfs:
        return X.merge(y, left_index=True, right_index=True)
    return X, y


def construct_design_matrix_with_formula(df, target_col='delta TM', order:int=2, remove_spaces=True, min_mutations=1):
    """ uses patsy to create a dmatrix with desired interaction"""

    data = construct_categorical_design_matrix(df, target_cols=target_col, merge_dfs = True, min_mutations=min_mutations)

    if remove_spaces: 
        data = data.rename(columns = {col:col.replace(' ', '_') for col in data.columns})
        target_col = target_col.replace(' ', '_')
        
    mutation_cols = data.columns[:-1]
    
    formula = f'{target_col}~({' + '.join(mutation_cols)})**{order}'
    y, X = patsy.dmatrices(formula, data=data, return_type='matrix')

    feature_names = X.design_info.column_names

    return pd.DataFrame(X, columns=feature_names, index = data.index).iloc[:,1:], y


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Calculates Adjusted R² with degrees of freedom guard."""
    if n - p - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def score_regression(
    y_true, y_pred, n: int | None = None, p: int | None = None
) -> pd.DataFrame:
    """Computes standard regression evaluation metrics."""
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    scores = {
        "rmse": root_mean_squared_error(y_true_arr, y_pred_arr),
        "r2": r2_score(y_true_arr, y_pred_arr),
        "pearson_corr": pearsonr(y_true_arr, y_pred_arr)[0],
        "spearman_corr": spearmanr(y_true_arr, y_pred_arr)[0],
    }

    if n is not None and p is not None:
        scores["adjusted_r2"] = adjusted_r2(scores["r2"], n, p)

    return pd.DataFrame(scores, index=["scores"]).T


# Standalone metric functions for scikit-learn scorers
def _pearson_metric(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def _spearman_metric(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def cross_validate_regression(estimator, X, y, cv: int = 10, as_dataframe: bool = True):
    """Evaluates an estimator using cross-validation across standard metrics."""
    scoring = {
        "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
        "pearson_r": make_scorer(_pearson_metric),
        "spearman_r": make_scorer(_spearman_metric),
    }

    cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring)

    if as_dataframe:
        return pd.DataFrame(cv_results).agg(["mean", "std"]).T
    return cv_results


def fit_epistatic_elastic_net(
    df: pd.DataFrame,
    target_col: str = "delta TM",
    mutation_col: str = "mutations",
    min_mutations: int = 2,
    order: int = 2,
    cv: int = 10,
):
    """
    Builds a sparse interaction design matrix and trains an ElasticNetCV model
    to extract first-order and higher-order epistatic coefficients.
    """
    filtered_df = df.query("num_mutations >= @min_mutations")

    # 1. Generate base 0/1 indicator table for single mutations
    X_single = construct_mutant_to_single_mutant_table(filtered_df, mutation_col)
    y = filtered_df[target_col].to_numpy()

    # 2. Memory-efficient interaction expansion (avoids Patsy string parsing)
    poly = PolynomialFeatures(degree=order, interaction_only=True, include_bias=False)
    # Convert to sparse matrix to prevent RAM blowups
    X_sparse = poly.fit_transform(csr_matrix(X_single.values))
    feature_names = poly.get_feature_names_out(X_single.columns)

    # 3. Cross-validated ElasticNet over L1 ratios and regularization strengths
    l1_ratios = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]  # Tests Ridge-heavy to pure Lasso
    model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1, random_state=42)
    model.fit(X_sparse, y)

    # 4. Extract nonzero epistatic coefficients
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        .query("coefficient != 0")
        .sort_values("coefficient", key=abs, ascending=False)
    )

    return model, coef_df


def extract_epistatic_attributions(
    model: ElasticNetCV,
    feature_names: list[str],
    mutation_cols: list[str],
    eps: float = 1e-5,
) -> pd.DataFrame:
    """Extracts and parses all non-zero coefficients from the fitted ElasticNet model."""
    coefs = model.coef_

    # Retain non-zero coefficients above numerical noise floor
    active_mask = np.abs(coefs) > eps

    records = []
    for feat, beta in zip(feature_names[active_mask], coefs[active_mask]):
        parts = feat.split(" ")
        if len(parts) == 1:
            records.append(
                {
                    "feature": feat,
                    "order": 1,
                    "mut_1": parts[0],
                    "mut_2": np.nan,
                    "beta": beta,
                    "abs_effect": abs(beta),
                }
            )
        elif len(parts) == 2:
            records.append(
                {
                    "feature": feat,
                    "order": 2,
                    "mut_1": parts[0],
                    "mut_2": parts[1],
                    "beta": beta,
                    "abs_effect": abs(beta),
                }
            )

    df_attr = pd.DataFrame(records)
    return df_attr.sort_values(by="abs_effect", ascending=False).reset_index(drop=True)


def attribute_variant(
    variant_row_sparse: csr_matrix,
    model: ElasticNetCV,
    poly: PolynomialFeatures,
    mutation_cols: list[str],
) -> pd.DataFrame:
    """Decomposes a specific multi-mutant variant into its baseline,

    additive, and pairwise epistatic contribution lines.
    """
    feature_names = poly.get_feature_names_out(mutation_cols)
    coefs = model.coef_

    # Element-wise product of active indicators and coefficients
    active_indices = variant_row_sparse.indices
    active_coefs = coefs[active_indices]
    active_features = feature_names[active_indices]

    breakdown = [
        {
            "term": "Baseline (Intercept / WT)",
            "order": 0,
            "contribution": model.intercept_,
        }
    ]

    for feat, val in zip(active_features, active_coefs):
        if abs(val) > 1e-6:
            order = 2 if " " in feat else 1
            breakdown.append(
                {
                    "term": feat,
                    "order": order,
                    "contribution": val,
                }
            )

    df_var = pd.DataFrame(breakdown)
    df_var["predicted_total"] = df_var["contribution"].sum()
    return df_var


def fit_and_cross_validate_elasticnet_pipeline(
    df: pd.DataFrame,
    target_col: str,
    is_multiplicative: bool = False,
    min_mutations:int =1,
    order:int =2,
):


    X, y = construct_design_matrix_with_formula(
        df, min_mutations=min_mutations, order=order, target_col=target_col
    )

    if is_multiplicative:
        y = np.log(y).ravel()

    else: 
        y = y.ravel()

    feature_names = X.columns

    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        cv=10,
        random_state=42,
        n_jobs=-1,
    )
    enet_cv.fit(X, y)

    cv_metrics_df = cross_validate_regression(enet_cv, X, y, cv=10, as_dataframe=True)

    coefficients = pd.Series(enet_cv.coef_, index=feature_names)
    
    selected_epistasis = coefficients[coefficients != 0].drop('Intercept', errors='ignore')

    return enet_cv, cv_metrics_df, X, y, selected_epistasis



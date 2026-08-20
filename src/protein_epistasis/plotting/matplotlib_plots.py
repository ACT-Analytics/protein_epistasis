# Authored By Christopher Rivera and Gemini.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

from .config import PLOT_PARAMETERS


def scatter_plot_with_identity(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: Axes,
    hue: str = "Number of Mutations",
    style: str | None = None,  # Column for distinct marker shapes
    marker: str | dict | None = None,  # Fixed shape (e.g. 's', '^', 'D') or dict map
    palette: str = "colorblind",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    markersize: int = 50,
    limits: tuple[float, float] | None = None,
    center: float | None = None,
    span: float | None = None,
    padding: float = 0.05,
    ref_lines: tuple[float, ...] | float | None = (0,),
    use_log: bool = False,
    log_base: float = 10,
) -> Axes:
    data = df.copy()

    # Log transformation
    if use_log:
        log_fn = np.log10 if log_base == 10 else (np.log2 if log_base == 2 else np.log)
        x_plot = f"_log_{x_col}"
        y_plot = f"_log_{y_col}"
        data[x_plot] = log_fn(data[x_col])
        data[y_plot] = log_fn(data[y_col])

        if ref_lines is not None:
            if isinstance(ref_lines, (int, float)):
                ref_lines = (ref_lines,)
            ref_lines = tuple(log_fn(val) for val in ref_lines if val > 0)
    else:
        x_plot, y_plot = x_col, y_col

    # Prepare marker / style arguments for Seaborn
    scatter_kwargs = {}
    if style is not None:
        scatter_kwargs["style"] = style
        if isinstance(marker, (dict, list)):
            scatter_kwargs["markers"] = marker
    elif marker is not None and isinstance(marker, str):
        scatter_kwargs["marker"] = marker

    # Scatter plot
    ax = sns.scatterplot(
        data=data,
        x=x_plot,
        y=y_plot,
        hue=hue,
        palette=palette,
        alpha=0.75,
        legend=True,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
        s=markersize,
        **scatter_kwargs,
    )

    # Identity line (y = x)
    ax.axline((0, 0), slope=1, color="#4A5568", linestyle="--", alpha=0.3)

    # Reference lines
    if ref_lines is not None:
        if isinstance(ref_lines, (int, float)):
            ref_lines = (ref_lines,)
        for val in ref_lines:
            ax.axhline(val, color="#A0AEC0", linestyle=":", alpha=0.75, linewidth=1)
            ax.axvline(val, color="#A0AEC0", linestyle=":", alpha=0.75, linewidth=1)

    if not xlabel:
        xlabel = x_col.replace("_", " ")
    if not ylabel:
        ylabel = y_col.replace("_", " ")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    # Determine identical limits
    if limits is not None:
        low, high = limits
    elif center is not None:
        if span is None:
            max_dist = max(
                abs(data[x_plot].min() - center),
                abs(data[x_plot].max() - center),
                abs(data[y_plot].min() - center),
                abs(data[y_plot].max() - center),
                abs(0 - center),
            )
            half_span = max_dist * (1 + padding)
        else:
            half_span = span / 2.0
        low, high = center - half_span, center + half_span
    else:
        min_val = min(data[x_plot].min(), data[y_plot].min(), 0)
        max_val = max(data[x_plot].max(), data[y_plot].max(), 0)

        if span is not None:
            c = (min_val + max_val) / 2.0
            low, high = c - span / 2.0, c + span / 2.0
        else:
            range_val = max_val - min_val
            pad_val = (range_val * padding) if range_val != 0 else 1.0
            low, high = min_val - pad_val, max_val + pad_val

    if limits is None:
        low = min(low, 0)
        high = max(high, 0)

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_epi_scatterplots(
    df: pd.DataFrame,
    dataset: str = "",
    measurements: list[str] | str | None = None,
    width: float = 6,
    height: float = 6,
    markersize: float = 100,
    title_fontsize: int = 20,
    style: str | None = None,
):
    if not measurements:
        measurements = ["delta TM", "Binding Multiplicative", "EC50 Multiplicative"]

    if isinstance(measurements, str):
        measurements = [measurements]

    n_cols = len(measurements)
    fig, axes = plt.subplots(1, n_cols, figsize=(width * n_cols, height))

    if n_cols == 1:
        axes = [axes]

    for ax, measurement in zip(axes, measurements):
        if style is not None and isinstance(style, dict):
            current_style = style.get(measurement, None)
        elif style is not None and isinstance(style, str):
            current_style = style
        else:
            current_style = None

        scatter_plot_with_identity(
            df,
            ax=ax,
            markersize=markersize,
            style=current_style,
            **PLOT_PARAMETERS[measurement],
        )

    title_text = (
        f"Scatter plots for epistasis for {dataset}"
        if dataset
        else "Scatter plots for epistasis"
    )

    # 1. Increase font size and raise the suptitle vertical position (y=1.02)
    fig.suptitle(title_text, fontsize=title_fontsize, y=1.02, fontweight="bold")

    sns.despine()

    # 2. Tighten layout first, then adjust top spacing to prevent collisions
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    return fig


def plot_elasticnet_tm_coefficients(
    fitted_model, X, y, feature_names=None, ax=None, font_size=14, markersize=10, cv=10
):
    """
    Extracts non-zero coefficients from ElasticNetCV, calculates CV error bars,
    and plots a sorted horizontal coefficient plot for Tm changes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, len(feature_names) * 0.4 + 2))

    # --- 1. Extract and Filter Coefficients ---
    if not feature_names:
        feature_names = fitted_model.feature_names_in_

    df = pd.DataFrame({"Feature": feature_names, "Coef": fitted_model.coef_})
    # Keep only the mutations that feature selection didn't drop
    df = df[df["Coef"] != 0].copy()

    if df.empty:
        print("No non-zero coefficients found by the Elastic Net model.")
        return ax

    # --- 2. Calculate Coefficient Stability (Error Bars) ---
    best_alpha = fitted_model.alpha_
    best_l1 = fitted_model.l1_ratio_

    # Use the same number of cv folds as your ElasticNetCV (default is 5)
    cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_coefs = []

    # Convert X and y to numpy arrays for reliable indexing
    X_arr = np.array(X)
    y_arr = np.array(y)

    for train_idx, _ in cv.split(X_arr):
        fold_model = ElasticNet(
            alpha=best_alpha, l1_ratio=best_l1, max_iter=fitted_model.max_iter
        )
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])
        cv_coefs.append(fold_model.coef_)

    # Map the standard deviation of the fold coefficients back to the active features
    coef_std = np.std(cv_coefs, axis=0)
    df["Error"] = coef_std[df.index]

    # Sort from largest negative effect to largest positive effect (reads bottom-to-top)
    df = df.sort_values(by="Coef", ascending=True).reset_index(drop=True)

    # --- 3. Plotting ---
    # Assign intuitive colors: crimson for destabilizing (<0), teal for stabilizing (>0)
    colors = ["crimson" if c < 0 else "teal" for c in df["Coef"]]

    # Draw the horizontal error bars and markers
    ax.errorbar(
        df["Coef"],
        df["Feature"],
        xerr=df["Error"],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=3,
        elinewidth=1.5,
        ms=markersize/2,
    )
    # Overlay the colored scatter points on top of the black errorbar dots
    ax.scatter(df["Coef"], df["Feature"], color=colors, s=markersize, zorder=3)

    # Draw a clean vertical baseline reference at zero
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1.2)

    # --- 4. Custom Styling (Clean Dictionary Approach) ---
    ax.set(
        xlabel=r"$\Delta T_m (^\circ C)$",
        ylabel="",
        title="Elastic Net Selected Mutation Effects",
    )

    ax.tick_params(axis="both", labelsize=font_size - 2)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Tighten margins around the elements cleanly
    plt.tight_layout()

    return ax

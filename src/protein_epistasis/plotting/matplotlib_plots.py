import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

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
        scatter_plot_with_identity(
            df, ax=ax, markersize=markersize, **PLOT_PARAMETERS[measurement]
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

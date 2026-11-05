# Authored By Christopher Rivera and Gemini.

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

from .config import PLOT_PARAMETERS


def add_corner_text(
    ax: Axes,
    text: str,
    corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_left",
    font_size: float = 9.0,
    fontweight: str = "normal",
    color: str = "#2D3748",
    linespacing: float = 1.35,
    box_facecolor: str = "none",
    box_edgecolor: str = "none",
) -> Axes:
    """Adds a multiline text block to a specified corner of an axes."""
    corner_configs = {
        "top_left": {"x": 0.04, "y": 0.96, "ha": "left", "va": "top"},
        "top_right": {"x": 0.96, "y": 0.96, "ha": "right", "va": "top"},
        "bottom_left": {"x": 0.04, "y": 0.04, "ha": "left", "va": "bottom"},
        "bottom_right": {"x": 0.96, "y": 0.04, "ha": "right", "va": "bottom"},
    }
    cfg = corner_configs.get(corner, corner_configs["top_left"])

    if box_facecolor == "none" and box_edgecolor == "none":
        bbox = None
    else:
        bbox = {
            "boxstyle": "round,pad=0.45",
            "facecolor": box_facecolor,
            "edgecolor": box_edgecolor,
            "linewidth": 0.75 if box_edgecolor != "none" else 0.0,
        }

    ax.text(
        cfg["x"],
        cfg["y"],
        text,
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight=fontweight,
        color=color,
        horizontalalignment=cfg["ha"],
        verticalalignment=cfg["va"],
        multialignment="left",
        linespacing=linespacing,
        bbox=bbox,
        zorder=5,
    )
    return ax


def scatter_plot_with_identity(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: Axes,
    hue: str | None = "Number of Mutations",
    style: str | None = None,
    marker: str | dict | list | None = None,
    palette: str = "colorblind",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    markersize: int = 50,
    # Axis limit controls
    limits: tuple[float, float] | list[float] | np.ndarray | tuple | list | None = None,
    xlim: tuple[float, float] | list[float] | np.ndarray | None = None,
    ylim: tuple[float, float] | list[float] | np.ndarray | None = None,
    equal_limits: bool = True,
    equal_aspect: bool = True,
    center: float | None = None,
    span: float | None = None,
    padding: float = 0.05,
    ref_lines: tuple[float, ...] | list[float] | np.ndarray | float | None = (0.0,),
    use_log: bool = False,
    log_base: float = 10.0,
    auto_log_labels: bool = True,
    show_legend: bool = True,
    legend_outside: bool = True,
    legend_title: str | None = None,
    include_zero: bool = True,
    # Font size controls
    fontsize: float = 11.0,
    title_fontsize: float | None = None,
    label_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    legend_fontsize: float | None = None,
    corner_fontsize: float | None = None,
    # Metadata corner text arguments
    project: str | None = None,
    data_source: str | None = None,
    x_desc: str | None = None,
    y_desc: str | None = None,
    extra_entries: list[str] | None = None,
    corner_text: str | list[str] | None = None,
    corner_pos: Literal["top_left", "top_right"] | None = "top_left",
    corner_facecolor: str = "none",
    corner_edgecolor: str = "none",
) -> Axes:
    # 1. Resolve hierarchical font sizes
    res_title_size = title_fontsize if title_fontsize is not None else fontsize + 2
    res_label_size = label_fontsize if label_fontsize is not None else fontsize
    res_tick_size = tick_fontsize if tick_fontsize is not None else max(fontsize - 2, 7)
    res_legend_size = legend_fontsize if legend_fontsize is not None else max(fontsize - 2, 7)
    res_corner_size = corner_fontsize if corner_fontsize is not None else max(fontsize - 2, 7)

    # 2. Normalize reference lines safely
    norm_ref_lines: tuple[float, ...] = ()
    if ref_lines is not None:
        if isinstance(ref_lines, (int, float, np.number)):
            norm_ref_lines = (float(ref_lines),)
        else:
            norm_ref_lines = tuple(float(v) for v in np.asarray(ref_lines).ravel())

    # 3. Handle log transformation and column mapping
    if use_log:
        log_fn = np.log10 if log_base == 10 else (np.log2 if log_base == 2 else np.log)
        x_plot = f"_log_{x_col}"
        y_plot = f"_log_{y_col}"

        data = df[[c for c in [x_col, y_col, hue, style] if c is not None and c in df.columns]].copy()
        data[x_plot] = log_fn(data[x_col])
        data[y_plot] = log_fn(data[y_col])
        norm_ref_lines = tuple(float(log_fn(val)) for val in norm_ref_lines if val > 0)
        include_zero = False
    else:
        x_plot, y_plot = x_col, y_col
        data = df

    # 4. Configure marker and style mappings
    scatter_kwargs = {}
    if style is not None:
        scatter_kwargs["style"] = style
        if isinstance(marker, (dict, list)):
            scatter_kwargs["markers"] = marker
    elif marker is not None and isinstance(marker, str):
        scatter_kwargs["marker"] = marker

    # 5. Draw scatter points
    has_hue = hue is not None and (hue in data.columns if isinstance(hue, str) else True)
    has_style = style is not None and (style in data.columns if isinstance(style, str) else True)
    draw_legend = show_legend and (has_hue or has_style)

    sns.scatterplot(
        data=data,
        x=x_plot,
        y=y_plot,
        hue=hue if (isinstance(hue, str) and hue in data.columns) else None,
        palette=palette,
        alpha=0.8,
        legend=draw_legend,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
        s=markersize,
        **scatter_kwargs,
    )

    # 6. Identity line (y = x)
    ax.axline((0, 0), slope=1, color="#4A5568", linestyle="--", alpha=0.4, zorder=1)

    # 7. Reference lines
    for val in norm_ref_lines:
        ax.axhline(val, color="#A0AEC0", linestyle=":", alpha=0.75, linewidth=1, zorder=1)
        ax.axvline(val, color="#A0AEC0", linestyle=":", alpha=0.75, linewidth=1, zorder=1)

    # 8. Labels, title, and tick font sizes
    raw_xlabel = xlabel if xlabel is not None else x_col.replace("_", " ")
    raw_ylabel = ylabel if ylabel is not None else y_col.replace("_", " ")

    if use_log and auto_log_labels:
        if log_base == 10:
            log_sym = r"\log_{10}"
        elif log_base == 2:
            log_sym = r"\log_{2}"
        elif np.isclose(log_base, np.e):
            log_sym = r"\ln"
        else:
            log_sym = f"\\log_{{{log_base}}}"

        def _format_log_label(lbl: str) -> str:
            clean = lbl.strip()
            if clean.lower().startswith(("log", r"$\log", r"\log", "ln", r"$\ln", r"\ln")):
                return clean
            return f"${log_sym}$ ({clean})"

        resolved_xlabel = _format_log_label(raw_xlabel)
        resolved_ylabel = _format_log_label(raw_ylabel)
    else:
        resolved_xlabel = raw_xlabel
        resolved_ylabel = raw_ylabel

    ax.set_xlabel(resolved_xlabel, fontsize=res_label_size)
    ax.set_ylabel(resolved_ylabel, fontsize=res_label_size)
    if title is not None:
        ax.set_title(title, fontsize=res_title_size)
    ax.tick_params(axis="both", labelsize=res_tick_size)

    # 9. Enforce equal limits unless explicitly specified otherwise
    spec_x: tuple[float, float] | None = None
    spec_y: tuple[float, float] | None = None

    if xlim is not None:
        arr_x = np.asarray(xlim, dtype=float).ravel()
        spec_x = (float(arr_x[0]), float(arr_x[1]))

    if ylim is not None:
        arr_y = np.asarray(ylim, dtype=float).ravel()
        spec_y = (float(arr_y[0]), float(arr_y[1]))

    if limits is not None:
        arr_lim = np.asarray(limits, dtype=float)
        if arr_lim.ndim == 2 and arr_lim.shape == (2, 2):
            if spec_x is None:
                spec_x = (float(arr_lim[0, 0]), float(arr_lim[0, 1]))
            if spec_y is None:
                spec_y = (float(arr_lim[1, 0]), float(arr_lim[1, 1]))
        elif arr_lim.size == 2:
            flat = arr_lim.ravel()
            shared = (float(flat[0]), float(flat[1]))
            if spec_x is None:
                spec_x = shared
            if spec_y is None:
                spec_y = shared
        elif arr_lim.size == 4:
            flat = arr_lim.ravel()
            if spec_x is None:
                spec_x = (float(flat[0]), float(flat[1]))
            if spec_y is None:
                spec_y = (float(flat[2]), float(flat[3]))

    def _compute_bounds(series_list: list[pd.Series]) -> tuple[float, float]:
        min_cands = [float(s.min()) for s in series_list if not s.empty and not s.isna().all()]
        max_cands = [float(s.max()) for s in series_list if not s.empty and not s.isna().all()]
        if not min_cands or not max_cands:
            return (0.0, 1.0)
        if include_zero:
            min_cands.append(0.0)
            max_cands.append(0.0)

        min_val = min(min_cands)
        max_val = max(max_cands)

        if center is not None:
            if span is None:
                max_dist = max(
                    abs(min_val - center),
                    abs(max_val - center),
                    abs(0.0 - center) if include_zero else 0.0,
                )
                half_span = max_dist * (1 + padding)
            else:
                half_span = span / 2.0
            low, high = center - half_span, center + half_span
        elif span is not None:
            c = (min_val + max_val) / 2.0
            low, high = c - span / 2.0, c + span / 2.0
        else:
            range_val = max_val - min_val
            pad_val = (range_val * padding) if range_val != 0 else 1.0
            low, high = min_val - pad_val, max_val + pad_val

        if include_zero:
            low = min(low, 0.0)
            high = max(high, 0.0)

        return float(low), float(high)

    # Resolution cascade
    if spec_x is not None and spec_y is not None:
        # Both explicitly given -> respect distinct values
        resolved_xlim, resolved_ylim = spec_x, spec_y
    elif spec_x is not None and spec_y is None:
        # Only X given -> Y inherits X to maintain equal limits
        resolved_xlim, resolved_ylim = spec_x, spec_x
    elif spec_y is not None and spec_x is None:
        # Only Y given -> X inherits Y to maintain equal limits
        resolved_xlim, resolved_ylim = spec_y, spec_y
    else:
        # Neither given -> use pooled data bounds if equal_limits is True
        if equal_limits:
            shared_bounds = _compute_bounds([data[x_plot], data[y_plot]])
            resolved_xlim, resolved_ylim = shared_bounds, shared_bounds
        else:
            resolved_xlim = _compute_bounds([data[x_plot]])
            resolved_ylim = _compute_bounds([data[y_plot]])

    ax.set_xlim(resolved_xlim[0], resolved_xlim[1])
    ax.set_ylim(resolved_ylim[0], resolved_ylim[1])

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")

    # 10. Multi-line corner metadata text (transparent by default)
    if corner_pos is not None:
        entries: list[str] = []

        if corner_text is not None:
            if isinstance(corner_text, str):
                entries.extend(corner_text.strip().splitlines())
            else:
                entries.extend(corner_text)
        else:
            if project is not None:
                entries.append(f"Project: {project}")
            if data_source is not None:
                entries.append(f"Data Source: {data_source}")
            if x_desc is not None:
                entries.append(f"X: {x_desc}")
            if y_desc is not None:
                entries.append(f"Y: {y_desc}")
            if extra_entries is not None:
                entries.extend(extra_entries)

        if entries:
            formatted_text = "\n".join(entries)
            add_corner_text(
                ax,
                formatted_text,
                corner=corner_pos,
                font_size=res_corner_size,
                box_facecolor=corner_facecolor,
                box_edgecolor=corner_edgecolor,
            )

    # 11. Legend positioning, styling, and font sizing
    current_legend = ax.get_legend()
    if not show_legend:
        if current_legend is not None:
            current_legend.remove()
    elif current_legend is not None:
        resolved_title = legend_title or current_legend.get_title().get_text()
        move_kwargs = {
            "title": resolved_title if resolved_title else None,
            "fontsize": res_legend_size,
            "title_fontsize": res_legend_size + 1,
            "frameon": True,
        }
        if legend_outside:
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0), **move_kwargs)
        else:
            sns.move_legend(ax, loc="best", **move_kwargs)

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
    legend_outside=True,
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
            legend_outside=legend_outside,
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
        ms=markersize / 2,
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


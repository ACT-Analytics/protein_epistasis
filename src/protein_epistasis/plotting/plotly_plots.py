import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PLOT_PARAMETERS


def add_scatter_with_identity_to_plotly(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    row: int = 1,
    col: int = 1,
    hue: str = "Number of Mutations",
    color_map: dict | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    markersize: int = 8,
    limits: tuple[float, float] | None = None,
    center: float | None = None,
    span: float | None = None,
    padding: float = 0.05,
    ref_lines: tuple[float, ...] | float | None = (0,),
    use_log: bool = False,
    log_base: float = 10,
    seen_legend_groups: set | None = None,
) -> go.Figure:
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

    # Compute limits including (0,0)
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

    # Plot scatter traces
    if hue in data.columns:
        categories = sorted(data[hue].unique())
        for cat in categories:
            sub_df = data[data[hue] == cat]
            show_leg = False
            if seen_legend_groups is not None and cat not in seen_legend_groups:
                show_leg = True
                seen_legend_groups.add(cat)

            color = color_map.get(cat) if color_map else None

            fig.add_trace(
                go.Scatter(
                    x=sub_df[x_plot],
                    y=sub_df[y_plot],
                    mode="markers",
                    name=str(cat),
                    legendgroup=str(cat),
                    showlegend=show_leg,
                    marker={
                        "size": markersize,
                        "color": color,
                        "line": {"width": 0.5, "color": "black"},
                        "opacity": 0.75,
                    },
                    hovertemplate=(
                        f"<b>{hue}: {cat}</b><br>"
                        f"{'log10(' if use_log else ''}{x_col}{')' if use_log else ''}: %{{x:.3f}}<br>"
                        f"{'log10(' if use_log else ''}{y_col}{')' if use_log else ''}: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=data[x_plot],
                y=data[y_plot],
                mode="markers",
                showlegend=False,
                marker={
                    "size": markersize,
                    "line": {"width": 0.5, "color": "black"},
                    "opacity": 0.75,
                },
                hovertemplate=f"{x_plot}: %{{x:.3f}}<br>{y_plot}: %{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    # Identity Line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": "#4A5568", "dash": "dash", "width": 1.5},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    # Reference Lines
    if ref_lines is not None:
        if isinstance(ref_lines, (int, float)):
            ref_lines = (ref_lines,)
        for val in ref_lines:
            fig.add_hline(
                y=val,
                line={"color": "#A0AEC0", "dash": "dot", "width": 1},
                row=row,
                col=col,
            )
            fig.add_vline(
                x=val,
                line={"color": "#A0AEC0", "dash": "dot", "width": 1},
                row=row,
                col=col,
            )

    # Layout updates
    if not xlabel:
        xlabel = x_col.replace("_", " ")
    if not ylabel:
        ylabel = y_col.replace("_", " ")

    x_axis_name = f"x{col}" if col > 1 else "x"
    fig.update_xaxes(title_text=xlabel, range=[low, high], row=row, col=col)
    fig.update_yaxes(
        title_text=ylabel,
        range=[low, high],
        scaleanchor=x_axis_name,
        scaleratio=1,
        row=row,
        col=col,
    )

    return fig


def plot_epi_scatterplots_plotly(
    df: pd.DataFrame,
    dataset: str = "",
    measurements: list[str] | str | None = None,
    hue: str = "Number of Mutations",
    width_per_plot: int = 600,
    height: int = 600,
    markersize: int = 8,
    palette: list[str] | None = None,
) -> go.Figure:
    if not measurements:
        measurements = ["delta TM", "Binding Multiplicative", "EC50 Multiplicative"]

    if isinstance(measurements, str):
        measurements = [measurements]

    n_cols = len(measurements)
    subplot_titles = [PLOT_PARAMETERS[m]["title"] for m in measurements]

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    color_map = {}
    if hue in df.columns:
        unique_hues = sorted(df[hue].unique())
        colors = palette or px.colors.qualitative.Plotly
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_hues)}

    seen_legend_groups = set()

    for col_idx, measurement in enumerate(measurements, start=1):
        params = PLOT_PARAMETERS[measurement]
        add_scatter_with_identity_to_plotly(
            fig=fig,
            df=df,
            x_col=params["x_col"],
            y_col=params["y_col"],
            row=1,
            col=col_idx,
            hue=hue,
            color_map=color_map,
            xlabel=params.get("xlabel"),
            ylabel=params.get("ylabel"),
            ref_lines=params.get("ref_lines"),
            use_log=params.get("use_log", False),
            markersize=markersize,
            seen_legend_groups=seen_legend_groups,
        )

    title_text = (
        f"Scatter plots for epistasis for {dataset}"
        if dataset
        else "Scatter plots for epistasis"
    )
    fig.update_layout(
        title={"text": title_text, "font": {"size": 18}, "x": 0.5},
        width=width_per_plot * n_cols,
        height=height,
        template="plotly_white",
        legend_title_text=hue if hue in df.columns else None,
    )

    return fig

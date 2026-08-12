import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PLOT_PARAMETERS

# Common Matplotlib-to-Plotly symbol translation helper
MATPLOTLIB_TO_PLOTLY_SYMBOLS = {
    "o": "circle",
    "s": "square",
    "^": "triangle-up",
    "v": "triangle-down",
    "D": "diamond",
    "x": "x",
    "+": "cross",
    "*": "star",
}


def add_scatter_with_identity_to_plotly(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    row: int = 1,
    col: int = 1,
    hue: str = "Number of Mutations",
    style: str | None = None,  # Column to vary marker shapes
    symbol: str | dict | None = None,  # Fixed symbol or mapping dict
    color_map: dict | None = None,
    hover_data: list[str] | str | None = None,  # Additional metadata columns
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

    # Helper function to format numbers cleanly (e.g., 1.0 -> "1")
    def clean_fmt(val):
        if val is None or pd.isna(val):
            return None
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val)

    # Normalize hover_data to a list
    if hover_data is None:
        hover_cols = []
    elif isinstance(hover_data, str):
        hover_cols = [hover_data]
    else:
        hover_cols = list(hover_data)

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

    # Symbol resolution
    def get_symbol_name(sym):
        return MATPLOTLIB_TO_PLOTLY_SYMBOLS.get(sym, sym)

    plotly_symbols_cycle = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "cross",
        "x",
        "star",
    ]

    # Build Plotly Traces grouped by hue and style
    group_cols = []
    if hue in data.columns:
        group_cols.append(hue)
    if style and style in data.columns and style != hue:
        group_cols.append(style)

    if group_cols:
        grouped = data.groupby(group_cols, observed=True)

        # Style mapping dictionary
        style_symbol_map = {}
        if style and style in data.columns:
            unique_styles = sorted(data[style].unique())
            if isinstance(symbol, dict):
                style_symbol_map = {
                    s: get_symbol_name(symbol.get(s, "circle")) for s in unique_styles
                }
            elif isinstance(symbol, str):
                style_symbol_map = {s: get_symbol_name(symbol) for s in unique_styles}
            else:
                style_symbol_map = {
                    s: plotly_symbols_cycle[i % len(plotly_symbols_cycle)]
                    for i, s in enumerate(unique_styles)
                }

        for group_key, sub_df in grouped:
            raw_hue_val = sub_df[hue].iloc[0] if hue in data.columns else None
            raw_style_val = (
                sub_df[style].iloc[0]
                if (style and style in data.columns)
                else None
            )

            hue_str = clean_fmt(raw_hue_val)
            style_str = clean_fmt(raw_style_val)

            # Internal group ID for trace linking
            leg_group = f"{hue_str}_{style_str}" if style_str else hue_str

            # Human-readable display label in the legend
            if hue_str and style_str:
                display_name = f"{hue_str} ({style_str})"
            elif hue_str:
                display_name = hue_str
            else:
                display_name = style_str or ""

            show_leg = False
            if seen_legend_groups is not None and leg_group not in seen_legend_groups:
                show_leg = True
                seen_legend_groups.add(leg_group)

            color = color_map.get(raw_hue_val) if color_map else None

            # Resolve marker symbol
            if raw_style_val and raw_style_val in style_symbol_map:
                marker_symbol = style_symbol_map[raw_style_val]
            elif isinstance(symbol, str):
                marker_symbol = get_symbol_name(symbol)
            else:
                marker_symbol = "circle"

            # Build Hover Text / Template
            hover_text_lines = []
            if hue_str:
                hover_text_lines.append(f"<b>{hue}: {hue_str}</b>")
            if style_str:
                hover_text_lines.append(f"<b>{style}: {style_str}</b>")
            hover_text_lines.append(
                f"{'log10(' if use_log else ''}{x_col}{')' if use_log else ''}: %{{x:.3f}}"
            )
            hover_text_lines.append(
                f"{'log10(' if use_log else ''}{y_col}{')' if use_log else ''}: %{{y:.3f}}"
            )

            # Extra hover metadata columns
            customdata = None
            if hover_cols:
                valid_hover_cols = [c for c in hover_cols if c in sub_df.columns]
                if valid_hover_cols:
                    customdata = sub_df[valid_hover_cols].values
                    for idx, hcol in enumerate(valid_hover_cols):
                        hover_text_lines.append(f"{hcol}: %{{customdata[{idx}]}}")

            hover_template = "<br>".join(hover_text_lines) + "<extra></extra>"

            fig.add_trace(
                go.Scatter(
                    x=sub_df[x_plot],
                    y=sub_df[y_plot],
                    mode="markers",
                    name=display_name,
                    legendgroup=leg_group,
                    showlegend=show_leg,
                    customdata=customdata,
                    marker={
                        "size": markersize,
                        "color": color,
                        "symbol": marker_symbol,
                        "line": {"width": 0.5, "color": "black"},
                        "opacity": 0.75,
                    },
                    hovertemplate=hover_template,
                ),
                row=row,
                col=col,
            )
    else:
        # Single trace without grouping
        marker_sym = get_symbol_name(symbol) if isinstance(symbol, str) else "circle"
        hover_text_lines = [
            f"{x_col}: %{{x:.3f}}",
            f"{y_col}: %{{y:.3f}}",
        ]
        customdata = None
        if hover_cols:
            valid_hover_cols = [c for c in hover_cols if c in data.columns]
            if valid_hover_cols:
                customdata = data[valid_hover_cols].values
                for idx, hcol in enumerate(valid_hover_cols):
                    hover_text_lines.append(f"{hcol}: %{{customdata[{idx}]}}")

        fig.add_trace(
            go.Scatter(
                x=data[x_plot],
                y=data[y_plot],
                mode="markers",
                showlegend=False,
                customdata=customdata,
                marker={
                    "size": markersize,
                    "symbol": marker_sym,
                    "line": {"width": 0.5, "color": "black"},
                    "opacity": 0.75,
                },
                hovertemplate="<br>".join(hover_text_lines) + "<extra></extra>",
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
    style: (
        str | dict | None
    ) = None,  # Categorical column name OR dict mapping measurement -> column
    symbol: (
        str | dict | None
    ) = None,  # Symbol name, style-to-symbol dict, OR measurement-to-symbol dict
    hover_data: list[str] | str | None = None,  # Metadata cols for hover
    width_per_plot: int = 420,
    height: int = 450,
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

        # 1. Resolve current style safely
        current_style = style.get(measurement) if isinstance(style, dict) else style

        # 2. Resolve current symbol if parameterized per measurement
        if isinstance(symbol, dict) and any(m in symbol for m in measurements):
            current_symbol = symbol.get(measurement)
        else:
            current_symbol = symbol

        add_scatter_with_identity_to_plotly(
            fig=fig,
            df=df,
            x_col=params["x_col"],
            y_col=params["y_col"],
            row=1,
            col=col_idx,
            hue=hue,
            style=current_style,
            symbol=current_symbol,
            color_map=color_map,
            hover_data=hover_data,
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
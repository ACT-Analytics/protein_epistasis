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

    # Symbol resolution helper
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

    # Map style column values to marker shapes
    style_symbol_map = {}
    unique_styles = []
    if style and style in data.columns:
        unique_styles = sorted(data[style].dropna().unique())
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

    # 1. GROUP BY HUE ONLY (Color Traces)
    if hue in data.columns:
        grouped = data.groupby(hue, observed=True)

        for raw_hue_val, sub_df in grouped:
            hue_str = clean_fmt(raw_hue_val)
            leg_group = f"hue_{hue}_{hue_str}"
            display_name = hue_str

            show_leg = False
            if seen_legend_groups is not None and leg_group not in seen_legend_groups:
                show_leg = True
                seen_legend_groups.add(leg_group)

            color = color_map.get(raw_hue_val) if color_map else None

            # Map symbols array per point if style is present
            if style and style in sub_df.columns:
                marker_symbols = [
                    style_symbol_map.get(s, "circle") for s in sub_df[style]
                ]
            elif isinstance(symbol, str):
                marker_symbols = get_symbol_name(symbol)
            else:
                marker_symbols = "circle"

            # Setup hover data
            valid_hover_cols = []
            if style and style in sub_df.columns:
                valid_hover_cols.append(style)
            for c in hover_cols:
                if c in sub_df.columns and c not in valid_hover_cols:
                    valid_hover_cols.append(c)

            hover_text_lines = [f"<b>{hue}: {hue_str}</b>"]
            customdata = None
            if valid_hover_cols:
                customdata = sub_df[valid_hover_cols].values
                for idx, hcol in enumerate(valid_hover_cols):
                    hover_text_lines.append(f"<b>{hcol}</b>: %{{customdata[{idx}]}}")

            hover_text_lines.append(
                f"{'log10(' if use_log else ''}{x_col}{')' if use_log else ''}: %{{x:.3f}}"
            )
            hover_text_lines.append(
                f"{'log10(' if use_log else ''}{y_col}{')' if use_log else ''}: %{{y:.3f}}"
            )

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
                        "symbol": marker_symbols,
                        "line": {"width": 0.5, "color": "black"},
                        "opacity": 0.75,
                    },
                    hovertemplate=hover_template,
                ),
                row=row,
                col=col,
            )

    # 2. GROUP BY STYLE ONLY (if hue is absent)
    elif style and style in data.columns:
        grouped = data.groupby(style, observed=True)
        for raw_style_val, sub_df in grouped:
            style_str = clean_fmt(raw_style_val)
            leg_group = f"style_{style}_{style_str}"
            display_name = style_str

            show_leg = False
            if seen_legend_groups is not None and leg_group not in seen_legend_groups:
                show_leg = True
                seen_legend_groups.add(leg_group)

            marker_symbol = style_symbol_map.get(raw_style_val, "circle")

            valid_hover_cols = [c for c in hover_cols if c in sub_df.columns]
            hover_text_lines = [f"<b>{style}: {style_str}</b>"]
            customdata = None
            if valid_hover_cols:
                customdata = sub_df[valid_hover_cols].values
                for idx, hcol in enumerate(valid_hover_cols):
                    hover_text_lines.append(f"<b>{hcol}</b>: %{{customdata[{idx}]}}")

            hover_text_lines.append(
                f"{'log10(' if use_log else ''}{x_col}{')' if use_log else ''}: %{{x:.3f}}"
            )
            hover_text_lines.append(
                f"{'log10(' if use_log else ''}{y_col}{')' if use_log else ''}: %{{y:.3f}}"
            )

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
                    hover_text_lines.append(f"<b>{hcol}</b>: %{{customdata[{idx}]}}")

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

    # 3. DUMMY LEGEND ENTRIES FOR SHAPES (when both hue and style are present)
    if hue in data.columns and style and style in data.columns:
        for style_val in unique_styles:
            style_str = clean_fmt(style_val)
            style_leg_key = f"dummy_shape_{style}_{style_str}"

            if (
                seen_legend_groups is not None
                and style_leg_key not in seen_legend_groups
            ):
                seen_legend_groups.add(style_leg_key)
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=f"{style}: {style_str}",
                        legendgroup=f"shape_key_{style_str}",
                        showlegend=True,
                        marker={
                            "size": markersize,
                            "color": "#4A5568",  # Neutral dark slate color
                            "symbol": style_symbol_map[style_val],
                            "line": {"width": 0.5, "color": "black"},
                        },
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
    style: str | dict | None = None,  # Categorical column or dict map
    symbol: str | dict | None = None,  # Symbol name or dict map
    hover_data: list[str] | str | None = None,
    width_per_plot: int = 588,
    height: int = 630,
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

        current_style = style.get(measurement) if isinstance(style, dict) else style

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

    # Dynamic legend title
    legend_title_parts = []
    if hue in df.columns:
        legend_title_parts.append(hue)
    if isinstance(style, str) and style in df.columns and style != hue:
        legend_title_parts.append(style)

    legend_title = " / ".join(legend_title_parts) if legend_title_parts else None

    fig.update_layout(
        title={"text": title_text, "font": {"size": 18}, "x": 0.5},
        width=width_per_plot * n_cols,
        height=height,
        template="plotly_white",
        legend_title_text=legend_title,
    )

    return fig


def plot_regression_results(
    model,
    X,
    y,
    target_name=None,
    feature_names=None,
    title=None,
    model_name="ElasticNet",
    hovertext:str|None=None, 
    width:float=1000, 
    height: float=600
):
    """Plots non-zero regression coefficients alongside predicted vs.

    actual values, dynamically adapting to any target variable `y`.

    Parameters:
    -----------
    model : fitted estimator
        Trained regression model with `.predict()` and `.coef_`.
    X : array-like or pd.DataFrame
        Feature matrix.
    y : array-like or pd.Series
        True target values.
    target_name : str, optional
        Label for the target variable (e.g., 'ΔT<sub>m</sub>', 'IC50', 'Yield').
        If None, attempts to infer from `y.name` or defaults to 'Target'.
    feature_names : list-like, optional
        Names of the features. Inferred if X is a DataFrame.
    title : str, optional
        Custom main figure title. If None, auto-generated from model and target names.
    model_name : str, default='ElasticNet'
        Name of the model to display in titles.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # 1. Infer target name if not provided
    if target_name is None:
        target_name = (
            getattr(y, "name", None) if getattr(y, "name", None) else "Target"
        )

    # 2. Predictions & target array
    preds = model.predict(X)
    y_true = np.asarray(y).ravel()

    # 3. Extract feature names and non-zero coefficients
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = X.columns
        else:
            feature_names = [f"Feature {i}" for i in range(len(model.coef_))]

    coef_series = pd.Series(model.coef_, index=feature_names)
    selected_coefs = coef_series[coef_series != 0].sort_values()

    # 4. Reference line limits (y = x)
    min_val = float(min(np.min(y_true), np.min(preds)))
    max_val = float(max(np.max(y_true), np.max(preds)))

    # 5. Build Subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Non-Zero Coefficients ({model_name})",
            f"Predicted vs. True ({target_name})",
        ),
        horizontal_spacing=0.15,
    )

    # Subplot 1: Coefficients
    fig.add_trace(
        go.Bar(
            x=selected_coefs.values,
            y=selected_coefs.index.astype(str),
            orientation="h",
            name="Coefficients",
            marker={"color": "#4C72B0"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Predicted vs Actual Scatter
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=preds,
            mode="markers",
            name="Predictions",
            marker={"size": 7, "opacity": 0.7, "color": "#55A868"},
            hovertext = hovertext,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Subplot 2: Identity reference line
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line={"color": "grey", "dash": "dash", "width": 1.5},
        opacity=0.4,
        row=1,
        col=2,
    )

    # 6. Dynamic Axis Labels & Layout
    fig.update_xaxes(title_text="Coefficient Value", row=1, col=1)
    fig.update_yaxes(title_text="Feature", row=1, col=1)

    fig.update_xaxes(title_text=f"{target_name} (True)", row=1, col=2)
    fig.update_yaxes(title_text=f"{target_name} (Predicted)", row=1, col=2)

    main_title = (
        title
        if title is not None
        else f"Results of {model_name} Fit for {target_name}"
    )

    fig.update_layout(
        title={"text": main_title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        width=width,
        height=height,
        margin={"t": 80, "b": 50, "l": 70, "r": 40},
    )

    return fig
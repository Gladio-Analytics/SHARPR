import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .theme import (
    build_style_registry,
    style_for_series,
    _apply_white_chart_theme,
    _style_color_lookup,
    _as_df,
    _series_for_plot,
    _default_palette,
    _white_xaxis_dict,
    _white_yaxis_dict,
)


def plot_loadings_wide(
    loadings_wide,
    factor_order=None,
    title="Factor Loadings and Alpha",
    show_values=True,
    height=560,
    color_map=None,
    annualize_alpha=False,
    alpha_unit="auto",
    periods_per_year=252,
    alpha_compound=True,
    legend_x=1.30,
    right_margin=280,
    alpha_axis_pad_frac=0.35,
    alpha_textposition="outside",
    style_registry=None
):
    d = loadings_wide.copy()
    d.index = [str(i).upper() for i in d.index]
    d.columns = [str(c) for c in d.columns]
    d = d.apply(pd.to_numeric, errors="coerce")

    if factor_order is None:
        factor_order = ["MKT", "HML", "SMB", "CMA", "RMW", "MOM", "STREV", "LTREV"]

    factor_order = [str(f).upper() for f in factor_order]
    idx_set = set(d.index.tolist())

    def _resolve_factor_name(f):
        if f in idx_set:
            return f
        if f == "SMT" and "SMB" in idx_set:
            return "SMB"
        if f == "SMB" and "SMT" in idx_set:
            return "SMT"
        return None

    factors = []
    for f in factor_order:
        r = _resolve_factor_name(f)
        if r is not None and r not in factors and r != "ALPHA":
            factors.append(r)

    tickers = d.columns.tolist()
    n_tickers = len(tickers)

    alpha_name = "ALPHA" if "ALPHA" in idx_set else None

    alpha_share = min(0.55, max(0.30, 0.035 * n_tickers))
    factor_share = 1.0 - alpha_share

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[factor_share, alpha_share],
        subplot_titles=("Factors", "Alpha"),
        vertical_spacing=0.22
    )

    for tkr in tickers:
        if not factors:
            continue
        y = d.loc[factors, tkr].values
        txt = [f"{v:.3f}" if pd.notna(v) else "" for v in y] if show_values else None
        st = style_for_series(tkr, style_registry)

        fig.add_trace(
            go.Bar(
                x=factors,
                y=y,
                name=tkr,
                marker_color=st["color"],
                text=txt,
                textposition="outside" if show_values else None,
                hovertemplate="<b>%{fullData.name}</b><br>Factor: %{x}<br>Loading: %{y:.6f}<extra></extra>"
            ),
            row=1,
            col=1
        )

    alpha_label = "Alpha"

    if alpha_name is not None:
        alpha_raw = d.loc[alpha_name, tickers].astype(float)

        if annualize_alpha:
            if alpha_unit == "auto":
                max_abs = float(np.nanmax(np.abs(alpha_raw.values))) if np.isfinite(alpha_raw.values).any() else np.nan
                unit = "decimal" if (np.isfinite(max_abs) and max_abs <= 0.02) else "percent"
            else:
                unit = str(alpha_unit).lower()

            a_dec = alpha_raw / 100.0 if unit == "percent" else alpha_raw
            a_ann_dec = (1.0 + a_dec).pow(periods_per_year) - 1.0 if alpha_compound else a_dec * periods_per_year
            alpha_vals_plot = a_ann_dec * 100.0
            alpha_label = "Alpha (annualized, %)"
        else:
            alpha_vals_plot = alpha_raw.copy()
            if alpha_unit == "percent":
                alpha_label = "Alpha (daily, %)"
            elif alpha_unit == "decimal":
                alpha_label = "Alpha (daily, decimal)"
            else:
                alpha_label = "Alpha (daily)"

        alpha_vals_plot = alpha_vals_plot.reindex(tickers)
        xs = [float(alpha_vals_plot.loc[t]) for t in tickers]

        if show_values:
            if annualize_alpha:
                txt = [f"{v:.2f}%" for v in xs]
            else:
                txt = [f"{v:.3f}" for v in xs]
        else:
            txt = None

        colors = [style_for_series(t, style_registry)["color"] for t in tickers]

        fig.add_trace(
            go.Bar(
                x=xs,
                y=tickers,
                orientation="h",
                marker_color=colors,
                text=txt,
                textposition=alpha_textposition if show_values else None,
                cliponaxis=False,
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>" + alpha_label + ": %{x:.6f}<extra></extra>"
            ),
            row=2,
            col=1
        )
    else:
        fig.add_annotation(
            x=0.5, y=0.5, xref="x domain", yref="y2 domain",
            text="ALPHA not found",
            showarrow=False
        )

    if factors:
        fig.update_xaxes(categoryorder="array", categoryarray=factors, row=1, col=1)

    fig.update_yaxes(title_text="Loading", row=1, col=1)

    if alpha_name is not None:
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=tickers,
            row=2,
            col=1
        )
        fig.update_yaxes(showticklabels=False, row=2, col=1)

        vmax = float(np.nanmax(np.abs(np.asarray(xs, dtype=float)))) if len(xs) else 0.0
        pad = float(alpha_axis_pad_frac) * vmax
        if np.isfinite(pad) and pad > 0:
            fig.update_xaxes(range=[-vmax - pad, vmax + pad], row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=int(height),
        barmode="group",
        legend=dict(
            x=float(legend_x),
            y=1.0,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(l=80, r=int(right_margin), t=85, b=60)
    )

    return _apply_white_chart_theme(fig)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
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


def forward_horizon_returns(returns, horizon=5, include_today=False):
    r = _as_df(returns).copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)

    r = r.dropna(how="any")
    if r.empty:
        return r

    L = np.log1p(r)
    cs = L.cumsum(axis=0)

    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be a positive integer.")

    if include_today:
        fsum = cs.shift(-(h - 1)) - cs.shift(1, fill_value=0.0)
    else:
        fsum = cs.shift(-h) - cs

    out = np.expm1(fsum)
    out.index.name = r.index.name
    return out


def fig_forward_horizon_distribution(
    returns_all,
    portfolio=None,
    horizons=(5, 21),
    bins=70,
    title=None,
    include_today=False,
    show_percentiles=(0.05, 0.50, 0.95),
    dropdown_x=1.02,
    dropdown_y=1.12,
):
    r = _as_df(returns_all)
    if r is None or r.empty:
        fig = go.Figure()
        fig.update_layout(title="No return data available", height=520)
        return _apply_white_chart_theme(fig)

    portfolios = [str(c) for c in r.columns]
    if len(portfolios) == 0:
        fig = go.Figure()
        fig.update_layout(title="No portfolios available", height=520)
        return _apply_white_chart_theme(fig)

    if portfolio is None:
        portfolio = portfolios[0]
    if portfolio not in portfolios:
        raise KeyError(f"portfolio '{portfolio}' not in returns_all columns.")

    fig = go.Figure()

    trace_map = {}
    shape_map = {}
    title_map = {}
    valid_portfolios = []

    for p in portfolios:
        trace_map[p] = []
        shape_map[p] = []

        added_any = False

        for h in horizons:
            fwd = forward_horizon_returns(
                r[[p]],
                horizon=h,
                include_today=include_today
            )[p].dropna()

            if fwd.empty:
                continue

            added_any = True
            vals = fwd.values.astype(float)

            fig.add_trace(
                go.Histogram(
                    x=vals,
                    nbinsx=int(bins),
                    name=f"Forward {int(h)}d",
                    opacity=0.45,
                    histnorm="probability density",
                    visible=False,
                )
            )
            trace_map[p].append(len(fig.data) - 1)

            if len(vals) >= 10 and np.nanstd(vals) > 0:
                kde = gaussian_kde(vals)
                xs = np.linspace(np.nanmin(vals), np.nanmax(vals), 220)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=kde(xs),
                        mode="lines",
                        name=f"KDE {int(h)}d",
                        visible=False,
                    )
                )
                trace_map[p].append(len(fig.data) - 1)

            for q in show_percentiles:
                xq = float(np.quantile(vals, q))
                shape_map[p].append(
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=xq,
                        x1=xq,
                        y0=0,
                        y1=1,
                        line=dict(width=1),
                        opacity=0.35,
                    )
                )

        if added_any:
            valid_portfolios.append(p)
            if title is None:
                title_map[p] = (
                    f"{p}: Forward Holding-Period Return Distribution "
                    f"({', '.join(str(int(h)) + 'd' for h in horizons)})"
                )
            else:
                title_map[p] = f"{title} — {p}"
        else:
            title_map[p] = f"{p}: no forward data available"

    if len(fig.data) == 0:
        fig.update_layout(title="No forward data available", height=520)
        return _apply_white_chart_theme(fig)

    default_portfolio = portfolio

    for idx in trace_map.get(default_portfolio, []):
        fig.data[idx].visible = True

    buttons = []
    n_traces = len(fig.data)

    for p in portfolios:
        visible = [False] * n_traces
        for idx in trace_map.get(p, []):
            visible[idx] = True

        buttons.append(
            dict(
                label=p,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": title_map.get(p, p),
                        "shapes": shape_map.get(p, []),
                    },
                ],
            )
        )

    fig.update_layout(
        title=title_map.get(default_portfolio, default_portfolio),
        xaxis_title="Forward Return (decimal)",
        yaxis_title="Density",
        barmode="overlay",
        height=520,
        legend_title="",
        shapes=shape_map.get(default_portfolio, []),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=dropdown_x,
                y=dropdown_y,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
            )
        ],
        annotations=[
            dict(
                text="Portfolio",
                x=dropdown_x,
                y=dropdown_y + 0.04,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            )
        ],
    )

    return _apply_white_chart_theme(fig)

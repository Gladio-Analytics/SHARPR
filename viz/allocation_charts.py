import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def plot_allocation_risk_sharpe_switcher(
    weights_all,
    rc_pct_all,
    sc_pct_all_heur,
    height=520,
    width=None,
    style_registry=None
):
    import plotly.graph_objects as go

    methods = list(weights_all.columns)
    tickers = list(weights_all.index)

    def _prep(df):
        X = df.reindex(index=tickers).fillna(0.0).astype(float)
        return X.reindex(columns=methods)

    W = _prep(weights_all)
    RC = _prep(rc_pct_all)
    SC = _prep(sc_pct_all_heur)

    fig = go.Figure()

    for t in tickers:
        st = style_for_series(t, style_registry)
        fig.add_trace(
            go.Bar(
                x=methods,
                y=W.loc[t].values,
                name=str(t),
                marker_color=st["color"],
                visible=True,
                showlegend=True,
            )
        )

    for t in tickers:
        st = style_for_series(t, style_registry)
        fig.add_trace(
            go.Bar(
                x=methods,
                y=RC.loc[t].values,
                name=str(t),
                marker_color=st["color"],
                visible=False,
                showlegend=False,
            )
        )

    for t in tickers:
        st = style_for_series(t, style_registry)
        fig.add_trace(
            go.Bar(
                x=methods,
                y=SC.loc[t].values,
                name=str(t),
                marker_color=st["color"],
                visible=False,
                showlegend=False,
            )
        )

    n = len(tickers)

    vis_alloc = [True] * n + [False] * n + [False] * n
    vis_risk = [False] * n + [True] * n + [False] * n
    vis_sharp = [False] * n + [False] * n + [True] * n

    leg_alloc = [True] * n + [False] * n + [False] * n
    leg_risk = [False] * n + [True] * n + [False] * n
    leg_sharp = [False] * n + [False] * n + [True] * n

    fig.update_layout(
        barmode="stack",
        title="Portfolio Allocation",
        height=height,
        width=width,
        margin=dict(l=40, r=240, t=70, b=40),
        legend=dict(
            title="Ticker",
            orientation="v",
            x=1.02,
            y=0.0,
            xanchor="left",
            yanchor="bottom",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Portfolio Allocation",
                        method="update",
                        args=[
                            {"visible": vis_alloc, "showlegend": leg_alloc},
                            {"title": "Portfolio Allocation"},
                        ],
                    ),
                    dict(
                        label="Risk Contribution",
                        method="update",
                        args=[
                            {"visible": vis_risk, "showlegend": leg_risk},
                            {"title": "Risk Contribution"},
                        ],
                    ),
                    dict(
                        label="Sharpe-Ratio Contribution",
                        method="update",
                        args=[
                            {"visible": vis_sharp, "showlegend": leg_sharp},
                            {"title": "Sharpe-Ratio Contribution"},
                        ],
                    ),
                ],
            )
        ],
    )

    fig.update_yaxes(tickformat=".0%")
    return _apply_white_chart_theme(fig)


def fig_capital_reallocation_bars(
    weights_all,
    capital=1.0,
    current_portfolio_name="Current Portfolio",
    show="delta_if_possible",
    as_pct=False,
    drop_current_from_dropdown=False,
    styles=None,
    default_bar_color="#888888",
    height=520,
    title="Capital allocation / re-allocation",
):
    if not isinstance(weights_all, pd.DataFrame):
        raise TypeError("weights_all must be a pandas DataFrame (index=tickers, columns=portfolios).")
    if weights_all.shape[1] < 1:
        raise ValueError("weights_all must have at least 1 portfolio column.")

    w = weights_all.copy()
    w.index = w.index.astype(str)
    w.columns = w.columns.astype(str)
    w = w.apply(pd.to_numeric, errors="coerce")

    tickers = [t.strip() for t in w.index.tolist()]
    w.index = tickers

    portfolios_all = list(w.columns)
    has_current = current_portfolio_name in portfolios_all

    show = str(show).lower()
    if show not in ("delta_if_possible", "delta", "absolute"):
        raise ValueError("show must be one of: 'delta_if_possible', 'delta', 'absolute'.")
    if show == "delta" and not has_current:
        raise KeyError(f"show='delta' requested but '{current_portfolio_name}' not found in weights_all.columns")

    if show == "absolute":
        mode = "absolute"
    elif show == "delta":
        mode = "delta"
    else:
        mode = "delta" if has_current else "absolute"

    w0 = w[current_portfolio_name] if has_current else None

    portfolios = portfolios_all.copy()
    if drop_current_from_dropdown and has_current:
        portfolios = [p for p in portfolios if p != current_portfolio_name]
    if not portfolios:
        raise ValueError("No portfolios left for dropdown after filtering.")

    def _ticker_color(tkr):
        if styles is None or not isinstance(styles, dict):
            return default_bar_color

        t = str(tkr).strip()
        candidates = (t, t.lower(), t.upper())

        for k in candidates:
            d = styles.get(k)
            if isinstance(d, dict) and d.get("color"):
                return d["color"]

        tlow = t.lower()
        for _, d in styles.items():
            if isinstance(d, dict):
                nm = str(d.get("name", "")).strip().lower()
                if nm == tlow and d.get("color"):
                    return d["color"]

        return default_bar_color

    bar_colors = [_ticker_color(t) for t in tickers]

    def _series_for_portfolio(p):
        wp = w[p].copy()

        if mode == "delta" and has_current:
            d = (wp - w0).astype(float)
            if as_pct:
                y = d.values * 100.0
                ytitle = "Δ weight vs Current (%)"
                ttl = f"{title}: {p} (Δ vs {current_portfolio_name})"
            else:
                y = d.values * float(capital)
                ytitle = f"Δ $ vs {current_portfolio_name}  (capital={capital:g})"
                ttl = f"{title}: {p} (Δ$ vs {current_portfolio_name})"
            return y, ytitle, ttl

        if as_pct:
            y = wp.values * 100.0
            ytitle = "Weight (%)"
            ttl = f"{title}: {p} (absolute)"
        else:
            y = wp.values * float(capital)
            ytitle = f"$ allocation  (capital={capital:g})"
            ttl = f"{title}: {p} (absolute $)"
        return y, ytitle, ttl

    p0 = portfolios[0]
    y0, ytitle0, ttl0 = _series_for_portfolio(p0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=y0,
            marker=dict(color=bar_colors),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
        )
    )

    buttons = []
    for p in portfolios:
        y, ytitle, ttl = _series_for_portfolio(p)
        buttons.append(
            dict(
                label=p,
                method="update",
                args=[
                    {"y": [y], "marker": [dict(color=bar_colors)]},
                    {
                        "title": {"text": ttl},
                        "xaxis": _white_xaxis_dict(title="Ticker", type="category", showgrid=False),
                        "yaxis": _white_yaxis_dict(title=ytitle, showgrid=True, zeroline=True),
                    },
                ],
            )
        )

    hint = (
        f" (sell = negative, buy = positive vs '{current_portfolio_name}')"
        if mode == "delta"
        else ""
    )

    fig.update_layout(
        title=ttl0 + hint,
        height=height,
        margin=dict(l=20, r=20, t=70, b=40),
        xaxis=_white_xaxis_dict(title="Ticker", type="category", showgrid=False),
        yaxis=_white_yaxis_dict(title=ytitle0, showgrid=True, zeroline=True),
        showlegend=False,
        updatemenus=[
            dict(
                type="dropdown",
                x=1.0,
                y=1.16,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                showactive=True,
            )
        ],
    )
    return _apply_white_chart_theme(fig)

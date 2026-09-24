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


def plot_return_panel(
    asset_returns,
    benchmark=None,
    risk_free_rate=None,
    custom_portfolio=None,
    current_portfolio=None,
    cumulative=True,
    title="Returns",
    palette=None,
    style_registry=None
):
    assets = _as_df(asset_returns)
    bench = _as_df(benchmark)
    rf = _as_df(risk_free_rate)
    custom = _as_df(custom_portfolio)
    current = _as_df(current_portfolio)

    if assets is None:
        return _apply_white_chart_theme(go.Figure())

    fig = go.Figure()

    for c in assets.columns:
        st = style_for_series(c, style_registry)
        y = _series_for_plot(assets[c], cumulative=cumulative)
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y.values,
                mode="lines",
                name=str(c),
                line=dict(color=st["color"], dash=st["dash"], width=st["width"]),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>" if cumulative else None
            )
        )

    def _add_one(df):
        if df is None:
            return
        for c in df.columns:
            st = style_for_series(c, style_registry)
            y = _series_for_plot(df[c], cumulative=cumulative)
            fig.add_trace(
                go.Scatter(
                    x=y.index,
                    y=y.values,
                    mode="lines",
                    name=str(c),
                    line=dict(color=st["color"], dash=st["dash"], width=st["width"]),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>" if cumulative else None
                )
            )

    _add_one(custom)
    _add_one(current)
    _add_one(bench)
    _add_one(rf)

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (decimal)" if cumulative else "Return (decimal)",
        legend_title_text=""
    )
    return _apply_white_chart_theme(fig)


def plot_cum_returns_and_drawdowns(
    portfolio_returns,
    title="Cumulative Returns & Drawdowns",
    height=720,
    width=None,
    style_registry=None
):
    df = portfolio_returns.copy().sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all")

    r = df.fillna(0.0)
    wealth = (1.0 + r).cumprod()
    cum = wealth - 1.0
    dd = wealth.div(wealth.cummax()).sub(1.0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=("Cumulative Return (decimal)", "Drawdown (%)")
    )

    for col in df.columns:
        name = str(col)
        st = style_for_series(name, style_registry)

        fig.add_trace(
            go.Scatter(
                x=cum.index, y=cum[col],
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(color=st["color"], dash=st["dash"], width=st["width"]),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd[col],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=st["color"], dash=st["dash"], width=st["width"]),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=height,
        width=width,
        margin=dict(l=50, r=230, t=80, b=50),
        legend=dict(
            x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            orientation="v",
            title_text=""
        )
    )

    fig.update_yaxes(title_text="Cumulative Return (decimal)", tickformat=".0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)

    return _apply_white_chart_theme(fig)


def plot_return_correlation_heatmap(
    asset_returns,
    benchmark=None,
    risk_free_rate=None,
    title="Return Correlation",
):
    returns = asset_returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if benchmark is not None:
        benchmark = benchmark.copy()
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]
        benchmark.name = "Benchmark"
        returns = pd.concat([returns, benchmark], axis=1)

    if risk_free_rate is not None:
        risk_free_rate = risk_free_rate.copy()
        if isinstance(risk_free_rate, pd.DataFrame):
            risk_free_rate = risk_free_rate.iloc[:, 0]
        risk_free_rate.name = "Risk Free Rate"
        returns = pd.concat([returns, risk_free_rate], axis=1)

    returns = returns.dropna()
    corr = returns.corr()

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr_lower = corr.mask(mask)

    text = corr_lower.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_lower.values,
            x=corr_lower.columns,
            y=corr_lower.index,
            zmin=-1, zmax=1, zmid=0,
            colorscale=[[0.0, "#b2182b"], [0.5, "#ffffff"], [1.0, "#1a9850"]],
            text=text.values,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1]),
            hoverongaps=False,
        )
    )

    n = len(corr.columns)
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(side="bottom", tickangle=-45, showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=max(500, 48 * n + 150),
        margin=dict(l=110, r=80, t=80, b=110),
    )

    return _apply_white_chart_theme(fig)

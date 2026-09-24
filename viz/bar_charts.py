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


def fig_metric_barchart(
    total_df,
    metric,
    baseline=None,
    mode="pct",
    pct_scale=100.0,
    eps=1e-12,
    portfolios=None,
    styles=None,
    default_bar_color="#888888",
    height=520,
    title=None,
):
    if not isinstance(total_df, pd.DataFrame):
        raise TypeError("total_df must be a DataFrame (rows=portfolios, cols=metrics).")
    if metric not in total_df.columns:
        raise KeyError(f"metric '{metric}' not found in total_df.columns")

    df = total_df.copy()
    if portfolios is not None:
        df = df.loc[list(portfolios)].copy()

    df = df.apply(pd.to_numeric, errors="coerce")

    ports = [str(p).strip() for p in df.index.astype(str)]
    df.index = ports
    s = df[metric].reindex(ports).astype(float)

    mode = str(mode).lower()
    if mode not in ("level", "abs", "pct"):
        raise ValueError("mode must be one of: 'level', 'abs', 'pct'.")

    if mode == "level" or baseline is None:
        y = s.values
        y_title = str(metric)
        ttl = title or f"{metric} (levels)"
    else:
        baseline = str(baseline).strip()
        if baseline not in ports:
            raise KeyError(f"baseline '{baseline}' not found in total_df.index (portfolios).")

        b = float(s.loc[baseline])
        denom = abs(b) if (np.isfinite(b) and abs(b) > eps) else eps

        if mode == "abs":
            y = (s - b).values
            y_title = f"{metric} (Δ vs {baseline})"
            ttl = title or f"{metric}: Δ vs {baseline}"
        else:
            y = ((s - b) / denom * pct_scale).values
            y_title = f"{metric} (% vs {baseline})"
            ttl = title or f"{metric}: % vs {baseline}"

    bar_colors = [style_for_series(p, styles).get("color", default_bar_color) for p in ports]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ports,
        y=y,
        marker=dict(color=bar_colors),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
    ))
    fig.update_layout(
        title=ttl,
        height=height,
        margin=dict(l=20, r=20, t=70, b=40),
        xaxis=dict(title="Portfolio", type="category"),
        yaxis=dict(title=y_title),
        showlegend=False,
    )
    return _apply_white_chart_theme(fig)


def fig_metric_dropdown_barchart(
    total_df,
    metrics=None,
    baseline="Current Portfolio",
    mode="pct",
    pct_scale=100.0,
    eps=1e-12,
    portfolios=None,
    styles=None,
    default_bar_color="#888888",
    height=520,
    title="Portfolio comparison",
):
    if not isinstance(total_df, pd.DataFrame):
        raise TypeError("total_df must be a DataFrame (rows=portfolios, cols=metrics).")

    df = total_df.copy()
    if portfolios is not None:
        df = df.loc[list(portfolios)].copy()

    df = df.apply(pd.to_numeric, errors="coerce")
    ports = [str(p).strip() for p in df.index.astype(str)]
    df.index = ports

    if metrics is None:
        metrics = [c for c in df.columns if df[c].notna().sum() >= 2]
    else:
        metrics = [c for c in metrics if c in df.columns]
    if not metrics:
        raise ValueError("No metrics available to plot.")

    baseline_used = baseline if (baseline is not None and str(baseline).strip() in ports) else None

    mode = str(mode).lower()
    if mode not in ("pct", "abs", "level"):
        raise ValueError("mode must be one of: 'pct', 'abs', 'level'.")

    mode_used = "level" if (baseline_used is None and mode in ("pct", "abs")) else mode
    bar_colors = [style_for_series(p, styles).get("color", default_bar_color) for p in ports]

    def _compute_y(metric):
        s = df[metric].reindex(ports).astype(float)

        if mode_used == "level" or baseline_used is None:
            y = s.values
            ytitle = str(metric)
            ttl = f"{title}: {metric} | levels"
            return y, ytitle, ttl

        b = float(s.loc[str(baseline_used).strip()])
        denom = abs(b) if (np.isfinite(b) and abs(b) > eps) else eps

        if mode_used == "abs":
            y = (s - b).values
            ytitle = f"{metric} (Δ vs {baseline_used})"
            ttl = f"{title}: {metric} | Δ vs {baseline_used}"
        else:
            y = ((s - b) / denom * pct_scale).values
            ytitle = f"{metric} (% vs {baseline_used})"
            ttl = f"{title}: {metric} | % vs {baseline_used}"

        return y, ytitle, ttl

    metric0 = metrics[0]
    y0, ytitle0, ttl0 = _compute_y(metric0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ports,
            y=y0,
            marker=dict(color=bar_colors),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
        )
    )

    buttons = []
    for m in metrics:
        y, ytitle, ttl = _compute_y(m)
        buttons.append(
            dict(
                label=str(m),
                method="update",
                args=[
                    {"y": [y], "x": [ports], "marker": [dict(color=bar_colors)]},
                    {
                        "title": {"text": ttl},
                        "xaxis": _white_xaxis_dict(title="Portfolio", type="category", showgrid=False),
                        "yaxis": _white_yaxis_dict(title=ytitle, showgrid=True, zeroline=True),
                    },
                ],
            )
        )

    subtitle = "" if baseline_used is None else f" (baseline={baseline_used}, mode={mode_used})"

    fig.update_layout(
        title=ttl0 + subtitle,
        height=height,
        margin=dict(l=20, r=20, t=70, b=40),
        xaxis=_white_xaxis_dict(title="Portfolio", type="category", showgrid=False),
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

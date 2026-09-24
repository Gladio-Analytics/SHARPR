import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
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


def _stats_row(stats, metric):
    if metric not in stats.index:
        raise KeyError(f"Metric '{metric}' not found in stats.index.")
    row = stats.loc[metric]
    if isinstance(row, pd.DataFrame):  # duplicate index edge case
        row = row.iloc[0]
    s = pd.to_numeric(pd.Series(row), errors="coerce")
    s.name = metric
    return s


def _infer_kind(metric):
    m = str(metric)
    if "Days" in m or "Day" in m or "Calendar" in m:
        return "days"
    pct_like = (
        "CAGR" in m
        or "Return" in m
        or "Drawdown" in m
        or "Volatility" in m
        or "Deviation" in m
        or "VaR" in m
        or "CVaR" in m
        or "Tracking Error" in m
        or "Active Return" in m
        or "Win Rate" in m
        or m.startswith("Risk of Ruin")
    )
    return "pct" if pct_like else "ratio"


def _valueformat(kind):
    if kind == "pct":
        return lambda v: "" if pd.isna(v) else f"{v*100:.2f}%"
    if kind == "days":
        return lambda v: "" if pd.isna(v) else f"{v:.0f}"
    return lambda v: "" if pd.isna(v) else f"{v:.3f}"


def _apply_transform(series, transform):
    if transform is None:
        return series
    if callable(transform):
        return series.apply(transform)
    t = str(transform).lower()
    if t == "abs":
        return series.abs()
    if t == "neg":
        return -series
    if t == "identity":
        return series
    raise ValueError("transform must be None, callable, or one of: 'abs','neg','identity'.")


def _default_transform(metric):
    m = str(metric).lower()
    # Make “pain” metrics positive for readability
    if "max. drawdown" in m or m == "max drawdown":
        return "abs"
    if "var" in m or "cvar" in m:
        return "abs"
    return None


def _default_higher_is_better(metric):
    m = str(metric).lower()
    # These are “lower is better”
    for kw in ["volatility", "semi-deviation", "tracking error", "ulcer index", "risk of ruin"]:
        if kw in m:
            return False
    # For drawdown/VAR (often negative), we transform to abs by default, so lower is better after abs
    if "max. drawdown" in m or "var" in m or "cvar" in m:
        return False
    return True


def plot_gain_pain_scatter(
    stats,
    x_metric,
    y_metric,
    z_metric=None,
    size_metric=None,
    color_metric=None,
    portfolios=None,
    transform_map=None,
    title=None,
    styles=None,
    default_color="#888888"
):
    if portfolios is None:
        portfolios = list(stats.columns)
    else:
        portfolios = list(portfolios)

    transform_map = {} if transform_map is None else dict(transform_map)

    def _norm_key(s):
        return str(s).strip().lower()

    def _style_color_lookup(label):
        if styles is None:
            return default_color
        k = _norm_key(label)
        if k in styles and isinstance(styles[k], dict) and "color" in styles[k]:
            return styles[k]["color"]
        for kk, v in styles.items():
            if isinstance(v, dict) and _norm_key(v.get("name", "")) == k and "color" in v:
                return v["color"]
        return default_color

    def get(metric):
        s = _stats_row(stats, metric).reindex(portfolios)
        tr = transform_map.get(metric, _default_transform(metric))
        return _apply_transform(s, tr)

    x = get(x_metric)
    y = get(y_metric)
    df = pd.DataFrame({"x": x, "y": y}, index=portfolios).dropna(subset=["x", "y"])

    if z_metric is not None:
        df["z"] = get(z_metric).reindex(df.index)

    if size_metric is not None:
        df["size_raw"] = get(size_metric).reindex(df.index)

    if color_metric is not None:
        df["color_raw"] = get(color_metric).reindex(df.index)

    marker = dict()

    if "size_raw" in df.columns:
        s = pd.to_numeric(df["size_raw"], errors="coerce")
        s2 = s.fillna(s.median())
        lo, hi = float(s2.min()), float(s2.max())
        if np.isclose(lo, hi):
            marker["size"] = np.full(len(df), 22.0)
        else:
            marker["size"] = 12 + (s2 - lo) / (hi - lo) * (44 - 12)

    if "color_raw" in df.columns:
        marker["color"] = pd.to_numeric(df["color_raw"], errors="coerce").values
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
        marker["colorbar"] = dict(title=color_metric)
    else:
        marker["color"] = [_style_color_lookup(p) for p in df.index]

    hover = []
    for p, row in df.iterrows():
        lines = [f"<b>{p}</b>",
                 f"{x_metric}: {row['x']}",
                 f"{y_metric}: {row['y']}"]
        if z_metric is not None and "z" in df.columns:
            lines.append(f"{z_metric}: {row['z']}")
        if "size_raw" in df.columns:
            lines.append(f"{size_metric}: {row['size_raw']}")
        if "color_raw" in df.columns:
            lines.append(f"{color_metric}: {row['color_raw']}")
        hover.append("<br>".join(lines) + "<extra></extra>")

    if title is None:
        title = f"{y_metric} vs {x_metric}" + (f" (3D: {z_metric})" if z_metric else "")

    if z_metric is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers+text",
            text=df.index.astype(str),
            textposition="top center",
            marker=marker,
            hovertemplate=hover,
            showlegend=False
        ))
        fig.update_layout(
            title=title,
            height=560,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title=x_metric,
            yaxis_title=y_metric
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        mode="markers+text",
        text=df.index.astype(str),
        textposition="top center",
        marker=marker,
        hovertemplate=hover,
        showlegend=False
    ))
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(xaxis_title=x_metric, yaxis_title=y_metric, zaxis_title=z_metric)
    )
    return _apply_white_chart_theme(fig)


def plot_scorecard_grid(
    stats,
    metrics,
    portfolios=None,
    kind_map=None,
    higher_is_better=None,
    transform_map=None,
    colorscale="RdYlGn",
    tile_size=46,
    title="Scorecard",
):
    if not isinstance(stats, pd.DataFrame):
        raise TypeError("stats must be a DataFrame (rows=metrics, cols=portfolios).")

    metrics_req = list(metrics)
    if len(metrics_req) == 0:
        raise ValueError("metrics must be a non-empty list.")

    if portfolios is None:
        portfolios = list(stats.columns)
    else:
        portfolios = [p for p in list(portfolios) if p in stats.columns]

    metrics_keep = [m for m in metrics_req if m in stats.index]

    if len(metrics_keep) > 10:
        raise ValueError("Use at most 10 available metrics for readability.")

    if len(portfolios) == 0 or len(metrics_keep) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} (nothing available to plot)",
            height=220,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    kind_map = {} if kind_map is None else dict(kind_map)
    higher_is_better = {} if higher_is_better is None else dict(higher_is_better)
    transform_map = {} if transform_map is None else dict(transform_map)

    val = pd.DataFrame(index=metrics_keep, columns=portfolios, dtype=float)
    for m in metrics_keep:
        s = _stats_row(stats, m).reindex(portfolios)
        tr = transform_map.get(m, _default_transform(m))
        s = _apply_transform(s, tr)
        val.loc[m] = s.values

    good = pd.DataFrame(index=metrics_keep, columns=portfolios, dtype=float)
    for m in metrics_keep:
        hib = higher_is_better.get(m, _default_higher_is_better(m))
        x = pd.to_numeric(val.loc[m], errors="coerce")
        if x.notna().sum() <= 1:
            good.loc[m] = np.nan
            continue
        ranks = x.rank(pct=True, method="average")
        good.loc[m] = ranks if hib else (1.0 - ranks)

    xs, ys, cs, texts, hovers = [], [], [], [], []
    for m in metrics_keep:
        fmt = _valueformat(kind_map.get(m, _infer_kind(m)))
        for p in portfolios:
            v = val.loc[m, p]
            g = good.loc[m, p]

            xs.append(p)
            ys.append(m)
            cs.append(g)
            texts.append(fmt(v))
            hovers.append(
                f"<b>{p}</b><br>"
                f"{m}: {fmt(v)}<br>"
                f"Rank score: {'' if pd.isna(g) else f'{g:.2f}'}"
                "<extra></extra>"
            )

    colors = []
    for g in cs:
        if pd.isna(g):
            colors.append("rgba(180,180,180,0.5)")
        else:
            colors.append(sample_colorscale(colorscale, float(np.clip(g, 0, 1)))[0])

    n_missing = len(metrics_req) - len(metrics_keep)
    title_used = title if n_missing == 0 else f"{title} (omitted {n_missing} unavailable metric(s))"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=texts,
            textposition="middle center",
            hovertemplate=hovers,
            marker=dict(
                symbol="square",
                size=tile_size,
                color=colors,
                line=dict(width=1, color="rgba(0,0,0,0.25)")
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        title=title_used,
        height=140 + 55 * len(metrics_keep),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(type="category", title="Portfolio"),
        yaxis=dict(type="category", title="Metric", autorange="reversed"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _apply_white_chart_theme(fig)


def _scorecard_prepared_criteria():
    return {
        "beat_the_benchmark": [
            "Annualized Active Return",
            "Active Mean (Arithmetic)",
            "Active Mean (Geometric)",
            "Alpha",
            "Information Ratio",
            "Information Ratio (Arithmetic)",
            "Information Ratio (Geometric)",
            "Tracking Error",
            "Treynor Ratio",
        ],
        "minimize_risk": [
            "Annualized Volatility",
            "Semi-Deviation",
            "Total Risk",
            "Beta",
            "Historical VaR (5%)",
            "Gaussian VaR (5%)",
            "Modified Gaussian VaR (5%)",
            "Conditional VaR (5%)",
            "Expected Shortfall",
            "Ulcer Index",
        ],
        "maximize_diversification": [
            "Diversification Ratio",
            "Effective Holdings",
            "Effective Bets",
            "Total Risk",
            "Annualized Volatility",
        ],
        "limit_losses": [
            ("Max. Drawdown", "Max Drawdown"),
            "Ulcer Index",
            "Worst Day",
            "Average Loss",
            "Historical VaR (5%)",
            "Conditional VaR (5%)",
            "Expected Shortfall",
            "Risk of Ruin (q/p)^15",
            "Recovery Factor",
        ],
        "maximize_return": [
            "Total Return",
            "CAGR",
            "Annualized Mean Return",
            "Average Return",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Omega Ratio",
            "Gain-To-Pain Ratio",
            "Profit Factor",
        ],
        "maximize_risk_adjusted_return": [
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Omega Ratio",
            "Gain-To-Pain Ratio",
            "Serenity Index",
            "Ulcer Performance Index",
            "Treynor Ratio",
            "Information Ratio",
            "Recovery Factor",
        ],
        "balanced": [
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            ("Max. Drawdown", "Max Drawdown"),
            "Ulcer Index",
            "Information Ratio",
            "Diversification Ratio",
            "Effective Bets",
            "Annualized Volatility",
        ],
    }


def _scorecard_prepared_labels():
    return {
        "beat_the_benchmark": "Beat the Benchmark",
        "minimize_risk": "Minimize Risk",
        "maximize_diversification": "Maximize Diversification",
        "limit_losses": "Limit Losses",
        "maximize_return": "Maximize Return",
        "maximize_risk_adjusted_return": "Maximize Risk-Adjusted Return",
        "balanced": "Balanced",
    }


def _scorecard_hib_overrides():
    return {
        "Total Return": True,
        "CAGR": True,
        "Annualized Mean Return": True,
        "Average Return": True,
        "Best Day": True,
        "Worst Day": True,
        "Annualized Volatility": False,
        "Semi-Deviation": False,
        "Sharpe Ratio": True,
        "Sortino Ratio": True,
        "Ulcer Performance Index": True,
        "Serenity Index": True,
        "Gain-To-Pain Ratio": True,
        "Omega Ratio": True,
        "Calmar Ratio": True,
        "Recovery Factor": True,
        "Max. Drawdown": False,
        "Max Drawdown": False,
        "Ulcer Index": False,
        "Beta": False,
        "Alpha": True,
        "Annualized Active Return": True,
        "Information Ratio": True,
        "Treynor Ratio": True,
        "Historical VaR (5%)": False,
        "Gaussian VaR (5%)": False,
        "Modified Gaussian VaR (5%)": False,
        "Conditional VaR (5%)": False,
        "Tail Ratio": True,
        "Common Sense Ratio": True,
        "Win Rate": True,
        "Average Win": True,
        "Average Loss": True,
        "Payoff Ratio": True,
        "Profit Factor": True,
        "Profit Ratio": True,
        "CPC Index": True,
        "Risk of Ruin (q/p)^15": False,
        "Diversification Ratio": True,
        "Total Risk": False,
        "Expected Shortfall": False,
        "Active Mean (Arithmetic)": True,
        "Active Mean (Geometric)": True,
        "Tracking Error": False,
        "Information Ratio (Arithmetic)": True,
        "Information Ratio (Geometric)": True,
        "Effective Holdings": True,
        "Effective Bets": True,
    }


def _resolve_scorecard_metrics(stats_index, requested_metrics):
    resolved = []
    missing = 0
    seen = set()

    for item in requested_metrics:
        if isinstance(item, (list, tuple)):
            found = next((m for m in item if m in stats_index), None)
        else:
            found = item if item in stats_index else None

        if found is None:
            missing += 1
            continue

        if found not in seen:
            resolved.append(found)
            seen.add(found)

    return resolved, missing


def _build_scorecard_payload(
    stats,
    metrics_req,
    portfolios,
    kind_map,
    higher_is_better,
    transform_map,
    colorscale,
    tile_size,
    title,
    preset_label,
    max_metrics,
):
    metrics_keep, n_missing = _resolve_scorecard_metrics(stats.index, metrics_req)

    if len(metrics_keep) > max_metrics:
        raise ValueError(
            f"Preset '{preset_label}' resolves to {len(metrics_keep)} available metrics. "
            f"Use at most {max_metrics} for readability."
        )

    if len(portfolios) == 0 or len(metrics_keep) == 0:
        title_used = f"{title} — {preset_label} (nothing available to plot)"
        return {
            "x": [],
            "y": [],
            "text": [],
            "hovertemplate": [],
            "marker_color": [],
            "title": title_used,
            "height": 220,
            "metrics_keep": [],
            "tile_size": tile_size,
        }

    val = pd.DataFrame(index=metrics_keep, columns=portfolios, dtype=float)
    for m in metrics_keep:
        s = _stats_row(stats, m).reindex(portfolios)
        tr = transform_map.get(m, _default_transform(m))
        s = _apply_transform(s, tr)
        val.loc[m] = s.values

    good = pd.DataFrame(index=metrics_keep, columns=portfolios, dtype=float)
    for m in metrics_keep:
        hib = higher_is_better.get(m, _default_higher_is_better(m))
        x = pd.to_numeric(val.loc[m], errors="coerce")
        if x.notna().sum() <= 1:
            good.loc[m] = np.nan
            continue
        ranks = x.rank(pct=True, method="average")
        good.loc[m] = ranks if hib else (1.0 - ranks)

    xs, ys, texts, hovers, cs = [], [], [], [], []
    for m in metrics_keep:
        fmt = _valueformat(kind_map.get(m, _infer_kind(m)))
        for p in portfolios:
            v = val.loc[m, p]
            g = good.loc[m, p]

            xs.append(p)
            ys.append(m)
            texts.append(fmt(v))
            cs.append(g)
            hovers.append(
                f"<b>{p}</b><br>"
                f"{m}: {fmt(v)}<br>"
                f"Rank score: {'' if pd.isna(g) else f'{g:.2f}'}"
                "<extra></extra>"
            )

    colors = []
    for g in cs:
        if pd.isna(g):
            colors.append("rgba(180,180,180,0.5)")
        else:
            colors.append(sample_colorscale(colorscale, float(np.clip(g, 0, 1)))[0])

    title_used = f"{title} — {preset_label}"
    if n_missing > 0:
        title_used += f" (omitted {n_missing} unavailable metric(s))"

    return {
        "x": xs,
        "y": ys,
        "text": texts,
        "hovertemplate": hovers,
        "marker_color": colors,
        "title": title_used,
        "height": 140 + 55 * len(metrics_keep),
        "metrics_keep": metrics_keep,
        "tile_size": tile_size,
    }


def plot_scorecard_grid_v2(
    stats,
    portfolios=None,
    preset_map=None,
    initial_preset="balanced",
    kind_map=None,
    higher_is_better=None,
    transform_map=None,
    colorscale="RdYlGn",
    tile_size=46,
    title="Scorecard",
    max_metrics=10,
):
    if not isinstance(stats, pd.DataFrame):
        raise TypeError("stats must be a DataFrame (rows=metrics, cols=portfolios).")

    if portfolios is None:
        portfolios = list(stats.columns)
    else:
        portfolios = [p for p in list(portfolios) if p in stats.columns]

    if len(portfolios) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} (nothing available to plot)",
            height=220,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    preset_map = _scorecard_prepared_criteria() if preset_map is None else dict(preset_map)
    if len(preset_map) == 0:
        raise ValueError("preset_map must contain at least one preset.")

    labels = _scorecard_prepared_labels()
    kind_map = {} if kind_map is None else dict(kind_map)

    hib = _scorecard_hib_overrides()
    if higher_is_better is not None:
        hib.update(dict(higher_is_better))

    transform_map = {} if transform_map is None else dict(transform_map)

    preset_names = list(preset_map.keys())
    if initial_preset not in preset_map:
        initial_preset = preset_names[0]

    payloads = {}
    for preset_name, metrics_req in preset_map.items():
        preset_label = labels.get(preset_name, preset_name.replace("_", " ").title())
        payloads[preset_name] = _build_scorecard_payload(
            stats=stats,
            metrics_req=metrics_req,
            portfolios=portfolios,
            kind_map=kind_map,
            higher_is_better=hib,
            transform_map=transform_map,
            colorscale=colorscale,
            tile_size=tile_size,
            title=title,
            preset_label=preset_label,
            max_metrics=max_metrics,
        )

    init = payloads[initial_preset]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=init["x"],
            y=init["y"],
            mode="markers+text",
            text=init["text"],
            textposition="middle center",
            hovertemplate=init["hovertemplate"],
            marker=dict(
                symbol="square",
                size=tile_size,
                color=init["marker_color"],
                line=dict(width=1, color="rgba(0,0,0,0.25)"),
            ),
            showlegend=False,
        )
    )

    buttons = []
    for preset_name in preset_names:
        p = payloads[preset_name]
        preset_label = labels.get(preset_name, preset_name.replace("_", " ").title())

        buttons.append(
            dict(
                label=preset_label,
                method="update",
                args=[
                    {
                        "x": [p["x"]],
                        "y": [p["y"]],
                        "text": [p["text"]],
                        "hovertemplate": [p["hovertemplate"]],
                        "marker.color": [p["marker_color"]],
                    },
                    {
                        "title": {"text": p["title"]},
                        "height": p["height"],
                        "xaxis": _white_xaxis_dict(title="Portfolio", type="category", showgrid=False),
                        "yaxis": _white_yaxis_dict(
                            title="Metric",
                            type="category",
                            autorange="reversed",
                            showgrid=False,
                            categoryorder="array",
                            categoryarray=p["metrics_keep"],
                            zeroline=False,
                        ),
                    },
                ],
            )
        )

    active_idx = preset_names.index(initial_preset)

    fig.update_layout(
        title=init["title"],
        height=init["height"],
        margin=dict(l=20, r=20, t=95, b=20),
        xaxis=_white_xaxis_dict(title="Portfolio", type="category", showgrid=False),
        yaxis=_white_yaxis_dict(
            title="Metric",
            type="category",
            autorange="reversed",
            showgrid=False,
            categoryorder="array",
            categoryarray=init["metrics_keep"],
            zeroline=False,
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                active=active_idx,
                showactive=True,
                x=1.0,
                xanchor="left",
                y=1.16,
                yanchor="top",
            )
        ],
    )

    return _apply_white_chart_theme(fig)

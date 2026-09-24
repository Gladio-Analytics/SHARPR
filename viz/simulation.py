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


def resolve_return_column(portfolio_returns_all, name, case_insensitive=True):
    cols = list(portfolio_returns_all.columns)

    if name in cols:
        return name

    key = str(name)
    key_low = key.lower()

    def _norm(s):
        return str(s).strip()

    def _norm_low(s):
        return _norm(s).lower()

    if case_insensitive:
        for c in cols:
            if _norm_low(c) == _norm_low(key):
                return c

    for c in cols:
        if case_insensitive:
            if key_low in str(c).lower():
                return c
        else:
            if key in str(c):
                return c

    raise KeyError(f"Could not find a column matching '{name}'. Available columns: {cols}")


def simulate_gbm_paths_from_returns(
    portfolio_returns_all,
    column,
    n_paths=1000,
    horizon_days=252,
    s0=1.0,
    use_log_returns=True,
    ddof=1,
    seed=42
):
    col = resolve_return_column(portfolio_returns_all, column, case_insensitive=True)
    r = portfolio_returns_all[col].copy()
    r = pd.to_numeric(r, errors="coerce").dropna()

    if r.shape[0] < 30:
        raise ValueError(f"Not enough return observations for '{col}' (need >= 30, got {r.shape[0]}).")

    if use_log_returns:
        x = np.log1p(r)
        mu = float(x.mean())
        sigma = float(x.std(ddof=ddof))
        drift = mu
    else:
        mu = float(r.mean())
        sigma = float(r.std(ddof=ddof))
        drift = mu - 0.5 * sigma * sigma

    if not (np.isfinite(mu) and np.isfinite(sigma)) or sigma <= 0:
        raise ValueError(f"Bad estimated params for '{col}': mu={mu}, sigma={sigma}")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(int(horizon_days), int(n_paths)))

    log_increments = drift + sigma * Z
    log_paths = np.vstack([np.zeros((1, int(n_paths))), np.cumsum(log_increments, axis=0)])

    S = float(s0) * np.exp(log_paths)

    idx = pd.RangeIndex(start=0, stop=int(horizon_days) + 1, step=1, name="t")
    cols = [f"path_{i+1}" for i in range(int(n_paths))]
    paths = pd.DataFrame(S, index=idx, columns=cols)

    params = {
        "column": col,
        "n_paths": int(n_paths),
        "horizon_days": int(horizon_days),
        "s0": float(s0),
        "use_log_returns": bool(use_log_returns),
        "mu_est": float(mu),
        "sigma_est": float(sigma),
        "drift_used": float(drift)
    }

    return paths, params


def simulate_and_plot_two_gbm(
    portfolio_returns_all,
    sim_cols,
    n_paths=1000,
    horizon_days=252,
    seed=7,
    title="GBM Simulation (two portfolios)",
    height=780,
    legend_x=1.30,
    right_margin=280,
    sim_line_alpha=0.12,
    sim_line_width=1.0,
    mean_line_width=2.4,
    density_bins=80,
    plot_paths=120,
    use_last_for_sim=None,
    plot_last_t=None,
    style_registry=None
):
    df = portfolio_returns_all.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(how="all")

    cols_all = [c for c in df.columns]
    sim_cols = list(sim_cols)

    if len(sim_cols) == 0 or len(sim_cols) > 2:
        raise ValueError("sim_cols must contain 1 or 2 column names")

    for c in sim_cols:
        if c not in df.columns:
            raise ValueError(f"'{c}' not found in portfolio_returns_all columns")

    abs_med = float(np.nanmedian(np.abs(df[sim_cols].to_numpy(dtype=float))))
    if np.isfinite(abs_med) and abs_med > 0.2:
        raise ValueError("Returns look like percent units (e.g., 1.2 for 1.2%). Use decimals (0.012).")

    wealth_all = (1.0 + df.fillna(0.0)).cumprod()
    wealth_last = wealth_all.iloc[-1]

    last_date = wealth_all.index[-1]

    if plot_last_t is None:
        wealth_plot = wealth_all
    else:
        t = int(plot_last_t)
        wealth_plot = wealth_all.iloc[-t:] if t > 0 else wealth_all.iloc[0:0]

    def _to_rgba(color, a):
        s = str(color).strip()
        if s.startswith("rgba(") and s.endswith(")"):
            inside = s[5:-1].split(",")
            if len(inside) >= 3:
                r = int(float(inside[0])); g = int(float(inside[1])); b = int(float(inside[2]))
                return f"rgba({r},{g},{b},{float(a)})"
        if s.startswith("rgb(") and s.endswith(")"):
            inside = s[4:-1].split(",")
            if len(inside) >= 3:
                r = int(float(inside[0])); g = int(float(inside[1])); b = int(float(inside[2]))
                return f"rgba({r},{g},{b},{float(a)})"
        if s.startswith("#") and len(s) == 7:
            r = int(s[1:3], 16); g = int(s[3:5], 16); b = int(s[5:7], 16)
            return f"rgba({r},{g},{b},{float(a)})"
        return f"rgba(60,60,60,{float(a)})"

    def _kde_1d(x, n_grid=240):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return None, None
        x_sorted = np.sort(x)
        x_min = float(x_sorted[0]); x_max = float(x_sorted[-1])
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            return None, None
        grid = np.linspace(x_min, x_max, int(n_grid))
        std = float(np.std(x, ddof=1))
        if not np.isfinite(std) or std <= 0:
            return None, None
        h = 1.06 * std * (x.size ** (-1/5))
        if not np.isfinite(h) or h <= 0:
            return None, None
        diffs = (grid[:, None] - x[None, :]) / h
        dens = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (h * np.sqrt(2*np.pi))
        return grid, dens

    def _simulate_one(col_name, rng):
        r = df[col_name].dropna().astype(float)
        if use_last_for_sim is not None:
            k = int(use_last_for_sim)
            if k > 0 and len(r) > k:
                r = r.iloc[-k:]

        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        drift = mu - 0.5 * sigma * sigma

        if not (np.isfinite(drift) and np.isfinite(sigma)) or sigma <= 0:
            raise ValueError(f"Bad estimated params for '{col_name}': drift={drift}, sigma={sigma}")

        s0 = float(wealth_last[col_name])
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError(f"Bad starting wealth for '{col_name}': {s0}")

        Z = rng.standard_normal(size=(int(horizon_days), int(n_paths)))
        log_incr = drift + sigma * Z
        log_paths = np.vstack([np.zeros((1, int(n_paths))), np.cumsum(log_incr, axis=0)])
        S = s0 * np.exp(log_paths)

        idx_fwd = pd.bdate_range(start=last_date, periods=int(horizon_days) + 1)
        paths = pd.DataFrame(S, index=idx_fwd, columns=[f"path_{i+1}" for i in range(int(n_paths))])
        terminal = paths.iloc[-1].to_numpy(dtype=float)

        meta = {"col": col_name, "s0": s0, "drift": drift, "sigma": sigma}
        return paths, terminal, meta

    rng = np.random.default_rng(int(seed))
    sims = {}
    terminals = {}
    metas = {}

    for c in sim_cols:
        p, t, m = _simulate_one(c, rng)
        sims[c] = p
        terminals[c] = t
        metas[c] = m

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        subplot_titles=("Cumulative Wealth & Simulated Paths", "Simulated Terminal Wealth Density"),
        vertical_spacing=0.18
    )

    for c in cols_all:
        st = style_for_series(c, style_registry)
        fig.add_trace(
            go.Scatter(
                x=wealth_plot.index,
                y=wealth_plot[c].values,
                mode="lines",
                name=str(c),
                line=dict(color=st["color"], dash=st["dash"], width=st["width"]),
                hoverinfo="skip",
                showlegend=True
            ),
            row=1,
            col=1
        )

    for c in sim_cols:
        st = style_for_series(c, style_registry)
        base_color = st["color"]
        path_color = _to_rgba(base_color, sim_line_alpha)

        P = sims[c]
        k = int(min(max(plot_paths, 0), P.shape[1]))

        if k > 0:
            for j in range(k):
                fig.add_trace(
                    go.Scatter(
                        x=P.index,
                        y=P.iloc[:, j].values,
                        mode="lines",
                        line=dict(color=path_color, width=float(sim_line_width)),
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    row=1,
                    col=1
                )

        mean_path = P.mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=P.index,
                y=mean_path.values,
                mode="lines",
                name=str(c),
                line=dict(color=base_color, width=float(mean_line_width), dash="dot"),
                hoverinfo="skip",
                showlegend=False
            ),
            row=1,
            col=1
        )

        term = np.asarray(terminals[c], dtype=float)
        term = term[np.isfinite(term)]
        if term.size > 0:
            fig.add_trace(
                go.Histogram(
                    x=term,
                    nbinsx=int(density_bins),
                    histnorm="probability density",
                    name=str(c),
                    marker=dict(color=_to_rgba(base_color, 0.45)),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=2,
                col=1
            )

            gx, gd = _kde_1d(term, n_grid=240)
            if gx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=gx,
                        y=gd,
                        mode="lines",
                        name=str(c),
                        line=dict(color=base_color, width=3),
                        hoverinfo="skip",
                        showlegend=False
                    ),
                    row=2,
                    col=1
                )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=int(height),
        barmode="overlay",
        legend=dict(
            x=float(legend_x),
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=70, r=int(right_margin), t=90, b=60)
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Wealth Index (start=1)", row=1, col=1)
    fig.update_xaxes(title_text="Terminal wealth", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)

    out = {
        "wealth_all": wealth_all,
        "wealth_last": wealth_last,
        "sims": sims,
        "terminals": terminals,
        "metas": metas,
        "last_date": last_date
    }

    n_wealth = len(cols_all)
    for i, tr in enumerate(fig.data):
        tr.showlegend = (i < n_wealth)

    return _apply_white_chart_theme(fig), out

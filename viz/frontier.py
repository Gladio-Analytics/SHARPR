import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cvxpy as cp
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


def plot_efficient_frontier_v2_2(
    asset_returns,
    portfolio_returns_all=None,
    grid="risk",
    n_points=140,
    annualize=True,
    annualization=252,
    long_only=False,
    bound=None,
    leverage_limit=2.0,
    ridge=1e-10,
    title="Efficient Frontier",
    show_asset_labels=True,
    show_portfolio_labels=True,
    height=620,
    width=None,
    solver_preference=("OSQP", "ECOS", "SCS")
):
    X = asset_returns.copy()
    X.index = pd.to_datetime(X.index)
    X = X.sort_index()
    X = X.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    tickers = list(X.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("Need at least 2 assets")

    mu_d = X.mean().values.astype(float)
    C_d = X.cov().values.astype(float)
    C_d = 0.5 * (C_d + C_d.T) + float(ridge) * np.eye(n)

    grid = str(grid).lower().strip()
    if grid not in ("risk", "return"):
        raise ValueError("grid must be 'risk' or 'return'")

    def _constraints(w):
        cons = [cp.sum(w) == 1]
        if long_only:
            cons.append(w >= 0)
        if bound is not None:
            b = float(bound)
            cons.append(w <= b)
            if not long_only:
                cons.append(w >= -b)
        if (not long_only) and (leverage_limit is not None):
            cons.append(cp.norm1(w) <= float(leverage_limit))
        return cons

    def _solve(prob):
        last_err = None
        for s in solver_preference:
            try:
                prob.solve(solver=getattr(cp, s), verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return True
            except Exception as e:
                last_err = e
        raise ValueError(f"CVXPY solve failed: {last_err}")

    w = cp.Variable(n)
    prob_gmv = cp.Problem(cp.Minimize(cp.quad_form(w, C_d)), _constraints(w))
    _solve(prob_gmv)
    if w.value is None:
        raise ValueError("GMV solve returned no weights")
    w_gmv = np.asarray(w.value, dtype=float).reshape(-1)
    mu_gmv = float(w_gmv @ mu_d)
    sig_gmv = float(np.sqrt(max(w_gmv @ C_d @ w_gmv, 0.0)))

    w2 = cp.Variable(n)
    prob_max = cp.Problem(cp.Maximize(mu_d @ w2), _constraints(w2))
    _solve(prob_max)
    if w2.value is None:
        raise ValueError("Max-return solve returned no weights")
    w_max = np.asarray(w2.value, dtype=float).reshape(-1)
    mu_max = float(w_max @ mu_d)
    sig_at_max = float(np.sqrt(max(w_max @ C_d @ w_max, 0.0)))

    if not (np.isfinite(mu_gmv) and np.isfinite(mu_max) and np.isfinite(sig_gmv) and np.isfinite(sig_at_max)):
        raise ValueError("Frontier endpoints failed")

    pts = []

    if grid == "risk":
        if sig_at_max <= sig_gmv + 1e-12:
            raise ValueError("Risk range degenerate. Try grid='return' or relax constraints.")
        sig_targets = np.linspace(sig_gmv, sig_at_max, int(n_points))

        for sig_t in sig_targets:
            wv = cp.Variable(n)
            cons = _constraints(wv) + [cp.quad_form(wv, C_d) <= float(sig_t ** 2)]
            prob = cp.Problem(cp.Maximize(mu_d @ wv), cons)
            _solve(prob)
            if wv.value is None:
                continue
            ww = np.asarray(wv.value, dtype=float).reshape(-1)
            if not np.all(np.isfinite(ww)):
                continue
            ret_d = float(ww @ mu_d)
            vol_d = float(np.sqrt(max(ww @ C_d @ ww, 0.0)))
            pts.append((ret_d, vol_d))
    else:
        if mu_max <= mu_gmv + 1e-12:
            raise ValueError("Return range degenerate. Try grid='risk' or relax constraints.")
        ret_targets = np.linspace(mu_gmv, mu_max, int(n_points))

        for tr in ret_targets:
            wv = cp.Variable(n)
            cons = _constraints(wv) + [mu_d @ wv >= float(tr)]
            prob = cp.Problem(cp.Minimize(cp.quad_form(wv, C_d)), cons)
            _solve(prob)
            if wv.value is None:
                continue
            ww = np.asarray(wv.value, dtype=float).reshape(-1)
            if not np.all(np.isfinite(ww)):
                continue
            ret_d = float(ww @ mu_d)
            vol_d = float(np.sqrt(max(ww @ C_d @ ww, 0.0)))
            pts.append((ret_d, vol_d))

    if len(pts) < 5:
        raise ValueError("Frontier construction failed (too few points)")

    fr = pd.DataFrame(pts, columns=["ret_d", "vol_d"]).drop_duplicates()
    fr = fr.sort_values("vol_d").reset_index(drop=True)

    keep = []
    best_ret = -np.inf
    for i in range(len(fr)):
        r = float(fr.loc[i, "ret_d"])
        if r > best_ret + 1e-12:
            keep.append(i)
            best_ret = r
    fr = fr.loc[keep].reset_index(drop=True)

    xs, ys = [], []
    last_x = -np.inf
    for _, row in fr.iterrows():
        x = float(row["vol_d"])
        y = float(row["ret_d"])
        if x > last_x + 1e-12:
            xs.append(x)
            ys.append(y)
            last_x = x

    if len(xs) < 3:
        raise ValueError("Filtered frontier too small. Try relaxing constraints or increasing n_points.")

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    if annualize:
        fr_x = xs * np.sqrt(annualization)
        fr_y = ys * annualization
        asset_mu = X.mean() * annualization
        asset_vol = X.std(ddof=0) * np.sqrt(annualization)
        xlab = "Volatility (annualized)"
        ylab = "Return (annualized)"
    else:
        fr_x = xs
        fr_y = ys
        asset_mu = X.mean()
        asset_vol = X.std(ddof=0)
        xlab = "Volatility (daily)"
        ylab = "Return (daily)"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fr_x, y=fr_y,
            mode="lines",
            line=dict(color="blue", width=3),
            line_shape="linear",
            hovertemplate="EF<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=asset_vol.values,
            y=asset_mu.values,
            mode="markers+text" if show_asset_labels else "markers",
            text=tickers,
            textposition="top center",
            marker=dict(color="blue", size=10, symbol="circle"),
            hovertemplate="%{text}<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
            showlegend=False
        )
    )

    if portfolio_returns_all is not None:
        P = portfolio_returns_all.copy()
        P.index = pd.to_datetime(P.index)
        P = P.sort_index()

        cols = []
        for c in P.columns:
            name = str(c)
            low = name.lower()
            if ("risk-free" in low) or ("risk free" in low):
                continue
            cols.append(c)

        if cols:
            if annualize:
                mu_p = P[cols].mean() * annualization
                vol_p = P[cols].std(ddof=0) * np.sqrt(annualization)
            else:
                mu_p = P[cols].mean()
                vol_p = P[cols].std(ddof=0)

            x_red, y_red, txt_red = [], [], []
            x_blk, y_blk, txt_blk = [], [], []

            for c in cols:
                name = str(c)
                low = name.lower()
                xv = float(vol_p[c])
                yv = float(mu_p[c])
                is_black = ("benchmark" in low) or (low == "current portfolio") or (low == "custom portfolio")
                if is_black:
                    x_blk.append(xv); y_blk.append(yv); txt_blk.append(name)
                else:
                    x_red.append(xv); y_red.append(yv); txt_red.append(name)

            if x_red:
                fig.add_trace(
                    go.Scatter(
                        x=x_red, y=y_red,
                        mode="markers+text" if show_portfolio_labels else "markers",
                        text=txt_red,
                        textposition="middle right",
                        marker=dict(color="red", size=10, symbol="circle"),
                        hovertemplate="%{text}<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
                        showlegend=False
                    )
                )

            if x_blk:
                fig.add_trace(
                    go.Scatter(
                        x=x_blk, y=y_blk,
                        mode="markers+text" if show_portfolio_labels else "markers",
                        text=txt_blk,
                        textposition="middle right",
                        marker=dict(color="black", size=10, symbol="circle"),
                        hovertemplate="%{text}<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
                        showlegend=False
                    )
                )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(l=60, r=60, t=70, b=60),
        xaxis_title=xlab,
        yaxis_title=ylab
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _apply_white_chart_theme(fig)

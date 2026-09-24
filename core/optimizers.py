import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog


def _as_1d_array(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _feasible_bound_check(n, bound):
    if bound <= 0:
        raise ValueError("bound must be > 0")
    if n * bound + 1e-12 < 1.0:
        raise ValueError("Infeasible constraints: n * bound < 1. Increase bound or number of assets.")


def _fallback_feasible_weights(n, bound):
    _feasible_bound_check(n, bound)
    w = np.zeros(n, dtype=float)
    rem = 1.0
    for i in range(n):
        take = min(bound, rem)
        w[i] = take
        rem -= take
        if rem <= 1e-15:
            break
    w = w / w.sum()
    return w


def _to_weights_df(w, tickers, bound=1.0):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    n = len(w)

    if s <= 0:
        w = np.repeat(1.0 / n, n)
    else:
        w = w / s

    if bound is not None and float(w.max()) > bound + 1e-10:
        w = np.clip(w, 0.0, bound)
        s2 = w.sum()
        if s2 <= 0:
            w = np.repeat(1.0 / n, n)
        else:
            w = w / s2

    out = pd.DataFrame({"weight": w}, index=list(tickers))
    out.index.name = "Ticker"
    return out


def _init_guess(n, tickers, w0=None, bound=1.0):
    _feasible_bound_check(n, bound)
    if w0 is None:
        x = np.repeat(1.0 / n, n)
        if x.max() > bound + 1e-12:
            x = _fallback_feasible_weights(n, bound)
        return x

    if isinstance(w0, pd.DataFrame):
        x = w0.iloc[:, 0].reindex(tickers).fillna(0.0).values
    elif isinstance(w0, pd.Series):
        x = w0.reindex(tickers).fillna(0.0).values
    elif isinstance(w0, dict):
        x = pd.Series(w0).reindex(tickers).fillna(0.0).values
    else:
        x = _as_1d_array(w0)

    x = np.clip(x, 0.0, bound)
    s = float(x.sum())
    if s <= 0:
        return _fallback_feasible_weights(n, bound)
    x = x / s
    if x.max() > bound + 1e-10:
        x = np.clip(x, 0.0, bound)
        s2 = float(x.sum())
        if s2 <= 0:
            return _fallback_feasible_weights(n, bound)
        x = x / s2
    return x


def portfolio_return(w, er):
    return float(np.dot(w, er))


def portfolio_var(w, cov):
    return float(np.dot(w, np.dot(cov, w)))


def portfolio_vol(w, cov):
    v = portfolio_var(w, cov)
    return float(np.sqrt(max(v, 0.0)))


def div_ratio(w, cov):
    vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = portfolio_vol(w, cov)
    if denom <= 0:
        return -np.inf
    return float(np.dot(w, vols) / denom)


def risk_contribution_rcpct(w, cov):
    w = _as_1d_array(w)
    var = portfolio_var(w, cov)
    if var <= 0:
        return np.full_like(w, np.nan, dtype=float)
    sw = np.dot(cov, w)
    return (w * sw) / var


def marginal_risk_contribution_vol(w, cov):
    w = _as_1d_array(w)
    sig = portfolio_vol(w, cov)
    if sig <= 0:
        return np.full_like(w, np.nan, dtype=float)
    sw = np.dot(cov, w)
    return sw / sig


def max_div_port(cov, w0=None, bound=1.0):
    tickers = list(cov.columns)
    C = cov.reindex(index=tickers, columns=tickers).values
    n = C.shape[0]
    _feasible_bound_check(n, bound)
    x0 = _init_guess(n, tickers, w0, bound=bound)
    bnds = [(0.0, bound)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        return -div_ratio(w, C)

    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"disp": False})
    if not res.success:
        raise ValueError(res.message)
    return _to_weights_df(res.x, tickers, bound=bound)


def msr(rfr, er, cov, w0=None, bound=1.0, rfr_mode="mean", rfr_index=None):
    tickers = list(cov.columns)
    C = cov.reindex(index=tickers, columns=tickers).values
    n = C.shape[0]
    _feasible_bound_check(n, bound)

    mu = er.reindex(tickers).values if isinstance(er, pd.Series) else _as_1d_array(er)
    if mu.size != n:
        raise ValueError("msr: er length does not match cov dimension")

    rf = _rfr_to_scalar_daily(rfr, index=rfr_index, mode=rfr_mode)

    x0 = _init_guess(n, tickers, w0, bound=bound)
    bnds = [(0.0, bound)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        sig = portfolio_vol(w, C)
        if sig <= 0:
            return 1e9
        return -((portfolio_return(w, mu) - rf) / sig)

    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"disp": False})
    if not res.success:
        raise ValueError(res.message)
    return _to_weights_df(res.x, tickers, bound=bound)


def gmv(cov, w0=None, bound=1.0, ridge=1e-10, maxiter=5000, ftol=1e-12):
    tickers = list(cov.columns)
    C = cov.reindex(index=tickers, columns=tickers).values.astype(float)
    n = C.shape[0]
    _feasible_bound_check(n, bound)

    C = 0.5 * (C + C.T)
    if ridge is not None and ridge > 0:
        C = C + ridge * np.eye(n)

    if w0 is None:
        ones = np.ones(n)
        try:
            w_cf = np.linalg.solve(C, ones)
            w_cf = np.clip(w_cf, 0.0, bound)
            if w_cf.sum() > 0:
                x0 = w_cf / w_cf.sum()
            else:
                x0 = _fallback_feasible_weights(n, bound)
        except Exception:
            x0 = _fallback_feasible_weights(n, bound)
    else:
        x0 = _init_guess(n, tickers, w0, bound=bound)

    bnds = [(0.0, bound)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        return float(w @ C @ w)

    def jac(w):
        return 2.0 * (C @ w)

    res = minimize(
        objective, x0, jac=jac, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"disp": False, "maxiter": int(maxiter), "ftol": float(ftol)}
    )
    if not res.success:
        raise ValueError(res.message)

    return _to_weights_df(res.x, tickers, bound=bound)


def target_risk_contributions(target_risk, cov, w0=None, bound=1.0):
    tickers = list(cov.columns)
    C = cov.reindex(index=tickers, columns=tickers).values
    n = C.shape[0]
    _feasible_bound_check(n, bound)

    tr = _as_1d_array(target_risk)
    if tr.size != n:
        raise ValueError("target_risk_contributions: target_risk length mismatch")
    s = float(tr.sum())
    if s <= 0:
        raise ValueError("target_risk_contributions: target_risk sum must be > 0")
    tr = tr / s

    x0 = _init_guess(n, tickers, w0, bound=bound)
    bnds = [(0.0, bound)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        rc_pct = risk_contribution_rcpct(w, C)
        if np.any(np.isnan(rc_pct)):
            return 1e9
        return float(np.sum((rc_pct - tr) ** 2))

    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"disp": False})
    if not res.success:
        raise ValueError(res.message)
    return _to_weights_df(res.x, tickers, bound=bound)


def equal_risk_contributions(cov, w0=None, bound=1.0):
    n = cov.shape[0]
    return target_risk_contributions(np.repeat(1.0 / n, n), cov, w0=w0, bound=bound)


def cvar_find(asset_returns, beta=0.95, bound=1.0):
    if not (0.0 < beta < 1.0):
        raise ValueError("cvar_find: beta must be in (0,1)")
    X = asset_returns.copy()
    X = X.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X.empty:
        raise ValueError("cvar_find: no data after dropna(how='any')")

    tickers = list(X.columns)
    R = X.values
    q, n = R.shape
    _feasible_bound_check(n, bound)

    m = 1.0 / (q * (1.0 - beta))

    c = np.r_[1.0, np.repeat(m, q), np.zeros(n)]

    A_ub = np.zeros((q, 1 + q + n))
    b_ub = np.zeros(q)
    for i in range(q):
        A_ub[i, 0] = -1.0
        A_ub[i, 1 + i] = -1.0
        A_ub[i, 1 + q:] = -R[i, :]

    A_eq = np.zeros((1, 1 + q + n))
    A_eq[0, 1 + q:] = 1.0
    b_eq = np.array([1.0])

    lp_bounds = [(None, None)] + [(0.0, None)] * q + [(0.0, bound)] * n

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=lp_bounds, method="highs")
    if not res.success:
        raise ValueError(res.message)

    return _to_weights_df(res.x[1 + q:], tickers, bound=bound)


def calculate_cumulative_returns_rebalanced(weights, asset_returns):
    w = _as_1d_array(weights)
    rp = asset_returns.values @ w
    wealth = np.cumprod(1.0 + rp)
    return pd.Series(wealth, index=asset_returns.index)


def max_dd_from_wealth(wealth):
    x = _as_1d_array(wealth)
    peak = np.maximum.accumulate(x)
    dd = x / peak - 1.0
    return float(-dd.min())


def minimize_max_drawdown(asset_returns, w0=None, bound=1.0):
    X = asset_returns.copy()
    X = X.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X.empty:
        raise ValueError("minimize_max_drawdown: no data after dropna(how='any')")

    tickers = list(X.columns)
    n = X.shape[1]
    _feasible_bound_check(n, bound)

    x0 = _init_guess(n, tickers, w0, bound=bound)
    bnds = [(0.0, bound)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def objective(w):
        wealth = calculate_cumulative_returns_rebalanced(w, X)
        return max_dd_from_wealth(wealth)

    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"disp": False})
    if not res.success:
        raise ValueError(res.message)

    return _to_weights_df(res.x, tickers, bound=bound)


def run_all_optimizers(asset_returns, cov_matrix=None, sample_means=None, rfr=0.0, beta=0.95, bound=1.0, w0=None, rfr_mode="mean"):
    X = asset_returns.copy()
    X = X.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if X.empty:
        raise ValueError("run_all_optimizers: asset_returns empty after dropna(how='any')")

    tickers = list(X.columns)
    n = len(tickers)
    _feasible_bound_check(n, bound)

    if cov_matrix is None:
        cov_matrix = X.cov()
    else:
        cov_matrix = cov_matrix.reindex(index=tickers, columns=tickers)

    if sample_means is None:
        sample_means = X.mean()
    else:
        if isinstance(sample_means, pd.Series):
            sample_means = sample_means.reindex(tickers)
        else:
            sample_means = pd.Series(sample_means, index=tickers)

    rf_scalar = _rfr_to_scalar_daily(rfr, index=X.index, mode=rfr_mode)

    w_max_div = max_div_port(cov_matrix, w0=w0, bound=bound)
    w_msr = msr(rfr=rf_scalar, er=sample_means, cov=cov_matrix, w0=w0, bound=bound)
    w_gmv = gmv(cov_matrix, w0=w0, bound=bound)
    w_rp = equal_risk_contributions(cov_matrix, w0=w0, bound=bound)
    w_es = cvar_find(X, beta=beta, bound=bound)
    w_mdd = minimize_max_drawdown(X, w0=w0, bound=bound)

    weights_all = pd.concat(
        [
            w_max_div.rename(columns={"weight": "Max. Diversification"}),
            w_msr.rename(columns={"weight": "Max. Sharpe Ratio"}),
            w_gmv.rename(columns={"weight": "Min. Variance"}),
            w_rp.rename(columns={"weight": "Risk Parity"}),
            w_es.rename(columns={"weight": "Min. Expected Shortfall"}),
            w_mdd.rename(columns={"weight": "Min. Drawdown"}),
        ],
        axis=1
    ).reindex(index=tickers)

    weights_all.index.name = "Ticker"
    weights_all = weights_all.div(weights_all.sum(axis=0), axis=1)

    return weights_all, {"rfr_scalar_daily": rf_scalar, "asset_returns_clean": X, "cov_matrix": cov_matrix, "sample_means": sample_means}


def _rfr_to_scalar_daily(rfr, index=None, mode="mean"):
    if rfr is None:
        return 0.0
    if isinstance(rfr, (int, float, np.floating)):
        return float(rfr)
    if isinstance(rfr, pd.DataFrame):
        if rfr.shape[1] != 1:
            raise ValueError("rfr DataFrame must be 1-column")
        s = rfr.iloc[:, 0]
    elif isinstance(rfr, pd.Series):
        s = rfr
    else:
        return float(rfr)

    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if index is not None:
        idx = pd.to_datetime(index)
        s = s.reindex(idx).ffill()

    v = np.asarray(s.values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0
    if mode == "last":
        return float(v[-1])
    return float(v.mean())

import numpy as np
import pandas as pd
from .optimizers import _rfr_to_scalar_daily, _as_1d_array


def expected_shortfall_loss_table(weights_all, asset_returns, beta=0.95):
    P = portfolio_returns_from_weights_all(weights_all, asset_returns)
    out = {}
    for c in P.columns:
        x = _as_1d_array(P[c].values)
        x = x[~np.isnan(x)]
        if x.size == 0:
            out[c] = np.nan
            continue
        losses = -x
        k = max(1, int(np.floor((1.0 - beta) * losses.size)))
        out[c] = float(np.sort(losses)[-k:].mean())
    return pd.DataFrame({"Expected Shortfall (loss)": pd.Series(out)})


def component_risk_contribution_pct(weights_all, cov_matrix):
    tickers = list(cov_matrix.columns)
    W = weights_all.reindex(tickers).fillna(0.0).clip(lower=0.0)
    W = W.div(W.sum(axis=0), axis=1)

    S = cov_matrix.reindex(index=tickers, columns=tickers).values

    out = pd.DataFrame(index=tickers)
    for col in W.columns:
        w = W[col].values
        var = float(w @ S @ w)
        if var <= 0:
            out[col] = np.nan
            continue
        m = S @ w
        out[col] = (w * m) / var
    out.index.name = "Ticker"
    return out


def marginal_risk_contribution(weights_all, cov_matrix):
    tickers = list(cov_matrix.columns)
    W = weights_all.reindex(tickers).fillna(0.0).clip(lower=0.0)
    W = W.div(W.sum(axis=0), axis=1)

    S = cov_matrix.reindex(index=tickers, columns=tickers).values

    out = pd.DataFrame(index=tickers)
    for col in W.columns:
        w = W[col].values
        sig = np.sqrt(float(w @ S @ w))
        if sig <= 0:
            out[col] = np.nan
            continue
        Sw = S @ w
        out[col] = Sw / sig
    out.index.name = "Ticker"
    return out


def portfolio_returns_from_weights_all(weights_all, asset_returns):
    tickers = list(asset_returns.columns)

    def _exclude_col(c):
        name = str(c)
        low = name.lower()
        if name in ("Current Portfolio", "Custom Portfolio"):
            return True
        if ("risk-free rate" in low) or ("risk free rate" in low):
            return True
        if "benchmark" in low:
            return True
        return False

    cols_keep = [c for c in weights_all.columns if not _exclude_col(c)]
    if len(cols_keep) == 0:
        raise ValueError("No portfolio columns left after excluding Current/Custom/Risk-Free/Benchmark.")

    W = weights_all[cols_keep].reindex(tickers).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    R = asset_returns.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    P = R.to_numpy() @ W.to_numpy()
    return pd.DataFrame(P, index=R.index, columns=list(W.columns))


def add_reference_returns(
    portfolio_returns,
    current_portfolio_return=None,
    custom_portfolio_return=None,
    benchmark=None,
    risk_free_rate=None
):
    out = portfolio_returns.copy()

    def _as_1col_df(x, default_name):
        if x is None:
            return None
        if isinstance(x, pd.Series):
            s = x.copy()
            name = s.name if (s.name is not None and str(s.name).strip() != "") else default_name
            s.name = name
            return s.to_frame()
        if isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError(f"{default_name} must be a Series or 1-column DataFrame")
            y = x.copy()
            col = y.columns[0]
            if col is None or str(col).strip() == "":
                y.columns = [default_name]
            return y
        raise ValueError(f"{default_name} must be a Series or a DataFrame")

    def _add(onecol_df):
        nonlocal out
        if onecol_df is None:
            return
        col = onecol_df.columns[0]
        if col in out.columns:
            raise ValueError(f"Column already exists in portfolio_returns: '{col}'")
        out = out.join(onecol_df.reindex(out.index), how="left")

    _add(_as_1col_df(current_portfolio_return, "Current Portfolio"))
    _add(_as_1col_df(custom_portfolio_return, "Custom Portfolio"))
    _add(_as_1col_df(benchmark, "Benchmark"))
    _add(_as_1col_df(risk_free_rate, "Risk-Free Rate"))

    return out


def risk_contribution_pct_all(weights_all, cov_matrix):
    tickers = list(cov_matrix.columns)
    C = cov_matrix.reindex(index=tickers, columns=tickers).values.astype(float)
    C = 0.5 * (C + C.T)

    W = weights_all.reindex(index=tickers).fillna(0.0).clip(lower=0.0)
    W = W.div(W.sum(axis=0), axis=1)

    out = pd.DataFrame(index=tickers, columns=W.columns, dtype=float)

    for col in W.columns:
        w = W[col].values.astype(float)
        var = float(w @ C @ w)
        if var <= 0:
            out[col] = np.nan
            continue
        Sw = C @ w
        out[col] = (w * Sw) / var

    out.index.name = "Ticker"
    return out


def risk_contribution_dispersion_all(rc_pct_all):
    """Population std dev of each portfolio's per-ticker risk contribution (from
    risk_contribution_pct_all). 0 = risk spread perfectly evenly across tickers
    (true risk parity); higher = risk concentrated in fewer tickers."""
    return rc_pct_all.std(axis=0, ddof=0).rename("Risk Contribution Dispersion")


def sharpe_contribution_pct_all(weights_all, er, cov_matrix, rfr=0.0, rfr_mode="mean", rfr_index=None, mode="marginal"):
    tickers = list(cov_matrix.columns)
    C = cov_matrix.reindex(index=tickers, columns=tickers).values.astype(float)
    C = 0.5 * (C + C.T)

    if isinstance(er, pd.Series):
        mu = er.reindex(tickers).values.astype(float)
    else:
        mu = np.asarray(er, dtype=float).reshape(-1)
        if mu.size != len(tickers):
            raise ValueError("er length mismatch with cov_matrix")

    rf = _rfr_to_scalar_daily(rfr, index=rfr_index, mode=rfr_mode)

    W = weights_all.reindex(index=tickers).fillna(0.0).clip(lower=0.0)
    W = W.div(W.sum(axis=0), axis=1)

    out = pd.DataFrame(index=tickers, columns=W.columns, dtype=float)

    for col in W.columns:
        w = W[col].values.astype(float)

        port_mu = float(w @ mu)
        port_ex = port_mu - rf
        var = float(w @ C @ w)
        sig = float(np.sqrt(max(var, 0.0)))
        if sig <= 0:
            out[col] = np.nan
            continue

        sharpe = port_ex / sig
        if not np.isfinite(sharpe) or abs(sharpe) <= 1e-15:
            out[col] = np.nan
            continue

        if mode == "heuristic":
            numer = w * (mu - rf)
            denom = float(numer.sum())
            if abs(denom) <= 1e-15:
                out[col] = np.nan
            else:
                out[col] = numer / denom
            continue

        if mode != "marginal":
            raise ValueError("mode must be 'marginal' or 'heuristic'")

        Sw = C @ w
        grad = (mu - rf) / sig - (port_ex / (sig ** 3)) * Sw

        contrib = w * grad
        total = float(contrib.sum())
        if abs(total) <= 1e-15:
            out[col] = np.nan
        else:
            out[col] = contrib / total

    out.index.name = "Ticker"
    return out


def get_wealth_level(portfolio_returns_all, start_value=1.0):
    r = portfolio_returns_all.copy().apply(pd.to_numeric, errors="coerce")
    r = r.dropna(how="all")
    if r.shape[0] == 0:
        raise ValueError("portfolio_returns_all has no usable return rows.")
    wealth_last = (1.0 + r).cumprod().iloc[-1] * float(start_value)
    return wealth_last

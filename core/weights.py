import numpy as np
import pandas as pd


def create_random_weights(tickers, seed=None, as_dict=True):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(tickers)))
    s = pd.Series(w, index=tickers, name="weight")
    return s.to_dict() if as_dict else s


def create_current_portfolio_weights(starting_weights, returns, portfolio_formation_date=None):
    if starting_weights is None:
        return None

    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()

    if portfolio_formation_date is None:
        formation_date = r.index.min()
    else:
        formation_date = pd.to_datetime(portfolio_formation_date)

    if formation_date < r.index.min():
        raise ValueError("portfolio_formation_date is before the start of the return series.")
    r = r.loc[r.index >= formation_date]

    if r.empty:
        raise ValueError("portfolio_formation_date is after the end of the return series.")

    w0 = pd.Series(starting_weights, dtype=float).reindex(r.columns).fillna(0.0)
    if w0.sum() <= 0:
        raise ValueError("starting_weights sum to 0.")
    w0 = w0 / w0.sum()

    rel_wealth = (1.0 + r.fillna(0.0)).cumprod()
    dollar = rel_wealth.mul(w0, axis=1)
    weights_eod = dollar.div(dollar.sum(axis=1), axis=0)

    return weights_eod


def calculate_drifting_portfolio_return(weights_eod, asset_returns, name):
    if weights_eod is None:
        return None

    w = weights_eod.copy()
    r = asset_returns.copy()

    w.index = pd.to_datetime(w.index)
    r.index = pd.to_datetime(r.index)

    idx = w.index.intersection(r.index)
    cols = w.columns.intersection(r.columns)

    w = w.loc[idx, cols].sort_index()
    r = r.loc[idx, cols].sort_index().fillna(0.0)

    w = w.div(w.sum(axis=1), axis=0)
    w_lag = w.shift(1)
    w_lag.iloc[0] = w.iloc[0]

    out = (w_lag * r).sum(axis=1).to_frame(name)
    return out


def calculate_current_portfolio_return(current_portfolio_weights, asset_returns):
    w = current_portfolio_weights.copy()
    r = asset_returns.copy()

    w.index = pd.to_datetime(w.index)
    r.index = pd.to_datetime(r.index)

    idx = w.index.intersection(r.index)
    cols = w.columns.intersection(r.columns)

    w = w.loc[idx, cols].sort_index()
    r = r.loc[idx, cols].sort_index()

    out = (w * r).sum(axis=1).to_frame("Current Portfolio")
    return out


def calculate_custom_portfolio_return(current_portfolio_weights, asset_returns):
    w = current_portfolio_weights.copy()
    r = asset_returns.copy()

    w.index = pd.to_datetime(w.index)
    r.index = pd.to_datetime(r.index)

    idx = w.index.intersection(r.index)
    cols = w.columns.intersection(r.columns)

    w = w.loc[idx, cols].sort_index()
    r = r.loc[idx, cols].sort_index()

    out = (w * r).sum(axis=1).to_frame("Custom Portfolio")
    return out


def _normalize_rfr(rfr, index, periods_per_year=252, scalar_is_annualized=None, name="Risk-Free Rate"):
    if rfr is None:
        return None

    idx = pd.to_datetime(index)

    if np.isscalar(rfr):
        x = float(rfr)

        if scalar_is_annualized is None:
            scalar_is_annualized = abs(x) > 0.02

        if scalar_is_annualized:
            daily = (1.0 + x) ** (1.0 / periods_per_year) - 1.0
        else:
            daily = x

        return pd.Series(daily, index=idx, name=name)

    s = rfr.copy()
    if isinstance(s, pd.DataFrame):
        if s.shape[1] != 1:
            raise ValueError("_normalize_rfr: rfr DataFrame must have exactly 1 column")
        col = s.columns[0]
        s = s[col]
        if name is None:
            name = str(col)

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.reindex(idx).ffill()
    s.name = name
    return s


def get_starting_weights(tickers, weights=None, purchase_prices=None, input_type="weights", tol=1e-8):
    if input_type == "weights":
        if weights is None:
            raise ValueError("weights must be provided when input_type='weights'")
        if len(weights) != len(tickers):
            raise ValueError("weights must have the same length as tickers")
        w = [float(x) for x in weights]
        if not np.isclose(sum(w), 1.0, atol=tol):
            raise ValueError("weights must sum to 1")
        return dict(zip(tickers, w))

    if input_type == "prices":
        if purchase_prices is None:
            raise ValueError("purchase_prices must be provided when input_type='prices'")
        if len(purchase_prices) != len(tickers):
            raise ValueError("purchase_prices must have the same length as tickers")
        p = [float(x) for x in purchase_prices]
        s = sum(p)
        w = [x / s for x in p]
        return dict(zip(tickers, w))

    raise ValueError("input_type must be 'weights' or 'prices'")


def add_benchmark_and_risk_free_returns(portfolio_returns, benchmark=None, risk_free_rate=None, periods_per_year=252, scalar_is_annualized=None):
    out = portfolio_returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    if benchmark is not None:
        b = benchmark.copy()
        if isinstance(b, pd.Series):
            b = b.to_frame()
        b.index = pd.to_datetime(b.index)
        b = b.sort_index()
        out = out.join(b, how="outer")

    if risk_free_rate is not None:
        rf_name = None
        if isinstance(risk_free_rate, pd.DataFrame) and risk_free_rate.shape[1] == 1:
            rf_name = str(risk_free_rate.columns[0])
        elif isinstance(risk_free_rate, pd.Series) and risk_free_rate.name:
            rf_name = str(risk_free_rate.name)
        rf_name = rf_name or "Risk-Free Rate"

        rf = _normalize_rfr(
            risk_free_rate,
            index=out.index,
            periods_per_year=periods_per_year,
            scalar_is_annualized=scalar_is_annualized,
            name=rf_name,
        )
        out = out.join(rf.to_frame(), how="outer")

    out = out.dropna(how="all").sort_index()
    return out


def check_weights_all(weights_all, bound=1.0, tol=1e-8):
    W = weights_all.copy()
    sums = W.sum(axis=0)
    mins = W.min(axis=0)
    maxs = W.max(axis=0)
    ok_sum = (sums - 1.0).abs() <= tol
    ok_min = mins >= -tol
    ok_max = maxs <= bound + tol
    return pd.DataFrame(
        {"sum": sums, "min": mins, "max": maxs, "ok_sum": ok_sum, "ok_min": ok_min, "ok_max": ok_max},
        index=W.columns
    )


def normalize_weights_all(weights_all, bound=1.0):
    W = weights_all.copy().astype(float)
    W = W.clip(lower=0.0)
    W = W.div(W.sum(axis=0), axis=1)
    if bound is not None:
        W = W.clip(upper=bound)
        W = W.div(W.sum(axis=0), axis=1)
    return W


def add_weights_to_weights_all(weights_all, current_portfolio_weights=None, custom_portfolio_weights=None):
    out = weights_all.copy()

    def _last_weights_df(wdf, name):
        if wdf is None:
            return None
        if not isinstance(wdf, pd.DataFrame):
            raise ValueError(f"{name} must be a DataFrame of time-varying weights")
        if wdf.empty:
            raise ValueError(f"{name} is empty")
        s = wdf.iloc[-1].copy()
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        s = s.clip(lower=0.0)
        if float(s.sum()) <= 0:
            raise ValueError(f"{name} last row sums to 0")
        s = s / float(s.sum())
        return s.rename(name).to_frame()

    cur = _last_weights_df(current_portfolio_weights, "Current Portfolio")
    if cur is not None:
        out = out.join(cur, how="left")

    cus = _last_weights_df(custom_portfolio_weights, "Custom Portfolio")
    if cus is not None:
        out = out.join(cus, how="left")

    out = out.fillna(0.0)
    out.index.name = "Ticker"
    return out


def weights_to_dollar_panel(
    weights_all,
    capital=100_000,
    current_col="Current Portfolio",
    weights_block_name="Weights",
    dollars_block_name="Dollar Allocation",
    delta_block_name="Rebalance Delta",
):
    if not isinstance(weights_all, pd.DataFrame):
        raise TypeError("weights_all must be a pandas DataFrame.")

    weights_num = weights_all.copy().apply(pd.to_numeric, errors="coerce")

    if weights_num.index.name is None:
        weights_num.index.name = "Ticker"

    dollars_num = weights_num * float(capital)

    delta_num = None
    if current_col in weights_num.columns:
        current_dollars = dollars_num[current_col]
        delta_num = dollars_num.sub(current_dollars, axis=0)

    def _fmt_weight(x):
        if pd.isna(x):
            return ""
        return f"{round(float(x), 4) * 100:.2f}%"

    def _fmt_dollar(x):
        if pd.isna(x):
            return ""
        x = round(float(x), 2)
        return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"

    weights_df = weights_num.map(_fmt_weight)
    dollars_df = dollars_num.map(_fmt_dollar)

    delta_df = None
    out = {
        weights_block_name: weights_df,
        dollars_block_name: dollars_df,
    }

    if delta_num is not None:
        delta_df = delta_num.map(_fmt_dollar)
        out[delta_block_name] = delta_df

    panel_df = pd.concat(out, axis=0)
    panel_df.index.names = ["Block", weights_num.index.name]

    return weights_df, dollars_df, delta_df, panel_df

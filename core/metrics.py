import numpy as np
import pandas as pd
from scipy.stats import linregress, norm, skew, kurtosis


def _max_drawdown_from_returns(r):
    x = pd.Series(r).dropna().astype(float)
    if x.empty:
        return np.nan
    wealth = (1.0 + x).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def _expected_shortfall_loss_from_returns(r, beta=0.95):
    x = pd.Series(r).dropna().astype(float)
    if x.empty:
        return np.nan
    losses = (-x).values
    k = max(1, int(np.floor((1.0 - beta) * len(losses))))
    tail = np.sort(losses)[-k:]
    return float(tail.mean())


def _sharpe_ratio_from_returns(r, rf=None):
    x = pd.Series(r).dropna().astype(float)
    if x.empty:
        return np.nan
    if rf is None:
        ex = x
    else:
        rf = pd.Series(rf).reindex(x.index).ffill()
        ex = x - rf
    mu = float(ex.mean())
    sd = float(ex.std(ddof=0))
    if sd <= 0:
        return np.nan
    return mu / sd


def _diversification_ratio_from_cov(w, cov):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.nan
    w = w / s
    C = np.asarray(cov, dtype=float)
    vols = np.sqrt(np.clip(np.diag(C), 0.0, None))
    denom = float(np.sqrt(max(w @ C @ w, 0.0)))
    if denom <= 0:
        return np.nan
    return float((w @ vols) / denom)


def evaluate_portfolio_returns_all(
    portfolio_returns_all,
    asset_returns,
    weights_all=None,
    cov_matrix=None,
    risk_free_rate=None,
    beta=0.95,
    annualization=252,
    dd_as_positive=True,
    annualize=False
):
    R = portfolio_returns_all.copy()
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()

    def _as_series(x, name=None):
        if x is None:
            return None
        if isinstance(x, pd.Series):
            s = x.copy()
            if name is not None:
                s.name = name
            return s
        if isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError("Expected a Series or 1-col DataFrame")
            s = x.iloc[:, 0].copy()
            if name is not None:
                s.name = name
            return s
        raise ValueError("Expected a Series or 1-col DataFrame")

    def _max_drawdown_from_returns(r):
        x = pd.Series(r).dropna().astype(float)
        if x.empty:
            return np.nan
        wealth = (1.0 + x).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        return float(dd.min())

    def _expected_shortfall_loss_from_returns(r, beta=0.95):
        x = pd.Series(r).dropna().astype(float)
        if x.empty:
            return np.nan
        losses = (-x).values
        k = max(1, int(np.floor((1.0 - beta) * len(losses))))
        tail = np.sort(losses)[-k:]
        return float(tail.mean())

    def _sharpe_ratio_from_returns(r, rf=None):
        x = pd.Series(r).dropna().astype(float)
        if x.empty:
            return np.nan
        if rf is None:
            ex = x
        else:
            rf2 = pd.Series(rf).reindex(x.index).ffill()
            ex = x - rf2
        mu = float(ex.mean())
        sd = float(ex.std(ddof=0))
        if sd <= 0:
            return np.nan
        return mu / sd

    def _diversification_ratio_from_cov(w, cov):
        w = np.asarray(w, dtype=float)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s <= 0:
            return np.nan
        w = w / s
        C = np.asarray(cov, dtype=float)
        vols = np.sqrt(np.clip(np.diag(C), 0.0, None))
        denom = float(np.sqrt(max(w @ C @ w, 0.0)))
        if denom <= 0:
            return np.nan
        return float((w @ vols) / denom)

    rf = _as_series(risk_free_rate, name="Risk-Free Rate") if risk_free_rate is not None else None
    if rf is None:
        rf_cols = [c for c in R.columns if "risk-free rate" in str(c).lower()]
        if rf_cols:
            rf = _as_series(R[[rf_cols[0]]], name="Risk-Free Rate")

    metrics = {}
    for col in R.columns:
        if "risk-free rate" in str(col).lower():
            continue

        r = R[col].astype(float)

        vol = float(r.dropna().std(ddof=0))
        sharpe = _sharpe_ratio_from_returns(r, rf=rf)

        if annualize:
            vol = vol * np.sqrt(annualization)
            sharpe = sharpe * np.sqrt(annualization)

        dd_min = _max_drawdown_from_returns(r)
        dd_val = float(-dd_min) if (dd_as_positive and dd_min == dd_min) else float(dd_min)
        es_loss = _expected_shortfall_loss_from_returns(r, beta=beta)

        metrics[col] = {
            "Total Risk": vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": dd_val,
            "Expected Shortfall": es_loss,
        }

    out = pd.DataFrame(metrics).T

    if weights_all is not None:
        if cov_matrix is None:
            X = asset_returns.copy().apply(pd.to_numeric, errors="coerce").dropna(how="any")
            cov_matrix = X.cov()
        tickers = list(cov_matrix.columns)
        C = cov_matrix.reindex(index=tickers, columns=tickers).values

        W = weights_all.reindex(tickers).fillna(0.0).clip(lower=0.0)
        W = W.div(W.sum(axis=0), axis=1)

        out["Diversification Ratio"] = np.nan
        for method in W.columns:
            if method in out.index:
                out.loc[method, "Diversification Ratio"] = _diversification_ratio_from_cov(W[method].values, C)

    out = out[["Diversification Ratio", "Total Risk", "Sharpe Ratio", "Max Drawdown", "Expected Shortfall"]]
    return out


def _as_series(x, name=None):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        s = x.copy()
        if name is not None:
            s.name = name
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a 1-column DataFrame.")
        s = x.iloc[:, 0].copy()
        if name is not None:
            s.name = name
        return s
    raise TypeError("Expected a pandas Series or 1-column DataFrame.")


def _as_df(x):
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("Expected a pandas Series or DataFrame.")


def comp(returns):
    r = _as_df(returns)
    return (1.0 + r).prod(axis=0) - 1.0


def drawdown_series(returns, capital=1.0):
    r = _as_df(returns)
    wealth = capital * (1.0 + r).cumprod(axis=0)
    peaks = wealth.cummax(axis=0)
    dd = (wealth - peaks) / peaks
    return dd, wealth, peaks


def max_drawdown(returns):
    dd, _, _ = drawdown_series(returns)
    return dd.min(axis=0).to_frame(name="Max. Drawdown").T


def _drawdown_longest_one(dd_s):
    no_dd = (dd_s == 0)
    starts = (~no_dd) & (no_dd.shift(1, fill_value=True))
    ends = (no_dd) & ((~no_dd).shift(1, fill_value=False))

    starts = list(dd_s.index[starts])
    ends = list(dd_s.index[ends])

    if len(starts) == 0:
        return pd.Series(
            {"Start": pd.NaT, "Valley": pd.NaT, "End": pd.NaT, "Days": 0, "Max. Drawdown": 0.0}
        )

    if len(ends) == 0 or (starts and ends and starts[0] > ends[0]):
        starts = [dd_s.index[0]] + starts
    if len(ends) < len(starts):
        ends = ends + [dd_s.index[-1]]

    best_row = None
    best_days = -1
    for st, en in zip(starts, ends):
        seg = dd_s.loc[st:en]
        if seg.empty:
            continue
        valley = seg.idxmin()
        days = int((en - st).days) if hasattr(en, "to_pydatetime") else int((pd.Timestamp(en) - pd.Timestamp(st)).days)
        row = {
            "Start": st,
            "Valley": valley,
            "End": en,
            "Days": days,
            "Max. Drawdown": float(seg.min()),
        }
        if days > best_days:
            best_days = days
            best_row = row

    if best_row is None:
        return pd.Series({"Start": pd.NaT, "Valley": pd.NaT, "End": pd.NaT, "Days": 0, "Max. Drawdown": 0.0})
    return pd.Series(best_row)


def drawdown_info(returns):
    dd, _, _ = drawdown_series(returns)
    out = {}
    for c in dd.columns:
        out[c] = _drawdown_longest_one(dd[c])
    df = pd.DataFrame(out).T
    return df


def semidev(returns, threshold=0.0):
    r = _as_df(returns)
    neg = (r - threshold).where((r - threshold) < 0.0)
    return neg.std(axis=0).to_frame(name="Semi-Deviation").T


def var_hist(returns, level=5):
    r = _as_df(returns)
    q = r.quantile(level / 100.0, axis=0)
    return q.to_frame(name=f"Historical VaR ({level}%)").T


def cvar_hist(returns, level=5):
    r = _as_df(returns)
    v = r.quantile(level / 100.0, axis=0)

    def _cvar_col(x):
        thr = v.loc[x.name]
        tail = x[x <= thr]
        return float(tail.mean()) if len(tail) else np.nan

    c = r.apply(_cvar_col, axis=0)
    return c.to_frame(name=f"Conditional VaR ({level}%)").T


def var_gauss(returns, level=5, modified=False):
    r = _as_df(returns)
    z = norm.ppf(level / 100.0)

    if not modified:
        out = r.mean(axis=0) + z * r.std(axis=0, ddof=0)
        return out.to_frame(name=f"Gaussian VaR ({level}%)").T

    s = r.apply(lambda x: skew(x, nan_policy="omit"), axis=0)
    k = r.apply(lambda x: kurtosis(x, fisher=True, nan_policy="omit"), axis=0)
    z_cf = z + (z**2 - 1.0) * s / 6.0 + (z**3 - 3.0 * z) * k / 24.0 - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    out = r.mean(axis=0) + z_cf * r.std(axis=0, ddof=0)
    return out.to_frame(name=f"Modified Gaussian VaR ({level}%)").T


def win_rate(returns):
    r = _as_df(returns)
    denom = (r != 0).sum(axis=0).replace(0, np.nan)
    wr = (r > 0).sum(axis=0) / denom
    return wr.to_frame(name="Win Rate").T


def avg_return(returns):
    r = _as_df(returns)
    return r.replace(0, np.nan).mean(axis=0).to_frame(name="Average Return").T


def avg_win(returns):
    r = _as_df(returns)
    return r.where(r > 0).mean(axis=0).to_frame(name="Average Win").T


def avg_loss(returns):
    r = _as_df(returns)
    return r.where(r < 0).mean(axis=0).to_frame(name="Average Loss").T


def best(returns):
    r = _as_df(returns)
    return r.max(axis=0).to_frame(name="Best Return").T


def worst(returns):
    r = _as_df(returns)
    return r.min(axis=0).to_frame(name="Worst Return").T


def sharpe_ratio(returns, rf, periods=252):
    r = _as_df(returns)
    ex = r.sub(rf, axis=0)
    sr = ex.mean(axis=0) / ex.std(axis=0, ddof=0)
    return (sr * np.sqrt(periods)).to_frame(name="Sharpe Ratio").T


def sortino_ratio(returns, rf, periods=252):
    r = _as_df(returns)
    ex = r.sub(rf, axis=0)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean(axis=0))
    out = ex.mean(axis=0) / downside.replace(0, np.nan)
    return (out * np.sqrt(periods)).to_frame(name="Sortino Ratio").T


def treynor_ratio(returns, rf, benchmark, periods=252):
    r = _as_df(returns)
    ex = r.sub(rf, axis=0)

    betas = {}
    for c in r.columns:
        df = pd.concat([benchmark, r[c]], axis=1).dropna()
        if df.shape[0] < 3:
            betas[c] = np.nan
            continue
        slope, _, _, _, _ = linregress(df.iloc[:, 0].values, df.iloc[:, 1].values)
        betas[c] = slope
    beta_s = pd.Series(betas)

    tr = (ex.mean(axis=0) * periods) / beta_s.replace(0, np.nan)
    return tr.to_frame(name="Treynor Ratio").T


def omega_ratio(returns, required_return_annual=0.0, periods=252):
    r = _as_df(returns)
    thr = (1.0 + float(required_return_annual)) ** (1.0 / periods) - 1.0
    win = (r - thr).clip(lower=0.0).sum(axis=0)
    lose = (thr - r).clip(lower=0.0).sum(axis=0)
    out = win / lose.replace(0, np.nan)
    return out.to_frame(name="Omega Ratio").T


def gain_to_pain_ratio(returns):
    r = _as_df(returns)
    down = r.where(r < 0.0).sum(axis=0).abs()
    out = r.sum(axis=0) / down.replace(0, np.nan)
    return out.to_frame(name="Gain-To-Pain Ratio").T


def rsquared(returns, benchmark):
    r = _as_df(returns)
    out = {}
    for c in r.columns:
        df = pd.concat([benchmark, r[c]], axis=1).dropna()
        if df.shape[0] < 3:
            out[c] = np.nan
            continue
        _, _, rval, _, _ = linregress(df.iloc[:, 0].values, df.iloc[:, 1].values)
        out[c] = rval ** 2
    return pd.Series(out).to_frame(name="R-Squared").T


def alpha_beta(returns, benchmark, periods=252):
    r = _as_df(returns)
    alpha = {}
    beta = {}
    for c in r.columns:
        df = pd.concat([benchmark, r[c]], axis=1).dropna()
        if df.shape[0] < 3:
            alpha[c] = np.nan
            beta[c] = np.nan
            continue
        slope, intercept, _, _, _ = linregress(df.iloc[:, 0].values, df.iloc[:, 1].values)
        beta[c] = slope
        alpha[c] = intercept * periods
    out = pd.DataFrame({"Alpha": pd.Series(alpha), "Beta": pd.Series(beta)}).T
    return out


def information_ratio(returns, benchmark, periods=252):
    r = _as_df(returns)
    active = r.sub(benchmark, axis=0)
    ir = active.mean(axis=0) / active.std(axis=0, ddof=0)
    return (ir * np.sqrt(periods)).to_frame(name="Information Ratio").T


def payoff_ratio(returns):
    aw = avg_win(returns).iloc[0]
    al = avg_loss(returns).iloc[0].abs()
    out = aw / al.replace(0, np.nan)
    return out.to_frame(name="Payoff Ratio").T


def profit_factor(returns):
    r = _as_df(returns)
    wins = r.where(r >= 0.0).sum(axis=0)
    losses = r.where(r < 0.0).sum(axis=0).abs()
    out = wins / losses.replace(0, np.nan)
    return out.to_frame(name="Profit Factor").T


def profit_ratio(returns):
    r = _as_df(returns)
    mw = r.where(r > 0.0).mean(axis=0)
    ml = r.where(r < 0.0).mean(axis=0).abs()
    out = mw / ml.replace(0, np.nan)
    return out.to_frame(name="Profit Ratio").T


def cpc_index(returns):
    pf = profit_factor(returns).iloc[0]
    wr = win_rate(returns).iloc[0]
    pr = payoff_ratio(returns).iloc[0]
    out = pf * wr * pr
    return out.to_frame(name="CPC Index").T


def kelly_criterion(returns):
    p = win_rate(returns).iloc[0]
    b = payoff_ratio(returns).iloc[0]
    q = 1.0 - p
    out = (b * p - q) / b.replace(0, np.nan)
    return out.to_frame(name="Kelly Criterion").T


def tail_ratio(returns, cutoff=0.95):
    r = _as_df(returns)
    hi = r.quantile(cutoff, axis=0)
    lo = r.quantile(1.0 - cutoff, axis=0)
    out = (hi.abs() / lo.abs().replace(0, np.nan))
    return out.to_frame(name="Tail Ratio").T


def common_sense_ratio(returns, cutoff=0.95):
    pf = profit_factor(returns).iloc[0]
    tr = tail_ratio(returns, cutoff=cutoff).iloc[0]
    return (pf * tr).to_frame(name="Common Sense Ratio").T


def outlier_win_ratio(returns, quantile=0.99):
    r = _as_df(returns)
    q = r.quantile(quantile, axis=0)
    mw = r.where(r >= 0.0).mean(axis=0)
    out = q / mw.replace(0, np.nan)
    return out.to_frame(name="Outlier-to-Win Ratio").T


def outlier_loss_ratio(returns, quantile=0.01):
    r = _as_df(returns)
    q = r.quantile(quantile, axis=0).abs()
    ml = r.where(r < 0.0).mean(axis=0).abs()
    out = q / ml.replace(0, np.nan)
    return out.to_frame(name="Outlier-to-Loss Ratio").T


def cagr(returns):
    r = _as_df(returns)
    wealth_end = (1.0 + r).prod(axis=0)
    years = (r.index[-1] - r.index[0]).days / 365.25
    out = wealth_end ** (1.0 / years) - 1.0
    return out.to_frame(name="CAGR").T


def calmar_ratio(returns):
    c = cagr(returns).iloc[0]
    mdd = max_drawdown(returns).iloc[0].abs()
    out = c / mdd.replace(0, np.nan)
    return out.to_frame(name="Calmar Ratio").T


def ulcer_index(returns):
    dd, _, _ = drawdown_series(returns)
    ui = np.sqrt((dd ** 2).mean(axis=0))
    return ui.to_frame(name="Ulcer Index").T


def ulcer_performance_index(returns, rf_returns):
    c_strat = cagr(returns).iloc[0]
    c_rf = cagr(rf_returns.to_frame("RF")).iloc[0]["RF"]
    ui = ulcer_index(returns).iloc[0]
    out = (c_strat - c_rf) / ui.replace(0, np.nan)
    return out.to_frame(name="Ulcer Performance Index").T


def serenity_index(returns, rf_returns, cvar_level=5):
    c_strat = cagr(returns).iloc[0]
    c_rf = cagr(rf_returns.to_frame("RF")).iloc[0]["RF"]
    ui = ulcer_index(returns).iloc[0]

    dd, _, _ = drawdown_series(returns)
    dd_cvar = cvar_hist(dd, level=cvar_level).iloc[0].abs()
    vol = _as_df(returns).std(axis=0, ddof=0).replace(0, np.nan)
    pitfall = dd_cvar / vol

    out = (c_strat - c_rf) / (ui * pitfall).replace(0, np.nan)
    return out.to_frame(name="Serenity Index").T


def recovery_factor(returns):
    total = comp(returns)
    mdd = max_drawdown(returns).iloc[0].abs()
    out = total / mdd.replace(0, np.nan)
    return out.to_frame(name="Recovery Factor").T


def risk_of_ruin(returns, trials=15):
    p = win_rate(returns).iloc[0]
    q = 1.0 - p
    out = pd.Series(np.where((p > q) & (p > 0), (q / p) ** trials, 1.0), index=p.index)
    return out.to_frame(name=f"Risk of Ruin (q/p)^{trials}").T


def total_return(returns):
    r = _as_df(returns)
    out = (1.0 + r).prod(axis=0) - 1.0
    return out.to_frame(name="Total Return").T


def ann_mean_return(returns, periods=252):
    r = _as_df(returns)
    out = r.mean(axis=0) * periods
    return out.to_frame(name="Annualized Mean Return").T


def ann_vol(returns, periods=252):
    r = _as_df(returns)
    out = r.std(axis=0, ddof=0) * np.sqrt(periods)
    return out.to_frame(name="Annualized Volatility").T


def tracking_error(returns, benchmark, periods=252):
    r = _as_df(returns)
    active = r.sub(benchmark, axis=0)
    out = active.std(axis=0, ddof=0) * np.sqrt(periods)
    return out.to_frame(name="Tracking Error").T


def ann_active_return(returns, benchmark, periods=252):
    r = _as_df(returns)
    active = r.sub(benchmark, axis=0)
    out = active.mean(axis=0) * periods
    return out.to_frame(name="Annualized Active Return").T


def corr_with_benchmark(returns, benchmark):
    r = _as_df(returns)
    out = r.apply(lambda x: x.corr(benchmark), axis=0)
    return out.to_frame(name="Correlation (Benchmark)").T


def best_day(returns):
    r = _as_df(returns)
    return r.max(axis=0).to_frame(name="Best Day").T


def worst_day(returns):
    r = _as_df(returns)
    return r.min(axis=0).to_frame(name="Worst Day").T


def _align_daily_panels(returns, benchmark=None, rfr=None):
    rets = _as_df(returns).copy()

    if not isinstance(rets.index, pd.DatetimeIndex):
        rets.index = pd.to_datetime(rets.index)
    rets = rets.sort_index()

    bench = None
    if benchmark is not None:
        bench = _as_series(benchmark, name="Benchmark")
        if not isinstance(bench.index, pd.DatetimeIndex):
            bench.index = pd.to_datetime(bench.index)
        bench = bench.sort_index().dropna()
        bench.name = "Benchmark"

    rf = None
    if rfr is not None:
        rf = _as_series(rfr, name="RF")
        if not isinstance(rf.index, pd.DatetimeIndex):
            rf.index = pd.to_datetime(rf.index)
        rf = rf.sort_index().dropna()
        rf.name = "RF"

    parts = [rets]
    if bench is not None:
        parts.append(bench)
    if rf is not None:
        parts.append(rf)

    panel = pd.concat(parts, axis=1).dropna(how="any").sort_index()

    rets2 = panel[rets.columns]
    bench2 = panel["Benchmark"] if "Benchmark" in panel.columns else None
    rf2 = panel["RF"] if "RF" in panel.columns else None

    return rets2, bench2, rf2


def calc_port_stats(
    returns,
    rfr=None,
    benchmark=None,
    required_return_annual=0.0,
    periods=252,
    var_level=5,
    modified_var=False,
    tail_cutoff=0.95,
    ror_trials=15,
):
    rets, bench, rf = _align_daily_panels(returns, benchmark=benchmark, rfr=rfr)

    if rets is None or rets.empty:
        empty_dd = pd.DataFrame(columns=["Start", "Valley", "End", "Days", "Max. Drawdown"])
        return pd.DataFrame(), empty_dd

    parts = []

    parts.append(total_return(rets))
    parts.append(cagr(rets))
    parts.append(ann_mean_return(rets, periods=periods))
    parts.append(avg_return(rets))
    parts.append(best_day(rets))
    parts.append(worst_day(rets))

    parts.append(ann_vol(rets, periods=periods))
    parts.append(semidev(rets, threshold=0.0))

    if rf is not None:
        parts.append(sharpe_ratio(rets, rf, periods=periods))
        parts.append(sortino_ratio(rets, rf, periods=periods))
        parts.append(ulcer_performance_index(rets, rf))
        parts.append(serenity_index(rets, rf, cvar_level=var_level))

    parts.append(gain_to_pain_ratio(rets))
    parts.append(omega_ratio(rets, required_return_annual=required_return_annual, periods=periods))
    parts.append(calmar_ratio(rets))
    parts.append(recovery_factor(rets))

    parts.append(max_drawdown(rets))
    parts.append(ulcer_index(rets))

    if bench is not None:
        ab = alpha_beta(rets, bench, periods=periods)
        beta_row = ab.loc[["Beta"]] if "Beta" in ab.index else None
        alpha_row = ab.loc[["Alpha"]] if "Alpha" in ab.index else None

        parts.append(corr_with_benchmark(rets, bench))
        if beta_row is not None:
            parts.append(beta_row)
        parts.append(rsquared(rets, bench))
        if alpha_row is not None:
            parts.append(alpha_row)
        parts.append(ann_active_return(rets, bench, periods=periods))
        parts.append(information_ratio(rets, bench, periods=periods))

        if rf is not None:
            parts.append(treynor_ratio(rets, rf, bench, periods=periods))

    parts.append(var_hist(rets, level=var_level))
    parts.append(var_gauss(rets, level=var_level, modified=False))
    if modified_var:
        parts.append(var_gauss(rets, level=var_level, modified=True))
    parts.append(cvar_hist(rets, level=var_level))
    parts.append(tail_ratio(rets, cutoff=tail_cutoff))
    parts.append(common_sense_ratio(rets, cutoff=tail_cutoff))
    parts.append(outlier_win_ratio(rets, quantile=0.99))
    parts.append(outlier_loss_ratio(rets, quantile=0.01))

    parts.append(win_rate(rets))
    parts.append(avg_win(rets))
    parts.append(avg_loss(rets))
    parts.append(payoff_ratio(rets))
    parts.append(profit_factor(rets))
    parts.append(profit_ratio(rets))
    parts.append(cpc_index(rets))
    parts.append(kelly_criterion(rets))
    parts.append(risk_of_ruin(rets, trials=ror_trials))

    parts = [p for p in parts if p is not None]
    stats = pd.concat(parts, axis=0) if len(parts) > 0 else pd.DataFrame()

    dd_info = drawdown_info(rets)
    dd_info_fmt = dd_info.copy()
    for c in ["Start", "Valley", "End"]:
        if c in dd_info_fmt.columns:
            dd_info_fmt[c] = pd.to_datetime(dd_info_fmt[c], errors="coerce").dt.strftime("%Y-%m-%d")
    if "Days" in dd_info_fmt.columns:
        dd_info_fmt["Days"] = pd.to_numeric(dd_info_fmt["Days"], errors="coerce").fillna(0).astype(int)
    if "Max. Drawdown" in dd_info_fmt.columns:
        dd_info_fmt["Max. Drawdown"] = pd.to_numeric(dd_info_fmt["Max. Drawdown"], errors="coerce")

    return stats, dd_info_fmt

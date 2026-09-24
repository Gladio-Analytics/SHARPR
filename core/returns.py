import numpy as np
import pandas as pd


def ffill_prices(df, limit=None, also_bfill=False):
    x = df.sort_index().copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    x = x.ffill(limit=limit)
    if also_bfill:
        x = x.bfill(limit=limit)

    return x


def cut_complete_asset_sample(asset_returns, *dfs):
    R = asset_returns.copy()
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()
    R = R.dropna(how="any")

    idx = R.index

    out = [R]
    for d in dfs:
        if d is None:
            out.append(None)
            continue
        x = d.copy()
        if isinstance(x, pd.Series):
            x = x.to_frame()
        x.index = pd.to_datetime(x.index)
        x = x.sort_index().reindex(idx)
        out.append(x)

    return tuple(out)


def get_cov_matrix(asset_returns):
    return asset_returns.cov()


def get_corr_matrix(asset_returns):
    return asset_returns.corr()


def get_sample_means(asset_returns):
    return asset_returns.mean()


def get_mean_of_sample_means(asset_returns):
    return float(asset_returns.mean().mean())


def to_returns(df, log=False):
    if df is None:
        return None

    if not isinstance(df, (pd.Series, pd.DataFrame)):
        raise TypeError("df must be a pandas Series, DataFrame, or None.")

    x = df.copy()
    x.index = pd.to_datetime(x.index)
    x = x.sort_index().pct_change()

    if log:
        x = np.log1p(x)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x.dropna(how="all")


def prepare_return_panel(prices, benchmark=None, risk_free_rate=None, log=False):
    prices_ret = to_returns(prices, log=log)
    if prices_ret is None:
        raise ValueError("prices cannot be None.")

    benchmark_ret = to_returns(benchmark, log=log)
    risk_free_ret = to_returns(risk_free_rate, log=log)

    prices_ret, benchmark_ret, risk_free_ret = cut_complete_asset_sample(
        prices_ret, benchmark_ret, risk_free_ret
    )

    blocks = [prices_ret]
    if benchmark_ret is not None:
        blocks.append(benchmark_ret)
    if risk_free_ret is not None:
        blocks.append(risk_free_ret)

    panel = pd.concat(blocks, axis=1) if len(blocks) > 0 else None
    return prices_ret, benchmark_ret, risk_free_ret, panel

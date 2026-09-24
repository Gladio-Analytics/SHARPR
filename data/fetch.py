#IMPORTS

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from scipy.optimize import minimize, linprog
from scipy.interpolate import PchipInterpolator
import cvxpy as cp




def check_tickers(tickers):
    import yfinance as yf
    import pandas as pd
    import logging
    import io
    import contextlib

    if tickers is None:
        return [], [], None, "None of the tickers in the database"

    if not isinstance(tickers, (list, tuple)):
        raise TypeError("tickers must be a list or tuple of ticker strings.")

    tickers_in = [str(t).strip().upper() for t in tickers if str(t).strip()]

    if len(tickers_in) == 0:
        return [], [], None, "None of the tickers in the database"

    found = []
    not_found = []

    yf_logger = yf.utils.get_yf_logger()
    old_disabled = yf_logger.disabled
    old_level = yf_logger.level

    yf_logger.disabled = True
    yf_logger.setLevel(logging.CRITICAL)

    try:
        for t in tickers_in:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    df = yf.download(
                        t,
                        period="1d",
                        interval="1d",
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                    )

                ok = isinstance(df, pd.DataFrame) and not df.empty

                if ok and isinstance(df.columns, pd.MultiIndex):
                    ok = len(df.columns.get_level_values(0)) > 0

                if ok:
                    found.append(t)
                else:
                    not_found.append(t)

            except Exception:
                not_found.append(t)
    finally:
        yf_logger.disabled = old_disabled
        yf_logger.setLevel(old_level)

    usable_tickers = found.copy() if len(found) > 0 else None

    found_msg = f"Tickers: {', '.join(found)} found in the database" if len(found) > 0 else ""
    not_found_msg = f"Tickers: {', '.join(not_found)} not found in the database" if len(not_found) > 0 else ""

    if found_msg and not_found_msg:
        message = found_msg + "\n" + not_found_msg
    elif found_msg:
        message = found_msg
    elif not_found_msg:
        message = not_found_msg
    else:
        message = "None of the tickers in the database"

    return found, not_found, usable_tickers, message



def fetch_prices(tickers, tidy=False):
    tickers = [str(t).upper() for t in tickers]

    series_list = []
    for t in tqdm(tickers, desc="Downloading prices"):
        d = yf.download(
            t,
            period="max",
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False
        )
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.rename(t)
        series_list.append(c)

    close = pd.concat(series_list, axis=1)
    close = close.reindex(columns=tickers)
    close = close.sort_index().dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"

    if tidy:
        return close.stack(dropna=False).rename("Close").reset_index().rename(columns={"level_1": "Ticker"})
    return close

def _pick_ticker(x):
    return x if isinstance(x, str) else x[0]

def _is_empty_ticker(x):
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, (list, tuple, pd.Index)):
        if len(x) == 0:
            return True
        return str(x[0]).strip() == ""
    return False

def fetch_benchmark(benchmark_ticker=None, tidy=False):
    if _is_empty_ticker(benchmark_ticker):
        return None

    ticker = _pick_ticker(benchmark_ticker)
    df = fetch_prices([ticker], tidy=False)
    df.columns = [f"Benchmark ({ticker})"]

    if tidy:
        return df.stack(dropna=False).rename("Value").reset_index().rename(columns={"level_1": "Series"})
    return df

def fetch_risk_free_rate(risk_free_ticker=None, tidy=False):
    if _is_empty_ticker(risk_free_ticker):
        return None

    ticker = _pick_ticker(risk_free_ticker)
    df = fetch_prices([ticker], tidy=False)
    df.columns = [f"Risk-Free Rate ({ticker})"]

    if tidy:
        return df.stack(dropna=False).rename("Value").reset_index().rename(columns={"level_1": "Series"})
    return df

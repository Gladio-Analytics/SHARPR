import numpy as np
import pandas as pd


def _dedupe_index(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a DataFrame.")
    if not df.index.duplicated().any():
        return df
    return df.groupby(level=0).first()


def build_total_performance_metrics_df(
    metrics_df=None,
    stats=None,
    prefer="stats",
    drop_portfolio_prefixes=("Risk-Free Rate",),
    drop_portfolio_contains=("Risk-Free Rate",),
    drop_all_nan_portfolios=True,
):
    parts = []

    if metrics_df is not None:
        m = metrics_df.copy()
        if not isinstance(m, pd.DataFrame):
            raise TypeError("metrics_df must be a DataFrame (portfolios x metrics).")
        m = _dedupe_index(m)

        for c in m.columns:
            try:
                m[c] = pd.to_numeric(m[c])
            except (ValueError, TypeError):
                pass

        parts.append(("metrics_df", m))

    if stats is not None:
        s = stats.copy()
        if not isinstance(s, pd.DataFrame):
            raise TypeError("stats must be a DataFrame (metrics x portfolios).")
        s = _dedupe_index(s)
        s = s.apply(pd.to_numeric, errors="coerce")
        parts.append(("stats", s.T))

    if not parts:
        raise ValueError("Provide at least one of metrics_df or stats.")

    if prefer not in ("stats", "metrics_df"):
        raise ValueError("prefer must be 'stats' or 'metrics_df'.")

    by_name = dict(parts)
    if prefer == "stats":
        base = by_name.get("stats", pd.DataFrame())
        other = by_name.get("metrics_df", pd.DataFrame())
        total = base.combine_first(other)
    else:
        base = by_name.get("metrics_df", pd.DataFrame())
        other = by_name.get("stats", pd.DataFrame())
        total = base.combine_first(other)

    idx_str = total.index.astype(str)
    mask = pd.Series(False, index=total.index)

    for pfx in (drop_portfolio_prefixes or []):
        mask |= idx_str.str.startswith(str(pfx))
    for sub in (drop_portfolio_contains or []):
        mask |= idx_str.str.contains(str(sub), regex=False)

    total = total.loc[~mask].copy()

    if drop_all_nan_portfolios:
        total = total.dropna(how="all", axis=0)

    return total


def _as_single_series(x):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = str(s.name) if s.name is not None else None
        return s
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame (benchmark or risk-free).")
        s = x.iloc[:, 0].copy()
        s.name = str(x.columns[0])
        return s
    raise TypeError("Expected pandas Series or single-column DataFrame.")


def _infer_special_col(df, keyword_list):
    hits = [c for c in df.columns if any(k in str(c).lower() for k in keyword_list)]
    if len(hits) == 1:
        return hits[0]
    return None


def effective_n_holdings_all(weights_all, normalize=True):
    if not isinstance(weights_all, pd.DataFrame):
        raise TypeError("weights_all must be a pandas DataFrame (index=tickers, columns=portfolios).")

    W = weights_all.copy()
    W = W.apply(pd.to_numeric, errors="coerce")

    out = {}
    for col in W.columns:
        w = W[col].to_numpy(dtype=float)
        w = w[np.isfinite(w)]
        if w.size == 0:
            out[str(col)] = np.nan
            continue

        s = float(np.sum(w))
        if normalize:
            if not np.isfinite(s) or abs(s) < 1e-12:
                out[str(col)] = np.nan
                continue
            w = w / s

        denom = float(np.sum(w * w))
        out[str(col)] = (1.0 / denom) if (np.isfinite(denom) and denom > 0) else np.nan

    return pd.Series(out, name="Effective Holdings")


def effective_n_bets_all(weights_all, cov_matrix, normalize_weights=True):
    if not isinstance(weights_all, pd.DataFrame):
        raise TypeError("weights_all must be a pandas DataFrame (index=tickers, columns=portfolios).")
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame with tickers as index/columns.")

    W = weights_all.copy()
    W = W.apply(pd.to_numeric, errors="coerce")

    C = cov_matrix.copy()
    C = C.apply(pd.to_numeric, errors="coerce")

    tickers = [t for t in W.index if t in C.index and t in C.columns]
    if len(tickers) == 0:
        raise ValueError("No overlapping tickers between weights_all.index and cov_matrix index/columns.")

    C = C.loc[tickers, tickers].to_numpy(dtype=float)

    out = {}
    for col in W.columns:
        w = W.loc[tickers, col].to_numpy(dtype=float)
        if np.all(~np.isfinite(w)):
            out[str(col)] = np.nan
            continue

        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        s = float(np.sum(w))
        if normalize_weights:
            if not np.isfinite(s) or abs(s) < 1e-12:
                out[str(col)] = np.nan
                continue
            w = w / s

        v = float(w @ C @ w)
        if not np.isfinite(v) or v <= 0:
            out[str(col)] = np.nan
            continue

        m = C @ w
        rc = w * m
        tot = float(np.sum(rc))
        if not np.isfinite(tot) or abs(tot) < 1e-18:
            out[str(col)] = np.nan
            continue

        p = rc / tot
        denom = float(np.sum(p * p))
        out[str(col)] = (1.0 / denom) if (np.isfinite(denom) and denom > 0) else np.nan

    return pd.Series(out, name="Effective Bets")


def active_return_series_all(
    returns_all,
    benchmark=None,
    benchmark_col=None,
    risk_free=None,
    risk_free_col=None,
    include_current_custom=True
):
    R = returns_all.copy()
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()

    b = _as_single_series(benchmark)
    if benchmark_col is None:
        if b is not None and b.name is not None and b.name in R.columns:
            benchmark_col = b.name
        else:
            benchmark_col = _infer_special_col(R, ["benchmark"])

    if benchmark_col is None and b is None:
        return None

    if b is None:
        b = R[benchmark_col].copy()
        b.name = str(benchmark_col)
    else:
        b.index = pd.to_datetime(b.index)
        b = b.sort_index()

    rf = _as_single_series(risk_free)
    if risk_free_col is None:
        if rf is not None and rf.name is not None and rf.name in R.columns:
            risk_free_col = rf.name
        else:
            risk_free_col = _infer_special_col(R, ["risk-free", "risk free", "riskfree"])

    cols = list(R.columns)
    drop_cols = set()
    if benchmark_col is not None and benchmark_col in cols:
        drop_cols.add(benchmark_col)
    if risk_free_col is not None and risk_free_col in cols:
        drop_cols.add(risk_free_col)

    if not include_current_custom:
        for c in cols:
            low = str(c).lower()
            if ("current" in low and "portfolio" in low) or ("custom" in low and "portfolio" in low):
                drop_cols.add(c)

    port_cols = [c for c in cols if c not in drop_cols]

    idx = R.index.union(b.index)
    b2 = b.reindex(idx)

    out = pd.DataFrame(index=idx)
    for c in port_cols:
        out[str(c)] = R[c].reindex(idx) - b2

    out = out.sort_index()
    return out


def active_metrics_summary(
    returns_all,
    benchmark=None,
    benchmark_col=None,
    risk_free=None,
    risk_free_col=None,
    annualization=252,
    include_current_custom=True,
    min_obs=20
):
    R = returns_all.copy()
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()

    b = _as_single_series(benchmark)
    if benchmark_col is None:
        if b is not None and b.name is not None and b.name in R.columns:
            benchmark_col = b.name
        else:
            benchmark_col = _infer_special_col(R, ["benchmark"])

    if benchmark_col is None and b is None:
        return None

    if b is None:
        b = R[benchmark_col].copy()
        b.name = str(benchmark_col)
    else:
        b.index = pd.to_datetime(b.index)
        b = b.sort_index()

    rf = _as_single_series(risk_free)
    if risk_free_col is None:
        if rf is not None and rf.name is not None and rf.name in R.columns:
            risk_free_col = rf.name
        else:
            risk_free_col = _infer_special_col(R, ["risk-free", "risk free", "riskfree"])

    cols = list(R.columns)
    drop_cols = set()
    if benchmark_col is not None and benchmark_col in cols:
        drop_cols.add(benchmark_col)
    if risk_free_col is not None and risk_free_col in cols:
        drop_cols.add(risk_free_col)

    if not include_current_custom:
        for c in cols:
            low = str(c).lower()
            if ("current" in low and "portfolio" in low) or ("custom" in low and "portfolio" in low):
                drop_cols.add(c)

    port_cols = [c for c in cols if c not in drop_cols]

    rows = []
    for c in port_cols:
        rp = R[c].copy()
        df_pair = pd.concat([rp, b], axis=1, join="inner").dropna()

        if df_pair.shape[0] < int(min_obs):
            rows.append(dict(
                Portfolio=str(c),
                N=int(df_pair.shape[0]),
                AnnActive_Arith=np.nan,
                AnnActive_Geom=np.nan,
                TE=np.nan,
                IR_Arith=np.nan,
                IR_Geom=np.nan
            ))
            continue

        p = df_pair.iloc[:, 0].astype(float)
        bb = df_pair.iloc[:, 1].astype(float)

        a = (p - bb).astype(float)

        ann_active_arith = float(a.mean()) * float(annualization)
        te = float(a.std(ddof=1)) * float(np.sqrt(annualization))

        wp = float(np.prod(1.0 + p.values))
        wb = float(np.prod(1.0 + bb.values))
        if np.isfinite(wp) and np.isfinite(wb) and wb > 0 and wp > 0:
            ratio = wp / wb
            ann_active_geom = float(ratio ** (float(annualization) / float(len(p))) - 1.0)
        else:
            ann_active_geom = np.nan

        ir_arith = ann_active_arith / te if (np.isfinite(te) and te > 0) else np.nan
        ir_geom = ann_active_geom / te if (np.isfinite(te) and te > 0 and np.isfinite(ann_active_geom)) else np.nan

        rows.append(dict(
            Portfolio=str(c),
            N=int(len(p)),
            AnnActive_Arith=ann_active_arith,
            AnnActive_Geom=ann_active_geom,
            TE=te,
            IR_Arith=ir_arith,
            IR_Geom=ir_geom
        ))

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "Sample Size",
            "Active Mean (Arithmetic)",
            "Active Mean (Geometric)",
            "Tracking Error",
            "Information Ratio (Arithmetic)",
            "Information Ratio (Geometric)",
        ])

    out = pd.DataFrame(rows).set_index("Portfolio")
    out.columns = [
        "Sample Size",
        "Active Mean (Arithmetic)",
        "Active Mean (Geometric)",
        "Tracking Error",
        "Information Ratio (Arithmetic)",
        "Information Ratio (Geometric)",
    ]
    return out


def add_to_calc_stats(total_df,
    summary=None,
    effective_holdings=None,
    effective_bets=None,
    risk_contribution_dispersion=None,
    prefix_summary=None,
    summary_row_prefix=None
):
    if not isinstance(total_df, pd.DataFrame):
        raise TypeError("total_df must be a DataFrame (rows=portfolios, cols=metrics).")

    base = total_df.T.copy()
    base.columns = base.columns.astype(str).str.strip()
    base.index = base.index.astype(str).str.strip()

    out = base.copy()

    if summary is not None:
        if not isinstance(summary, pd.DataFrame):
            raise TypeError("summary must be a DataFrame or None.")

        if not summary.empty:
            add = summary.T.copy()
            add.columns = add.columns.astype(str).str.strip()
            add.index = add.index.astype(str).str.strip()

            add = add.reindex(columns=out.columns)

            if prefix_summary is not None:
                add.index = [f"{prefix_summary}{i}" for i in add.index]
            if summary_row_prefix is not None:
                add.index = [f"{summary_row_prefix}{i}" for i in add.index]

            add = add.dropna(how="all", axis=0)

            if not add.empty:
                out = pd.concat([out, add], axis=0)

    def _series_to_row(s, label):
        if s is None:
            return None

        if isinstance(s, pd.DataFrame):
            if s.shape[1] != 1:
                raise ValueError(f"{label} must be a Series or a 1-column DataFrame.")
            s = s.iloc[:, 0]

        if not isinstance(s, pd.Series):
            raise TypeError(f"{label} must be a Series, a 1-column DataFrame, or None.")

        s2 = s.copy()
        s2.index = s2.index.astype(str).str.strip()
        s2 = pd.to_numeric(s2, errors="coerce")

        row = s2.to_frame().T
        row.index = [str(s2.name).strip() if s2.name is not None else label]
        row = row.reindex(columns=out.columns)
        row = row.dropna(how="all", axis=0)

        return row if not row.empty else None

    eff_row = _series_to_row(effective_holdings, "Effective Holdings")
    bets_row = _series_to_row(effective_bets, "Effective Bets")
    disp_row = _series_to_row(risk_contribution_dispersion, "Risk Contribution Dispersion")

    rows = [r for r in [eff_row, bets_row, disp_row] if r is not None]
    if rows:
        out = pd.concat([out] + rows, axis=0)

    return out

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calc_loadings(
    ret_df,
    factor_df,
    factors=None,
    use_rf=True,
    cov_type="HAC",
    maxlags=5,
    min_obs=60,
    unit_threshold=0.02
):
    if isinstance(ret_df, pd.Series):
        ret_df = ret_df.to_frame()

    ret = ret_df.copy()
    fac = factor_df.copy()

    ret.index = pd.to_datetime(ret.index)
    fac.index = pd.to_datetime(fac.index)

    ret = ret.sort_index().apply(pd.to_numeric, errors="coerce")
    fac = fac.sort_index().apply(pd.to_numeric, errors="coerce")

    fac.columns = [str(c).upper() for c in fac.columns]

    if factors is None:
        factors = ["MKT", "HML", "SMB", "CMA", "RMW", "MOM", "STREV", "LTREV"]

    factors = [str(f).upper() for f in factors]
    factors = ["SMB" if f == "SMT" else f for f in factors]
    factors = [f for f in factors if f not in ["ALPHA", "RF"]]

    seen = set()
    x_cols = []
    for f in factors:
        if f not in seen:
            x_cols.append(f)
            seen.add(f)

    missing_x = [c for c in x_cols if c not in fac.columns]
    if missing_x:
        raise KeyError(f"factor_df is missing requested factors: {missing_x}")

    if use_rf and "RF" not in fac.columns:
        raise KeyError("use_rf=True but 'RF' is missing in factor_df.")

    needed_cols = x_cols + (["RF"] if use_rf else [])
    fac = fac[needed_cols]

    def _unit_tag(s):
        x = pd.to_numeric(s, errors="coerce").dropna().abs()
        if x.empty:
            return "unknown"
        med = float(x.median())
        return "percent" if med > float(unit_threshold) else "decimal"

    ret_tag = _unit_tag(ret.stack())
    anchor = x_cols[0] if len(x_cols) > 0 else ("RF" if use_rf else None)
    fac_tag = _unit_tag(fac[anchor]) if anchor is not None else "unknown"

    unit_action = "none"
    if ret_tag != "unknown" and fac_tag != "unknown" and ret_tag != fac_tag:
        if ret_tag == "decimal" and fac_tag == "percent":
            fac = fac / 100.0
            unit_action = "factor_df_div_100"
        elif ret_tag == "percent" and fac_tag == "decimal":
            ret = ret / 100.0
            unit_action = "ret_df_div_100"

    factor_order = ["ALPHA"] + x_cols

    def _sig(p):
        if pd.isna(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
        return ""

    rows = []

    for tkr in ret.columns:
        d = pd.concat([ret[[tkr]].rename(columns={tkr: "RET"}), fac], axis=1, join="inner").dropna()

        if d.shape[0] < min_obs:
            tmp = pd.DataFrame(index=factor_order, columns=["loading", "std_err", "t_stat", "p_value", "ci_low", "ci_high", "sig"])
            tmp["sig"] = ""
            tmp["ticker"] = tkr
            tmp["factor"] = tmp.index
            rows.append(tmp.reset_index(drop=True).set_index(["ticker", "factor"]))
            continue

        y = d["RET"] - d["RF"] if use_rf else d["RET"]
        X = sm.add_constant(d[x_cols], has_constant="add")

        if cov_type == "HAC":
            res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
        else:
            res = sm.OLS(y, X).fit(cov_type=cov_type)

        params = res.params.rename(index={"const": "ALPHA"}).reindex(factor_order)
        bse = res.bse.rename(index={"const": "ALPHA"}).reindex(factor_order)
        tvals = res.tvalues.rename(index={"const": "ALPHA"}).reindex(factor_order)
        pvals = res.pvalues.rename(index={"const": "ALPHA"}).reindex(factor_order)

        ci = res.conf_int()
        ci.index = ci.index.map(lambda x: "ALPHA" if x == "const" else x)
        ci = ci.reindex(factor_order)
        ci.columns = ["ci_low", "ci_high"]

        tmp = pd.DataFrame({
            "loading": params,
            "std_err": bse,
            "t_stat": tvals,
            "p_value": pvals,
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"]
        })
        tmp["sig"] = tmp["p_value"].map(_sig)
        tmp["ticker"] = tkr
        tmp["factor"] = tmp.index
        rows.append(tmp.reset_index(drop=True).set_index(["ticker", "factor"]))

    loadings_mi = pd.concat(rows, axis=0)

    full_index = pd.MultiIndex.from_product(
        [ret.columns.tolist(), factor_order],
        names=["ticker", "factor"]
    )
    loadings_mi = loadings_mi.reindex(full_index)

    loadings_wide = (
        loadings_mi["loading"]
        .unstack("ticker")
        .reindex(index=factor_order, columns=ret.columns.tolist())
    )

    loadings_mi.attrs["selected_factors"] = x_cols
    loadings_mi.attrs["use_rf"] = use_rf
    loadings_mi.attrs["ret_units_detected"] = ret_tag
    loadings_mi.attrs["factor_units_detected"] = fac_tag
    loadings_mi.attrs["unit_action"] = unit_action

    return loadings_mi, loadings_wide

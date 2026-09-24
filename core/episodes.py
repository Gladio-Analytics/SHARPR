import numpy as np
import pandas as pd
from .metrics import _as_df, drawdown_series


def _drawdown_spells_one(dd_s):
    dd_s = dd_s.dropna()
    if dd_s.empty:
        return pd.DataFrame(columns=[
            "Start", "Valley", "End",
            "Max. Drawdown", "Depth",
            "Recovered",
            "Underwater Days", "Calendar Days",
            "Peak→Valley Days",
            "Valley→Recovery Days",
            "Valley→End Days",
        ])

    is_dd = dd_s < 0
    starts = is_dd & (~is_dd.shift(1, fill_value=False))
    ends = (~is_dd) & (is_dd.shift(1, fill_value=False))

    start_idx = list(dd_s.index[starts])
    end_idx = list(dd_s.index[ends])

    if is_dd.iloc[0]:
        start_idx = [dd_s.index[0]] + start_idx
    if len(end_idx) < len(start_idx):
        end_idx = end_idx + [dd_s.index[-1]]

    rows = []
    for st, en in zip(start_idx, end_idx):
        seg = dd_s.loc[st:en]
        if seg.empty:
            continue

        valley = seg.idxmin()
        mdd = float(seg.min())
        depth = -mdd

        recovered = bool(dd_s.loc[en] == 0.0)

        uw_days = int(len(seg) - 1)  # trading days (daily, no gaps)
        cal_days = int((en - st).days)

        pv_days = int(len(seg.loc[:valley]) - 1)
        valley_to_end_days = int(len(seg.loc[valley:]) - 1)
        valley_to_recovery_days = valley_to_end_days if recovered else np.nan

        rows.append({
            "Start": st,
            "Valley": valley,
            "End": en,
            "Max. Drawdown": mdd,   # negative
            "Depth": depth,         # positive
            "Recovered": recovered,
            "Underwater Days": uw_days,
            "Calendar Days": cal_days,
            "Peak→Valley Days": pv_days,
            "Valley→Recovery Days": valley_to_recovery_days,
            "Valley→End Days": valley_to_end_days,
        })

    return pd.DataFrame(rows)


def drawdown_episodes_panel(returns, top_n=5, rank_by="Depth"):
    r = _as_df(returns)
    dd, _, _ = drawdown_series(r)

    out = []
    for col in dd.columns:
        spells = _drawdown_spells_one(dd[col])
        if spells.empty:
            continue

        if rank_by == "Depth":
            spells = spells.sort_values(["Depth", "Underwater Days"], ascending=[False, False])
        elif rank_by == "Underwater Days":
            spells = spells.sort_values(["Underwater Days", "Depth"], ascending=[False, False])
        elif rank_by == "Valley→Recovery Days":
            spells = spells.sort_values(["Valley→Recovery Days", "Depth"], ascending=[False, False])
        else:
            raise ValueError("rank_by must be one of: 'Depth', 'Underwater Days', 'Valley→Recovery Days'.")

        spells = spells.head(int(top_n)).copy()
        spells.insert(0, "Episode", np.arange(1, len(spells) + 1))
        spells.insert(0, "Portfolio", col)
        out.append(spells)

    if not out:
        cols = [
            "Portfolio", "Episode", "Start", "Valley", "End",
            "Max. Drawdown", "Depth", "Recovered",
            "Underwater Days", "Calendar Days",
            "Peak→Valley Days", "Valley→Recovery Days", "Valley→End Days",
        ]
        return pd.DataFrame(columns=cols).set_index(["Portfolio", "Episode"])

    df = pd.concat(out, axis=0, ignore_index=True)

    for c in ["Start", "Valley", "End"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")

    df["Episode"] = pd.to_numeric(df["Episode"], errors="coerce").astype(int)
    df["Recovered"] = df["Recovered"].astype(bool)

    df["Max. Drawdown"] = pd.to_numeric(df["Max. Drawdown"], errors="coerce")
    df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")

    int_cols = ["Underwater Days", "Calendar Days", "Peak→Valley Days", "Valley→End Days"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["Valley→Recovery Days"] = pd.to_numeric(df["Valley→Recovery Days"], errors="coerce")

    return df.set_index(["Portfolio", "Episode"]).sort_index()


def format_episodes_panel(panel):
    df = panel.copy()
    if df.empty:
        return df

    df["Max. Drawdown"] = (df["Max. Drawdown"] * 100).round(2)
    df["Depth"] = (df["Depth"] * 100).round(2)

    if "Valley→Recovery Days" in df.columns:
        df["Valley→Recovery Days"] = df["Valley→Recovery Days"].round(0)

    return df


def recovery_rallies_panel(returns, top_n=5, rank_by="Recovery Return"):
    r = _as_df(returns)
    dd, wealth, _ = drawdown_series(r)

    out = []
    for col in r.columns:
        dd_s = dd[col].dropna()
        w_s = wealth[col].reindex(dd_s.index)

        spells = _drawdown_spells_one(dd_s)
        if spells.empty:
            continue

        def _recovery_return(row):
            valley = row["Valley"]
            end = row["End"]
            if pd.isna(valley) or pd.isna(end):
                return np.nan
            return float(w_s.loc[end] / w_s.loc[valley] - 1.0) if w_s.loc[valley] != 0 else np.nan

        def _rally_days(row):
            valley = row["Valley"]
            end = row["End"]
            if pd.isna(valley) or pd.isna(end):
                return np.nan
            return int(len(w_s.loc[valley:end]) - 1)

        spells["Recovery Return"] = spells.apply(_recovery_return, axis=1)
        spells["Rally Days"] = spells.apply(_rally_days, axis=1)

        if rank_by == "Recovery Return":
            spells = spells.sort_values(["Recovery Return", "Depth"], ascending=[False, False])
        elif rank_by == "Rally Days":
            spells = spells.sort_values(["Rally Days", "Recovery Return"], ascending=[False, False])
        else:
            raise ValueError("rank_by must be one of: 'Recovery Return', 'Rally Days'.")

        spells = spells.head(int(top_n)).copy()
        spells.insert(0, "Episode", np.arange(1, len(spells) + 1))
        spells.insert(0, "Portfolio", col)

        keep = [
            "Portfolio", "Episode",
            "Start", "Valley", "End",
            "Recovered",
            "Max. Drawdown", "Depth",
            "Underwater Days", "Peak→Valley Days", "Valley→Recovery Days",
            "Recovery Return", "Rally Days",
        ]
        spells = spells[keep]
        out.append(spells)

    if not out:
        cols = [
            "Portfolio", "Episode", "Start", "Valley", "End", "Recovered",
            "Max. Drawdown", "Depth", "Underwater Days", "Peak→Valley Days",
            "Valley→Recovery Days", "Recovery Return", "Rally Days",
        ]
        return pd.DataFrame(columns=cols).set_index(["Portfolio", "Episode"])

    df = pd.concat(out, axis=0, ignore_index=True)

    for c in ["Start", "Valley", "End"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")

    df["Episode"] = pd.to_numeric(df["Episode"], errors="coerce").astype(int)
    df["Recovered"] = df["Recovered"].astype(bool)

    num_cols = ["Max. Drawdown", "Depth", "Recovery Return"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    int_cols = ["Underwater Days", "Peak→Valley Days", "Valley→Recovery Days", "Rally Days"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.set_index(["Portfolio", "Episode"]).sort_index()


def format_recovery_rallies_panel(panel):
    df = panel.copy()
    if df.empty:
        return df
    df["Max. Drawdown"] = (df["Max. Drawdown"] * 100).round(2)
    df["Depth"] = (df["Depth"] * 100).round(2)
    df["Recovery Return"] = (df["Recovery Return"] * 100).round(2)
    for c in ["Underwater Days", "Peak→Valley Days", "Valley→Recovery Days", "Rally Days"]:
        if c in df.columns:
            df[c] = df[c].round(0)
    return df


def _bullrun_spells_one(returns_s, dd_threshold=0.10):
    """
    returns_s: pd.Series of daily returns (no gaps)
    dd_threshold: e.g. 0.10 means bull run ends when price falls 10% from peak
    """
    r = returns_s.dropna()
    if r.empty:
        return pd.DataFrame(columns=[
            "Trough", "Peak", "Trigger",
            "Bull Return", "Bull Return %",
            "Trough→Peak Days", "Trough→Peak Calendar Days",
            "Peak→Trigger Days", "Complete",
            "Pullback at Trigger %",
        ])

    w = (1.0 + r).cumprod()
    idx = w.index

    trough_t = idx[0]
    trough_v = float(w.iloc[0])

    peak_t = trough_t
    peak_v = trough_v

    rows = []

    for t, v in w.iloc[1:].items():
        v = float(v)

        # update trough (new low resets the bull run)
        if v < trough_v:
            trough_t, trough_v = t, v
            peak_t, peak_v = t, v
            continue

        # update peak
        if v > peak_v:
            peak_t, peak_v = t, v

        # check correction from peak
        dd_from_peak = (v - peak_v) / peak_v  # negative in a pullback
        if dd_from_peak <= -float(dd_threshold) and peak_t > trough_t:
            bull_ret = peak_v / trough_v - 1.0

            tp_days = int(len(w.loc[trough_t:peak_t]) - 1)
            tp_cal_days = int((peak_t - trough_t).days)

            pt_days = int(len(w.loc[peak_t:t]) - 1)
            pullback_pct = dd_from_peak * 100.0

            rows.append({
                "Trough": trough_t,
                "Peak": peak_t,
                "Trigger": t,
                "Bull Return": bull_ret,
                "Bull Return %": bull_ret * 100.0,
                "Trough→Peak Days": tp_days,
                "Trough→Peak Calendar Days": tp_cal_days,
                "Peak→Trigger Days": pt_days,
                "Complete": True,
                "Pullback at Trigger %": pullback_pct,
            })

            # start a new bull run at trigger date
            trough_t, trough_v = t, v
            peak_t, peak_v = t, v

    # final open bull run (not yet corrected by threshold)
    if peak_t > trough_t:
        bull_ret = peak_v / trough_v - 1.0
        tp_days = int(len(w.loc[trough_t:peak_t]) - 1)
        tp_cal_days = int((peak_t - trough_t).days)

        # "trigger" is sample end (not a real trigger)
        end_t = idx[-1]
        pullback_end = (float(w.iloc[-1]) - peak_v) / peak_v * 100.0

        rows.append({
            "Trough": trough_t,
            "Peak": peak_t,
            "Trigger": end_t,
            "Bull Return": bull_ret,
            "Bull Return %": bull_ret * 100.0,
            "Trough→Peak Days": tp_days,
            "Trough→Peak Calendar Days": tp_cal_days,
            "Peak→Trigger Days": int(len(w.loc[peak_t:end_t]) - 1),
            "Complete": False,
            "Pullback at Trigger %": pullback_end,
        })

    df = pd.DataFrame(rows)
    return df


def bullrun_episodes_panel(returns, top_n=5, dd_threshold=0.10, rank_by="Bull Return"):
    """
    returns: DataFrame (columns=portfolios), daily returns
    rank_by: "Bull Return" or "Trough→Peak Days"
    """
    r = _as_df(returns)

    out = []
    for col in r.columns:
        spells = _bullrun_spells_one(r[col], dd_threshold=dd_threshold)
        if spells.empty:
            continue

        if rank_by == "Bull Return":
            spells = spells.sort_values(["Bull Return", "Trough→Peak Days"], ascending=[False, False])
        elif rank_by == "Trough→Peak Days":
            spells = spells.sort_values(["Trough→Peak Days", "Bull Return"], ascending=[False, False])
        else:
            raise ValueError('rank_by must be "Bull Return" or "Trough→Peak Days".')

        spells = spells.head(int(top_n)).copy()
        spells.insert(0, "Episode", np.arange(1, len(spells) + 1))
        spells.insert(0, "Portfolio", col)
        out.append(spells)

    if not out:
        cols = [
            "Portfolio", "Episode", "Trough", "Peak", "Trigger",
            "Bull Return", "Bull Return %",
            "Trough→Peak Days", "Trough→Peak Calendar Days",
            "Peak→Trigger Days", "Complete", "Pullback at Trigger %",
        ]
        return pd.DataFrame(columns=cols).set_index(["Portfolio", "Episode"])

    df = pd.concat(out, axis=0, ignore_index=True)

    for c in ["Trough", "Peak", "Trigger"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")

    df["Episode"] = pd.to_numeric(df["Episode"], errors="coerce").astype(int)
    df["Complete"] = df["Complete"].astype(bool)

    num_cols = ["Bull Return", "Bull Return %", "Pullback at Trigger %"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    int_cols = ["Trough→Peak Days", "Trough→Peak Calendar Days", "Peak→Trigger Days"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df.set_index(["Portfolio", "Episode"]).sort_index()

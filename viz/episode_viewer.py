import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def fig_wealth_drawdown_panel_viewer_tabbed(
    returns_all,
    episodes_df=None,
    rallies_df=None,
    bullruns_df=None,
    styles=None,
    portfolio=None,
    panel_kind="drawdowns",
    include_risk_free=False,
    exclude_contains=("risk-free rate",),
    height=860,
    width=None,
    title=None,
    shade_alpha=0.10,
    right_margin=60,
    portfolio_dropdown_x=0.62,
    portfolio_dropdown_y=1.14,
    panel_tabs_x=0.78,
    panel_tabs_y=1.14,
):
    def _norm(s):
        return str(s).strip().lower()

    def _canonical_kind(kind):
        k = _norm(kind)
        if k in ("drawdowns", "drawdown", "episodes"):
            return "drawdowns"
        if k in ("rallies", "recovery", "recovery_rallies"):
            return "rallies"
        if k in ("bullruns", "bullrun", "bull"):
            return "bullruns"
        raise ValueError("panel_kind must be one of: 'drawdowns', 'rallies', 'bullruns'.")

    def _clone_trace(tr, visible=None, drop_visible=False):
        d = tr.to_plotly_json() if hasattr(tr, "to_plotly_json") else dict(tr)
        if drop_visible:
            d.pop("visible", None)
        elif visible is not None:
            d["visible"] = bool(visible)
        return go.Figure(data=[d]).data[0]

    def _button_to_dict(btn):
        return btn.to_plotly_json() if hasattr(btn, "to_plotly_json") else dict(btn)

    available = []
    if episodes_df is not None:
        available.append("drawdowns")
    if rallies_df is not None:
        available.append("rallies")
    if bullruns_df is not None:
        available.append("bullruns")

    if len(available) == 0:
        raise ValueError("Provide at least one of episodes_df, rallies_df, bullruns_df.")

    kind0 = _canonical_kind(panel_kind)
    if kind0 not in available:
        kind0 = available[0]

    child_map = {}
    payload_map = {}
    common_ports = None

    for k in available:
        child = fig_wealth_drawdown_panel_viewer(
            returns_all=returns_all,
            episodes_df=episodes_df,
            rallies_df=rallies_df,
            bullruns_df=bullruns_df,
            panel_kind=k,
            styles=styles,
            portfolio=None,
            include_risk_free=include_risk_free,
            exclude_contains=exclude_contains,
            height=height,
            width=width,
            title=title,
            shade_alpha=shade_alpha,
        )
        child_map[k] = child

        if len(child.layout.updatemenus) == 0:
            raise ValueError(f"Child figure for panel '{k}' has no dropdown menu.")

        buttons = child.layout.updatemenus[0].buttons
        by_port = {}

        for btn in buttons:
            bd = _button_to_dict(btn)
            lab = str(bd.get("label", ""))
            if _norm(lab) == "none":
                continue

            args0 = bd["args"][0]
            vis = list(args0["visible"])
            idxs = [i for i, v in enumerate(vis) if bool(v)]
            traces = [_clone_trace(child.data[i], drop_visible=True) for i in idxs]
            by_port[lab] = traces

        if len(by_port) == 0:
            raise ValueError(f"No portfolio payloads found for panel '{k}'.")

        counts = {p: len(trs) for p, trs in by_port.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Inconsistent trace count across portfolios for panel '{k}'.")

        payload_map[k] = by_port

        ports_k = list(by_port.keys())
        if common_ports is None:
            common_ports = ports_k
        else:
            common_ports = [p for p in common_ports if p in by_port]

    if common_ports is None or len(common_ports) == 0:
        raise ValueError("No common portfolios available across panel kinds.")

    if not include_risk_free and exclude_contains:
        bad = tuple(_norm(x) for x in exclude_contains)
        common_ports = [p for p in common_ports if not any(b in _norm(p) for b in bad)]

    if len(common_ports) == 0:
        raise ValueError("No portfolios available after filtering.")

    port0 = portfolio if (portfolio is not None and portfolio in common_ports) else common_ports[0]

    base_layout = child_map[kind0].layout.to_plotly_json()
    base_layout.pop("updatemenus", None)
    base_layout.pop("annotations", None)
    base_layout["shapes"] = []

    fig = go.Figure()
    fig.update_layout(**base_layout)

    block_map = {}
    title_map = {k: (title if title is not None else f"Wealth + Drawdown + {k.title()}") for k in available}

    for k in available:
        idxs = []
        for tr in payload_map[k][port0]:
            fig.add_trace(_clone_trace(tr, visible=(k == kind0)))
            idxs.append(len(fig.data) - 1)
        block_map[k] = idxs

    n_traces = len(fig.data)

    def _visible_for_kind(kind):
        vis = [False] * n_traces
        for idx in block_map[kind]:
            vis[idx] = True
        return vis

    frames = []
    for p in common_ports:
        frame_data = []
        for k in available:
            frame_data.extend([_clone_trace(tr, drop_visible=True) for tr in payload_map[k][p]])
        frames.append(
            go.Frame(
                name=str(p),
                data=frame_data,
            )
        )
    fig.frames = frames

    portfolio_buttons = [
        dict(
            label=str(p),
            method="animate",
            args=[
                [str(p)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        )
        for p in common_ports
    ]

    panel_buttons = [
        dict(
            label=k.title(),
            method="update",
            args=[
                {"visible": _visible_for_kind(k)},
                {"title": {"text": title_map[k]}},
            ],
        )
        for k in available
    ]

    fig.update_layout(
        height=int(height),
    width=width,
    title=dict(
        text=title_map[kind0],
        x=0.02,
        xanchor="left",
        y=0.98,
        yanchor="top",
    ),
    margin=dict(l=60, r=int(right_margin), t=150, b=20),
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            x=float(portfolio_dropdown_x),
            y=float(portfolio_dropdown_y),
            xanchor="left",
            yanchor="top",
            showactive=True,
            buttons=portfolio_buttons,
        ),
        dict(
            type="buttons",
            direction="right",
            x=float(panel_tabs_x),
            y=float(panel_tabs_y),
            xanchor="left",
            yanchor="top",
            showactive=True,
            buttons=panel_buttons,
        ),
    ],
    annotations=[
        dict(
            text="Portfolio",
            x=float(portfolio_dropdown_x),
            y=float(portfolio_dropdown_y) + 0.055,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        ),
        dict(
            text="Panel",
            x=float(panel_tabs_x),
            y=float(panel_tabs_y) + 0.055,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        ),
    ],
    shapes=[],
    )

    return fig


def fig_wealth_drawdown_panel_viewer(
    returns_all,
    episodes_df=None,
    rallies_df=None,
    bullruns_df=None,
    panel_kind="drawdowns",
    styles=None,
    portfolio=None,
    include_risk_free=False,
    exclude_contains=("risk-free rate",),
    height=860,
    width=None,
    title=None,
    shade_alpha=0.10
):
    def _as_df(x):
        if isinstance(x, pd.Series):
            return x.to_frame()
        return x

    def _norm(s):
        return str(s).strip().lower()

    def _hex_to_rgba(hex_color, alpha):
        if hex_color is None:
            return f"rgba(136,136,136,{alpha})"
        s = str(hex_color).strip()
        if s.startswith("rgba("):
            return s
        if s.startswith("rgb("):
            inner = s[len("rgb("):-1]
            return f"rgba({inner},{alpha})"
        if not s.startswith("#"):
            return f"rgba(136,136,136,{alpha})"
        h = s.lstrip("#")
        if len(h) == 3:
            h = "".join([c * 2 for c in h])
        if len(h) != 6:
            return f"rgba(136,136,136,{alpha})"
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def _to_dt(v):
        return pd.to_datetime(v, errors="coerce")

    def _series_wealth_dd(r):
        w = (1.0 + r).cumprod()
        dd = w / w.cummax() - 1.0
        return w, dd

    def _pick_panel(kind):
        k = _norm(kind)
        if k in ("drawdowns", "drawdown", "episodes"):
            if episodes_df is None:
                raise ValueError("panel_kind='drawdowns' but episodes_df is None.")
            return episodes_df, "drawdowns"
        if k in ("rallies", "recovery", "recovery_rallies"):
            if rallies_df is None:
                raise ValueError("panel_kind='rallies' but rallies_df is None.")
            return rallies_df, "rallies"
        if k in ("bullruns", "bullrun", "bull"):
            if bullruns_df is None:
                raise ValueError("panel_kind='bullruns' but bullruns_df is None.")
            return bullruns_df, "bullruns"
        raise ValueError("panel_kind must be one of: 'drawdowns', 'rallies', 'bullruns'.")

    def _panel_portfolios(p):
        if p is None:
            return []
        if isinstance(p.index, pd.MultiIndex) and p.index.nlevels >= 1:
            return list(pd.Index(p.index.get_level_values(0)).unique())
        if "Portfolio" in p.columns:
            return list(pd.Index(p["Portfolio"]).unique())
        return []

    def _slice_panel(p, port):
        if p is None:
            return None
        if isinstance(p.index, pd.MultiIndex) and p.index.nlevels >= 1:
            try:
                return p.xs(port, level=0, drop_level=True)
            except Exception:
                return None
        if "Portfolio" in p.columns:
            return p[p["Portfolio"] == port].copy()
        return None

    def _panel_cols(kind):
        k = _norm(kind)
        if k == "drawdowns":
            return dict(
                shade=("Start", "End"),
                peak="Start",
                trough="Valley",
                vlines=("Start", "Valley", "End"),
                table_pref=[
                    "Episode", "Start", "Valley", "End",
                    "Depth", "Max. Drawdown",
                    "Underwater Days", "Calendar Days",
                    "Peak→Valley Days", "Valley→Recovery Days", "Valley→End Days"
                ],
                date_cols=("Start", "Valley", "End")
            )
        if k == "rallies":
            return dict(
                shade=("Valley", "End"),
                peak="End",
                trough="Valley",
                vlines=("Valley", "End"),
                table_pref=[
                    "Episode", "Start", "Valley", "End",
                    "Depth", "Recovery Return", "Rally Days",
                    "Underwater Days", "Peak→Valley Days", "Valley→Recovery Days"
                ],
                date_cols=("Start", "Valley", "End")
            )
        if k == "bullruns":
            return dict(
                shade=("Trough", "Peak"),
                peak="Peak",
                trough="Trough",
                vlines=("Trough", "Peak", "Trigger"),
                trigger="Trigger",
                table_pref=[
                    "Episode", "Trough", "Peak", "Trigger", "Complete",
                    "Bull Return", "Bull Return %", "Pullback at Trigger %",
                    "Trough→Peak Days", "Trough→Peak Calendar Days", "Peak→Trigger Days"
                ],
                date_cols=("Trough", "Peak", "Trigger")
            )
        raise ValueError("bad kind")

    def _maybe_numeric(series):
        s = pd.Series(series)
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            return sn
        return s

    def _coerce_table(df_port, kind):
        if df_port is None or len(df_port) == 0:
            return pd.DataFrame({"Info": ["No episodes available."]})

        d = df_port.copy()
        if "Portfolio" in d.columns:
            d = d.drop(columns=["Portfolio"])

        if "Episode" not in d.columns:
            if d.index.name == "Episode":
                d = d.reset_index()
            else:
                d = d.reset_index().rename(columns={"index": "Episode"})

        meta = _panel_cols(kind)
        cols_pref = meta["table_pref"]
        cols = [c for c in cols_pref if c in d.columns]
        if not cols:
            cols = list(d.columns)
        d = d[cols].copy()

        date_cols = [c for c in meta.get("date_cols", ()) if c in d.columns]
        for c in date_cols:
            d[c] = pd.to_datetime(d[c], errors="coerce").dt.strftime("%Y-%m-%d")

        def _is_days_col(c):
            return "day" in _norm(c)

        def _is_percentish_col(c):
            cc = _norm(c)
            if "%" in str(c):
                return True
            return any(x in cc for x in ("drawdown", "depth", "return", "pullback"))

        def _fmt(v, col):
            if pd.isna(v):
                return ""
            if isinstance(v, str):
                return v
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, (np.floating, float)):
                if _is_days_col(col) or _norm(col) == "episode":
                    return str(int(round(v))) if np.isfinite(v) else ""
                if _is_percentish_col(col):
                    if not np.isfinite(v):
                        return ""
                    pv = (100.0 * v) if abs(v) <= 2.5 else v
                    return f"{pv:.2f}%"
                return f"{v:.4f}" if np.isfinite(v) else ""
            return str(v)

        for c in d.columns:
            ser = d[c]
            if c not in date_cols:
                ser2 = _maybe_numeric(ser)
                d[c] = ser2.map(lambda x: _fmt(x, c))
            else:
                d[c] = ser.map(lambda x: "" if pd.isna(x) else str(x))

        return d

    panel_df, kind = _pick_panel(panel_kind)
    meta = _panel_cols(kind)

    r = _as_df(returns_all).copy().sort_index()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("returns_all must have a DatetimeIndex.")
    if r.shape[1] == 0:
        raise ValueError("returns_all has no columns.")

    panel_ports = _panel_portfolios(panel_df)
    ports = [c for c in panel_ports if c in r.columns] if panel_ports else list(r.columns)

    if not include_risk_free and exclude_contains:
        bad = tuple(_norm(x) for x in exclude_contains)
        ports = [p for p in ports if not any(b in _norm(p) for b in bad)]

    if len(ports) == 0:
        raise ValueError("No portfolios available after filtering.")

    dd_min_global = None
    for p in ports:
        w, dd = _series_wealth_dd(pd.to_numeric(r[p], errors="coerce").fillna(0.0))
        m = float(np.nanmin(dd.values)) if dd.size else -0.5
        m = m if np.isfinite(m) else -0.5
        dd_min_global = m if dd_min_global is None else min(dd_min_global, m)
    dd_min_global = -0.5 if dd_min_global is None or not np.isfinite(dd_min_global) else min(dd_min_global, -1e-6)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.25, 0.20],
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]]
    )

    trace_map = {}
    shapes_map = {}

    for p in ports:
        st = style_for_series(p, styles)
        color = st.get("color", "#888888")
        dash = st.get("dash", "solid")
        width_line = st.get("width", 2.0)
        fill = _hex_to_rgba(color, shade_alpha)

        w, dd = _series_wealth_dd(pd.to_numeric(r[p], errors="coerce").fillna(0.0))

        fig.add_trace(
            go.Scatter(
                x=w.index,
                y=w.values,
                mode="lines",
                line=dict(color=color, dash=dash, width=width_line),
                visible=False,
                showlegend=False
            ),
            row=1, col=1
        )
        idx_wealth = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                line=dict(color=color, dash=dash, width=width_line),
                fill="tozeroy",
                fillcolor=fill,
                visible=False,
                showlegend=False
            ),
            row=2, col=1
        )
        idx_dd = len(fig.data) - 1

        panel_p = _slice_panel(panel_df, p)

        peaks_x, peaks_y = [], []
        troughs_x, troughs_y = [], []
        vlines = []
        rects = []

        if panel_p is not None and len(panel_p) > 0:
            d = panel_p.copy()
            if isinstance(d.index, pd.MultiIndex):
                d = d.reset_index()

            shade_a, shade_b = meta["shade"]
            peak_col = meta["peak"]
            trough_col = meta["trough"]
            vline_cols = meta.get("vlines", ())
            trig_col = meta.get("trigger", None)

            def _col(name):
                return d[name] if name in d.columns else None

            if _col(peak_col) is not None:
                for dt in _to_dt(_col(peak_col)).dropna().tolist():
                    if dt in w.index:
                        peaks_x.append(dt)
                        peaks_y.append(float(w.loc[dt]))
            if _col(trough_col) is not None:
                for dt in _to_dt(_col(trough_col)).dropna().tolist():
                    if dt in w.index:
                        troughs_x.append(dt)
                        troughs_y.append(float(w.loc[dt]))

            for c in vline_cols:
                s = _col(c)
                if s is None:
                    continue
                for dt in _to_dt(s).dropna().tolist():
                    vlines.append((dt, c))

            if trig_col is not None and _col(trig_col) is not None:
                for dt in _to_dt(_col(trig_col)).dropna().tolist():
                    vlines.append((dt, "Trigger"))

            if _col(shade_a) is not None and _col(shade_b) is not None:
                xa = _to_dt(_col(shade_a))
                xb = _to_dt(_col(shade_b))
                for x0, x1 in zip(xa.tolist(), xb.tolist()):
                    if pd.isna(x0) or pd.isna(x1):
                        continue
                    if x1 < x0:
                        x0, x1 = x1, x0
                    rects.append((x0, x1))

        fig.add_trace(
            go.Scatter(
                x=peaks_x,
                y=peaks_y,
                mode="markers",
                marker=dict(size=9, color="#16FF32"),
                visible=False,
                showlegend=False
            ),
            row=1, col=1
        )
        idx_peaks = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(
                x=troughs_x,
                y=troughs_y,
                mode="markers",
                marker=dict(size=9, color="#FF2A2A"),
                visible=False,
                showlegend=False
            ),
            row=1, col=1
        )
        idx_troughs = len(fig.data) - 1

        table_df = _coerce_table(panel_p, kind)
        header_vals = list(table_df.columns)
        cell_vals = [table_df[c].tolist() for c in table_df.columns]

        fig.add_trace(
            go.Table(
                header=dict(values=header_vals, align="left"),
                cells=dict(values=cell_vals, align="left"),
                visible=False
            ),
            row=3, col=1
        )
        idx_table = len(fig.data) - 1

        trace_map[p] = (idx_wealth, idx_dd, idx_peaks, idx_troughs, idx_table)

        shapes = []
        rect_fill = "rgba(255,0,0,0.08)" if kind == "drawdowns" else ("rgba(0,0,0,0.06)" if kind == "rallies" else "rgba(0,255,0,0.06)")
        for x0, x1 in rects:
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="y2",
                x0=x0, x1=x1,
                y0=dd_min_global, y1=0,
                fillcolor=rect_fill,
                line=dict(width=0),
                layer="below"
            ))

        for dt, c in vlines:
            c0 = _norm(str(c))
            if "trough" in c0 or "valley" in c0:
                line_col = "#FF2A2A"
            elif "peak" in c0 or c0 == "start" or "end" in c0:
                line_col = "#16FF32"
            elif "trigger" in c0:
                line_col = "#AAAAAA"
            else:
                line_col = "#BBBBBB"

            shapes.append(dict(
                type="line",
                xref="x",
                yref="paper",
                x0=dt, x1=dt,
                y0=0, y1=1,
                line=dict(color=line_col, width=1, dash="dot"),
                layer="below"
            ))

        shapes_map[p] = shapes

    vis_none = [False] * len(fig.data)

    buttons = [dict(
        label="None",
        method="update",
        args=[{"visible": vis_none}, {"shapes": [], "title": title if title is not None else ""}]
    )]

    for p in ports:
        vis = [False] * len(fig.data)
        i_wealth, i_dd, i_peaks, i_troughs, i_table = trace_map[p]
        vis[i_wealth] = True
        vis[i_dd] = True
        vis[i_peaks] = True
        vis[i_troughs] = True
        vis[i_table] = True

        ttl = title if title is not None else f"{p}: Wealth + Drawdown + {kind.title()}"

        buttons.append(dict(
            label=str(p),
            method="update",
            args=[{"visible": vis}, {"shapes": shapes_map.get(p, []), "title": ttl}]
        ))

    fig.update_layout(
        height=height,
        width=width,
        title=dict(text=title if title is not None else "", x=0.5, xanchor="center", y=0.99, yanchor="top"),
        margin=dict(t=175, r=25, l=60, b=10),
        updatemenus=[dict(type="dropdown", x=0.0, y=1.18, xanchor="left", yanchor="top", buttons=buttons, showactive=True)]
    )

    fig.update_yaxes(title_text="Wealth Index (start=1)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (decimal)", row=2, col=1, range=[dd_min_global, 0])

    if portfolio is not None and portfolio in ports:
        for b in buttons:
            if b["label"] == str(portfolio):
                fig.update(b["args"][0], **b["args"][1])
                break

    return _apply_white_chart_theme(fig)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def build_style_registry(
    ticker_names=None,
    portfolio_names=None,
    palette=None,
    palette_tickers=None,
    palette_portfolios=None,
    ticker_dash_cycle=None,
    portfolio_dash_cycle=None,
    ticker_dash="solid",
    portfolio_dash="solid",
    ticker_width=2.0,
    portfolio_width=2.4
):
    ticker_names = [] if ticker_names is None else list(ticker_names)
    portfolio_names = [] if portfolio_names is None else list(portfolio_names)

    def _norm(x):
        return str(x).strip()

    def _key(x):
        return _norm(x).lower()

    def _default_palette_long():
        return (
            px.colors.qualitative.Alphabet
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Light24
            + px.colors.qualitative.G10
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Pastel
        )

    def _parse_rgb(s):
        s = str(s).strip()
        if s.startswith("#") and len(s) == 7:
            r = int(s[1:3], 16); g = int(s[3:5], 16); b = int(s[5:7], 16)
            return r, g, b
        if s.startswith("rgb(") and s.endswith(")"):
            inside = s[4:-1].split(",")
            if len(inside) >= 3:
                r = int(float(inside[0])); g = int(float(inside[1])); b = int(float(inside[2]))
                return r, g, b
        if s.startswith("rgba(") and s.endswith(")"):
            inside = s[5:-1].split(",")
            if len(inside) >= 3:
                r = int(float(inside[0])); g = int(float(inside[1])); b = int(float(inside[2]))
                return r, g, b
        return 0, 0, 0

    def _to_hex(rgb):
        r, g, b = [max(0, min(255, int(v))) for v in rgb]
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def _blend(c_hex, to_hex="#ffffff", t=0.25):
        r1, g1, b1 = _parse_rgb(c_hex)
        r2, g2, b2 = _parse_rgb(to_hex)
        r = int(round((1 - t) * r1 + t * r2))
        g = int(round((1 - t) * g1 + t * g2))
        b = int(round((1 - t) * b1 + t * b2))
        return _to_hex((r, g, b))

    def _expand_palette(base, n_needed):
        base = list(base)
        if len(base) >= n_needed:
            return base[:n_needed]
        out = []
        cycle = 0
        while len(out) < n_needed:
            t = 0.0 if cycle == 0 else min(0.60, 0.14 * cycle)
            for c in base:
                if len(out) >= n_needed:
                    break
                out.append(c if t == 0.0 else _blend(c, "#ffffff", t=t))
            cycle += 1
        return out[:n_needed]

    base_palette = _default_palette_long() if palette is None else list(palette)

    if palette_tickers is None:
        palette_tickers = base_palette
    if palette_portfolios is None:
        palette_portfolios = base_palette

    n_t = len(ticker_names)
    n_p = len(portfolio_names)

    ticker_colors = _expand_palette(palette_tickers, n_t)
    portfolio_colors = _expand_palette(palette_portfolios, n_p)

    if ticker_dash_cycle is not None:
        ticker_dash_cycle = list(ticker_dash_cycle)
        if len(ticker_dash_cycle) == 0:
            ticker_dash_cycle = None
    if portfolio_dash_cycle is not None:
        portfolio_dash_cycle = list(portfolio_dash_cycle)
        if len(portfolio_dash_cycle) == 0:
            portfolio_dash_cycle = None

    styles = {}

    for i, name in enumerate(ticker_names):
        d = ticker_dash if ticker_dash_cycle is None else ticker_dash_cycle[i % len(ticker_dash_cycle)]
        styles[_key(name)] = dict(
            color=ticker_colors[i],
            dash=str(d),
            width=float(ticker_width),
            marker_symbol="circle",
            kind="ticker",
            name=_norm(name),
        )

    for j, name in enumerate(portfolio_names):
        d = portfolio_dash if portfolio_dash_cycle is None else portfolio_dash_cycle[j % len(portfolio_dash_cycle)]
        styles[_key(name)] = dict(
            color=portfolio_colors[j],
            dash=str(d),
            width=float(portfolio_width),
            marker_symbol="diamond",
            kind="portfolio",
            name=_norm(name),
        )

    return styles


def _as_df(x):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(x, pd.DataFrame) and not x.empty:
        y = x.copy()
        y.index = pd.to_datetime(y.index)
        return y.sort_index()
    return None


def _series_for_plot(s, cumulative=True):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if cumulative:
        return (1 + s).cumprod() - 1
    return s


def _default_palette():
    return (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.G10
    )


def _add_block(fig, df, cumulative, color, dash, width):
    if df is None:
        return
    for c in df.columns:
        y = _series_for_plot(df[c], cumulative=cumulative)
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y.values,
                mode="lines",
                name=str(c),
                line=dict(color=color, dash=dash, width=width)
            )
        )


def _style_color_lookup(label, styles, default="#888888"):
    if styles is None or not isinstance(styles, dict):
        return default
    s = str(label).strip()
    keys = (s, s.lower(), s.upper())
    for k in keys:
        d = styles.get(k)
        if isinstance(d, dict) and d.get("color"):
            return d["color"]
    # fallback: match by style entry's "name"
    slow = s.lower()
    for _, d in styles.items():
        if isinstance(d, dict):
            nm = str(d.get("name", "")).strip().lower()
            if nm == slow and d.get("color"):
                return d["color"]
    return default


def style_for_series(name, styles):
    styles = {} if styles is None else styles

    s = str(name).strip()
    low = s.lower()

    if "benchmark" in low:
        return dict(color="black", dash="solid", width=2.8, marker_symbol="circle", kind="special", name=s)
    if ("risk-free" in low) or ("risk free" in low) or ("riskfree" in low):
        return dict(color="black", dash="dot", width=2.6, marker_symbol="circle", kind="special", name=s)
    if ("current" in low) and ("portfolio" in low):
        return dict(color="#7a7a7a", dash="dash", width=2.6, marker_symbol="circle", kind="special", name=s)
    if ("custom" in low) and ("portfolio" in low):
        return dict(color="#9a9a9a", dash="dot", width=2.6, marker_symbol="circle", kind="special", name=s)

    k = low
    if k in styles and isinstance(styles[k], dict):
        return styles[k]

    for _, v in styles.items():
        if isinstance(v, dict) and str(v.get("name", "")).strip().lower() == low:
            return v

    return dict(color="#444444", dash="solid", width=2.0, marker_symbol="circle", kind="unknown", name=s)


def _white_xaxis_dict(title=None, type="category", showgrid=False):
    return dict(
        title=title,
        type=type,
        showgrid=showgrid,
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.18)",
        linecolor="rgba(0,0,0,0.35)",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        automargin=True,
    )


def _white_yaxis_dict(
    title=None,
    type=None,
    showgrid=True,
    autorange=None,
    categoryorder=None,
    categoryarray=None,
    zeroline=True,
):
    out = dict(
        title=title,
        showgrid=showgrid,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=zeroline,
        zerolinecolor="rgba(0,0,0,0.18)",
        linecolor="rgba(0,0,0,0.35)",
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
        automargin=True,
    )
    if type is not None:
        out["type"] = type
    if autorange is not None:
        out["autorange"] = autorange
    if categoryorder is not None:
        out["categoryorder"] = categoryorder
    if categoryarray is not None:
        out["categoryarray"] = categoryarray
    return out


def _apply_white_chart_theme(fig):
    if fig is None:
        return fig

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#000000"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(0,0,0,0.10)",
            borderwidth=1,
            font=dict(color="#000000"),
        ),
    )

    try:
        fig.update_annotations(font=dict(color="#000000"))
    except Exception:
        pass

    try:
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.18)",
            linecolor="rgba(0,0,0,0.35)",
            tickfont=dict(color="#000000"),
            title_font=dict(color="#000000"),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.18)",
            linecolor="rgba(0,0,0,0.35)",
            tickfont=dict(color="#000000"),
            title_font=dict(color="#000000"),
        )
    except Exception:
        pass

    try:
        fig.update_scenes(
            xaxis=dict(
                showbackground=True,
                backgroundcolor="white",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.18)",
                color="#000000",
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor="white",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.18)",
                color="#000000",
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor="white",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.18)",
                color="#000000",
            ),
        )
    except Exception:
        pass

    return fig

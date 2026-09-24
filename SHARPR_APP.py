from pathlib import Path
from io import BytesIO
import math

import pandas as pd
import streamlit as st

from data import fetch
import core as backend
import viz


APP_TITLE = "SHARPR Portfolio Analytics"
DEFAULT_CAPITAL = 10_000.0
DEFAULT_SIM_PATHS = 1000
DEFAULT_SIM_HORIZON = 252
DEFAULT_FORWARD_HORIZONS = (5, 21)
DEFAULT_FACTORS = ["MKT", "HML", "SMB", "CMA", "RMW", "MOM"]
SPECIAL_PORTFOLIO_NAMES = {"current portfolio", "custom portfolio"}


def init_state():
    defaults = {
        "validated_assets": None,
        "validated_benchmark": None,
        "validated_rfr": None,
        "ticker_messages": {},
        "universe_signature": None,
        "portfolio_signature": None,
        "last_compute_signature": None,
        "has_current_portfolio": "No",
        "weights_editor_df": pd.DataFrame(columns=["Ticker", "Weight"]),
        "results": None,
        "capital": DEFAULT_CAPITAL,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_results_state():
    st.session_state["portfolio_signature"] = None
    st.session_state["last_compute_signature"] = None
    st.session_state["results"] = None
    st.session_state["last_error"] = None


def parse_ticker_text(text):
    if text is None:
        return []
    raw = str(text).replace("\n", ",").replace(";", ",").replace("\t", ",")
    parts = [x.strip().upper() for x in raw.split(",")]
    out = []
    seen = set()
    for item in parts:
        if not item:
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def parse_single_ticker(text):
    vals = parse_ticker_text(text)
    return vals[0] if vals else None


def make_weights_editor_df(tickers, weights_dict=None):
    weights_dict = {} if weights_dict is None else dict(weights_dict)
    rows = []
    for t in tickers:
        rows.append({"Ticker": str(t), "Weight": float(weights_dict.get(t, 0.0))})
    return pd.DataFrame(rows)


def make_universe_signature(asset_tickers, benchmark_ticker, risk_free_ticker):
    return (
        tuple(asset_tickers) if asset_tickers is not None else tuple(),
        "" if benchmark_ticker is None else str(benchmark_ticker),
        "" if risk_free_ticker is None else str(risk_free_ticker),
    )


def make_portfolio_signature(has_current_portfolio, weights_dict, capital, factor_file_state):
    cap = round(float(capital), 8)
    if not has_current_portfolio:
        return ("NO", cap, factor_file_state)
    items = tuple(sorted((str(k), round(float(v), 10)) for k, v in dict(weights_dict).items()))
    return ("YES", items, cap, factor_file_state)


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def cached_check_tickers(tickers_tuple):
    tickers = list(tickers_tuple) if tickers_tuple is not None else None
    found, not_found, usable, message = fetch.check_tickers(tickers)
    return found, not_found, usable, message


@st.cache_data(show_spinner=False)
def cached_fetch_prices(tickers_tuple):
    return fetch.fetch_prices(list(tickers_tuple), tidy=False)


@st.cache_data(show_spinner=False)
def cached_fetch_benchmark(ticker):
    return fetch.fetch_benchmark(ticker, tidy=False)


@st.cache_data(show_spinner=False)
def cached_fetch_risk_free_rate(ticker):
    return fetch.fetch_risk_free_rate(ticker, tidy=False)


def load_factor_df():
    factor_path = Path(__file__).resolve().parent / "factor_df.xlsx"
    if not factor_path.exists():
        return None, "factor_df.xlsx not found. Factor loadings will be skipped."
    try:
        factor_df = pd.read_excel(factor_path, index_col=0)
        factor_df.index = pd.to_datetime(factor_df.index)
        factor_df = factor_df.sort_index()
        return factor_df, None
    except Exception as exc:
        return None, f"Could not load factor_df.xlsx: {exc}"


def factor_file_state():
    factor_path = Path(__file__).resolve().parent / "factor_df.xlsx"
    if not factor_path.exists():
        return (False, None, None)
    stat = factor_path.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


def validate_inputs(asset_text, benchmark_text, rfr_text):
    assets_in = parse_ticker_text(asset_text)
    benchmark_in = parse_single_ticker(benchmark_text)
    rfr_in = parse_single_ticker(rfr_text)

    assets_found, assets_not_found, assets_usable, assets_message = cached_check_tickers(tuple(assets_in))

    benchmark_found = []
    benchmark_not_found = []
    benchmark_message = "No benchmark provided"
    benchmark_usable = None
    if benchmark_in:
        benchmark_found, benchmark_not_found, benchmark_usable, benchmark_message = cached_check_tickers((benchmark_in,))

    rfr_found = []
    rfr_not_found = []
    rfr_message = "No risk-free ticker provided"
    rfr_usable = None
    if rfr_in:
        rfr_found, rfr_not_found, rfr_usable, rfr_message = cached_check_tickers((rfr_in,))

    payload = {
        "assets": {
            "found": assets_found,
            "not_found": assets_not_found,
            "usable": assets_usable,
            "message": assets_message,
        },
        "benchmark": {
            "found": benchmark_found,
            "not_found": benchmark_not_found,
            "usable": benchmark_usable,
            "message": benchmark_message,
        },
        "risk_free": {
            "found": rfr_found,
            "not_found": rfr_not_found,
            "usable": rfr_usable,
            "message": rfr_message,
        },
    }
    return payload


def display_ticker_messages(messages):
    if not messages:
        return

    assets_msg = messages.get("assets", {}).get("message")
    if assets_msg:
        st.write(f"**Assets:** {assets_msg}")

    benchmark_msg = messages.get("benchmark", {}).get("message")
    if benchmark_msg:
        st.write(f"**Benchmark:** {benchmark_msg}")

    rfr_msg = messages.get("risk_free", {}).get("message")
    if rfr_msg:
        st.write(f"**Risk-free:** {rfr_msg}")


def normalize_weights_editor_df(df, tickers):
    x = df.copy()
    if "Ticker" not in x.columns or "Weight" not in x.columns:
        raise ValueError("Weights table must contain Ticker and Weight columns.")
    x = x[["Ticker", "Weight"]].copy()
    x["Ticker"] = x["Ticker"].astype(str).str.strip()
    x = x.drop_duplicates(subset=["Ticker"], keep="last")
    x = x.set_index("Ticker").reindex(list(tickers)).reset_index()
    x["Weight"] = pd.to_numeric(x["Weight"], errors="coerce").fillna(0.0)
    return x


def validate_current_portfolio(df, tickers, tol=1e-6):
    x = normalize_weights_editor_df(df, tickers)
    weights = {row["Ticker"]: safe_float(row["Weight"]) for _, row in x.iterrows()}
    total = sum(weights.values())
    negatives = [k for k, v in weights.items() if v < -tol]
    if negatives:
        raise ValueError(f"Weights must be non-negative. Problem tickers: {', '.join(negatives)}")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tol):
        raise ValueError(f"Current portfolio weights must sum to 1. Current sum: {total:.6f}")
    return weights, total, x


def portfolio_columns_only(df):
    if df is None or df.empty:
        return []
    cols = []
    for c in df.columns:
        low = str(c).strip().lower()
        if "benchmark" in low:
            continue
        if "risk-free" in low or "risk free" in low or "riskfree" in low:
            continue
        cols.append(str(c))
    return cols


def optimized_portfolio_columns(df):
    if df is None or df.empty:
        return []
    cols = []
    for c in df.columns:
        low = str(c).strip().lower()
        if low in SPECIAL_PORTFOLIO_NAMES:
            continue
        cols.append(str(c))
    return cols


def run_pipeline(asset_tickers, benchmark_ticker=None, risk_free_ticker=None, starting_weights=None, capital=DEFAULT_CAPITAL):
    benchmark = cached_fetch_benchmark(benchmark_ticker) if benchmark_ticker else None
    risk_free_rate = cached_fetch_risk_free_rate(risk_free_ticker) if risk_free_ticker else None
    prices = cached_fetch_prices(tuple(asset_tickers))
    prices = backend.ffill_prices(prices)

    factor_df, factor_warning = load_factor_df()

    benchmark_ret = backend.to_returns(benchmark)
    risk_free_ret = backend.to_returns(risk_free_rate)
    asset_returns = backend.to_returns(prices)
    asset_returns, benchmark_ret, risk_free_ret = backend.cut_complete_asset_sample(asset_returns, benchmark_ret, risk_free_ret)

    cov_matrix = backend.get_cov_matrix(asset_returns)
    corr_matrix = backend.get_corr_matrix(asset_returns)
    sample_means = backend.get_sample_means(asset_returns)
    mean_of_sample_means = backend.get_mean_of_sample_means(asset_returns)

    current_portfolio_weights = None
    current_portfolio_return = None
    if starting_weights is not None:
        current_portfolio_weights = backend.create_current_portfolio_weights(
            starting_weights=starting_weights,
            returns=asset_returns,
        )
        current_portfolio_return = backend.calculate_drifting_portfolio_return(
            current_portfolio_weights,
            asset_returns,
            "Current Portfolio",
        )

    custom_portfolio_weights = None
    custom_portfolio_return = None

    weights_all, info = backend.run_all_optimizers(
        asset_returns=asset_returns,
        rfr=risk_free_ret,
        beta=0.95,
        bound=1.0,
    )

    weights_all = backend.add_weights_to_weights_all(
        weights_all,
        current_portfolio_weights=current_portfolio_weights,
        custom_portfolio_weights=custom_portfolio_weights,
    )

    portfolio_returns = backend.portfolio_returns_from_weights_all(weights_all, asset_returns)
    returns_all = backend.add_reference_returns(
        portfolio_returns,
        current_portfolio_return=current_portfolio_return,
        custom_portfolio_return=custom_portfolio_return,
        benchmark=benchmark_ret,
        risk_free_rate=risk_free_ret,
    )

    metrics_df = backend.evaluate_portfolio_returns_all(
        portfolio_returns_all=returns_all,
        asset_returns=asset_returns,
        weights_all=weights_all,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_ret,
        beta=0.95,
    )

    rc_pct_all = backend.risk_contribution_pct_all(weights_all, cov_matrix)
    sc_pct_all_heur = backend.sharpe_contribution_pct_all(
        weights_all=weights_all,
        er=sample_means,
        cov_matrix=cov_matrix,
        rfr=risk_free_ret,
        rfr_mode="mean",
        rfr_index=asset_returns.index,
        mode="heuristic",
    )

    if factor_df is not None:
        loadings_mi, loadings_wide = backend.calc_loadings(
            returns_all,
            factor_df,
            factors=DEFAULT_FACTORS,
            use_rf=True,
        )
    else:
        loadings_mi = pd.DataFrame()
        loadings_wide = pd.DataFrame()

    stats, dd_info = backend.calc_port_stats(
        returns=returns_all,
        benchmark=benchmark_ret,
        rfr=risk_free_ret,
        required_return_annual=0.0,
        periods=252,
        var_level=5,
        modified_var=True,
    )

    total_df = backend.build_total_performance_metrics_df(
        metrics_df=metrics_df,
        stats=stats,
        prefer="stats",
    )

    episodes = backend.drawdown_episodes_panel(returns_all, top_n=5, rank_by="Depth")
    episodes_fmt = backend.format_episodes_panel(episodes)

    rallies = backend.recovery_rallies_panel(returns_all, top_n=5, rank_by="Recovery Return")
    rallies_fmt = backend.format_recovery_rallies_panel(rallies)

    bullruns = backend.bullrun_episodes_panel(
        returns_all,
        top_n=5,
        dd_threshold=0.10,
        rank_by="Bull Return",
    )

    summary = backend.active_metrics_summary(
        returns_all=returns_all,
        benchmark=benchmark_ret,
        annualization=252,
    )
    active_all = backend.active_return_series_all(
        returns_all=returns_all,
        benchmark=benchmark_ret,
        risk_free=risk_free_ret,
    )
    effective_bets = backend.effective_n_bets_all(weights_all, cov_matrix, normalize_weights=True)
    effective_holdings = backend.effective_n_holdings_all(weights_all, normalize=True)
    total_plus_summary = backend.add_to_calc_stats(
        total_df=total_df,
        summary=summary,
        effective_holdings=effective_holdings,
        effective_bets=effective_bets,
        summary_row_prefix="",
    )

    weights_df, dollars_df, delta_df, panel_df = backend.weights_to_dollar_panel(
        weights_all,
        capital=capital,
    )

    output_dict = backend.make_portfolio_output_dict(panel_df, total_plus_summary, loadings_wide)
    excel_bytes, excel_filename = backend.export_output_dict_to_excel(
        output_dict=output_dict,
        tickers=asset_tickers,
        for_streamlit=True,
    )

    wealth_levels = backend.get_wealth_level(returns_all, start_value=1.0)

    styles = viz.build_style_registry(
        ticker_names=list(asset_returns.columns),
        portfolio_names=list(returns_all.columns),
    )

    return {
        "prices": prices,
        "benchmark": benchmark_ret,
        "risk_free_rate": risk_free_ret,
        "asset_returns": asset_returns,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "sample_means": sample_means,
        "mean_of_sample_means": mean_of_sample_means,
        "current_portfolio_weights": current_portfolio_weights,
        "current_portfolio_return": current_portfolio_return,
        "weights_all": weights_all,
        "portfolio_returns": portfolio_returns,
        "returns_all": returns_all,
        "metrics_df": metrics_df,
        "rc_pct_all": rc_pct_all,
        "sc_pct_all_heur": sc_pct_all_heur,
        "loadings_mi": loadings_mi,
        "loadings_wide": loadings_wide,
        "stats": stats,
        "dd_info": dd_info,
        "total_df": total_df,
        "episodes": episodes,
        "episodes_fmt": episodes_fmt,
        "rallies": rallies,
        "rallies_fmt": rallies_fmt,
        "bullruns": bullruns,
        "summary": summary,
        "active_all": active_all,
        "effective_bets": effective_bets,
        "effective_holdings": effective_holdings,
        "total_plus_summary": total_plus_summary,
        "weights_df": weights_df,
        "dollars_df": dollars_df,
        "delta_df": delta_df,
        "panel_df": panel_df,
        "output_dict": output_dict,
        "excel_bytes": excel_bytes,
        "excel_filename": excel_filename,
        "wealth_levels": wealth_levels,
        "styles": styles,
        "optimizer_info": info,
        "factor_warning": factor_warning,
    }


def render_overview(results):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Assets", len(results["asset_returns"].columns))
    with col2:
        st.metric("Portfolios", len(results["returns_all"].columns))
    with col3:
        st.metric("Observations", len(results["asset_returns"]))

    st.subheader("Performance Statistics")
    st.dataframe(results["total_plus_summary"], width="stretch")

    st.download_button(
        "Download Excel report",
        data=results["excel_bytes"],
        file_name=results["excel_filename"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )


def render_returns_tab(results):
    fig_assets = viz.plot_return_panel(
        asset_returns=results["asset_returns"],
        benchmark=results["benchmark"],
        risk_free_rate=results["risk_free_rate"],
        custom_portfolio=None,
        current_portfolio=results["current_portfolio_return"],
        cumulative=True,
        title="Cumulative Returns of Portfolio Assets",
        style_registry=results["styles"],
    )
    st.plotly_chart(fig_assets, width="stretch")

    fig_portfolios = viz.plot_cum_returns_and_drawdowns(
        results["returns_all"],
        title="Cumulative Returns and Drawdowns of Portfolios",
        style_registry=results["styles"],
    )
    st.plotly_chart(fig_portfolios, width="stretch")

    default_forward_portfolio = "Max. Sharpe Ratio" if "Max. Sharpe Ratio" in results["returns_all"].columns else str(results["returns_all"].columns[0])
    fig_forward = viz.fig_forward_horizon_distribution(
        results["returns_all"],
        portfolio=default_forward_portfolio,
        horizons=DEFAULT_FORWARD_HORIZONS,
    )
    st.plotly_chart(fig_forward, width="stretch")


def render_allocation_tab(results, capital):
    fig_alloc = viz.plot_allocation_risk_sharpe_switcher(
        weights_all=results["weights_all"],
        rc_pct_all=results["rc_pct_all"],
        sc_pct_all_heur=results["sc_pct_all_heur"],
        style_registry=results["styles"],
    )
    st.plotly_chart(fig_alloc, width="stretch")

    fig_realloc = viz.fig_capital_reallocation_bars(
        results["weights_all"],
        capital=capital,
        current_portfolio_name="Current Portfolio",
        show="delta_if_possible",
        as_pct=False,
        styles=results["styles"],
    )
    st.plotly_chart(fig_realloc, width="stretch")

    with st.expander("Allocation tables", expanded=False):
        st.write("**Weights**")
        st.dataframe(results["weights_df"], width="stretch")
        st.write("**Dollar allocation**")
        st.dataframe(results["dollars_df"], width="stretch")
        if results["delta_df"] is not None:
            st.write("**Rebalance delta**")
            st.dataframe(results["delta_df"], width="stretch")


def render_frontier_tab(results):
    if len(results["asset_returns"].columns) < 2:
        st.info("Efficient frontier needs at least two valid assets.")
        return
    fig = viz.plot_efficient_frontier_v2_2(
        results["asset_returns"],
        portfolio_returns_all=results["returns_all"],
        grid="risk",
        annualize=True,
    )
    st.plotly_chart(fig, width="stretch")


def render_factors_tab(results):
    if results["factor_warning"]:
        st.info(results["factor_warning"])
    if results["loadings_wide"] is None or results["loadings_wide"].empty:
        st.write("No factor loadings available.")
        return
    fig = viz.plot_loadings_wide(
        results["loadings_wide"],
        annualize_alpha=True,
        alpha_unit="decimal",
        legend_x=1.0,
        right_margin=0.6,
        style_registry=results["styles"],
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(results["loadings_wide"], width="stretch")


def render_simulation_tab(results):
    sim_candidates = optimized_portfolio_columns(results["portfolio_returns"])
    if not sim_candidates:
        st.info("No optimized portfolios available for simulation.")
        return

    default_selection = sim_candidates[: min(2, len(sim_candidates))]
    selected = st.multiselect(
        "Simulation portfolios",
        options=sim_candidates,
        default=default_selection,
        max_selections=2,
        key="sim_portfolios",
    )

    if not selected:
        st.info("Select one or two portfolios for the GBM simulation.")
        return

    fig, _ = viz.simulate_and_plot_two_gbm(
        portfolio_returns_all=results["returns_all"],
        sim_cols=selected,
        n_paths=DEFAULT_SIM_PATHS,
        horizon_days=DEFAULT_SIM_HORIZON,
        plot_paths=250,
        use_last_for_sim=756,
        plot_last_t=800,
        seed=1,
        style_registry=results["styles"],
        legend_x=1.02,
        right_margin=90,
    )
    st.plotly_chart(fig, width="stretch")


def render_episodes_tab(results):
    viewer_candidates = portfolio_columns_only(results["returns_all"])
    default_portfolio = "Max. Sharpe Ratio" if "Max. Sharpe Ratio" in viewer_candidates else viewer_candidates[0]

    col1, col2 = st.columns([1, 1])
    with col1:
        portfolio_choice = st.selectbox(
            "Portfolio",
            options=viewer_candidates,
            index=viewer_candidates.index(default_portfolio),
            key="episode_portfolio",
        )
    with col2:
        panel_kind = st.selectbox(
            "Panel",
            options=["drawdowns", "rallies", "bullruns"],
            index=0,
            key="episode_panel_kind",
        )

    fig = viz.fig_wealth_drawdown_panel_viewer_tabbed(
        returns_all=results["returns_all"],
        episodes_df=results["episodes_fmt"],
        rallies_df=results["rallies_fmt"],
        bullruns_df=results["bullruns"],
        styles=results["styles"],
        portfolio=portfolio_choice,
        panel_kind=panel_kind,
        include_risk_free=False,
        width=1200,
        height=1180,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Episode tables", expanded=False):
        st.write("**Drawdowns**")
        st.dataframe(results["episodes_fmt"], width="stretch")
        st.write("**Rallies**")
        st.dataframe(results["rallies_fmt"], width="stretch")
        st.write("**Bull runs**")
        st.dataframe(results["bullruns"], width="stretch")


def render_scorecard_tab(results):
    fig_scorecard = viz.plot_scorecard_grid_v2(
        results["total_plus_summary"],
        portfolios=None,
        initial_preset="balanced",
        title="Portfolio Scorecard",
    )
    st.plotly_chart(fig_scorecard, width="stretch")

    fig_scatter = viz.plot_gain_pain_scatter(
        results["total_plus_summary"],
        x_metric="Annualized Mean Return",
        y_metric="Annualized Volatility",
        z_metric="Expected Shortfall",
        size_metric="Diversification Ratio",
        color_metric="Sharpe Ratio",
        title="Gain vs Pain Chart",
        styles=results["styles"],
    )
    st.plotly_chart(fig_scatter, width="stretch")

    fig_bar = viz.fig_metric_dropdown_barchart(
        results["total_plus_summary"].T,
        metrics=results["total_df"].columns,
        mode="level",
        styles=results["styles"],
        title="Portfolio comparison",
    )
    st.plotly_chart(fig_bar, width="stretch")


def render_data_tab(results):
    data_options = {
        "Raw prices": results["prices"],
        "Asset returns": results["asset_returns"],
        "Portfolio returns": results["returns_all"],
        "Weights all": results["weights_all"],
        "Covariance matrix": results["cov_matrix"],
        "Correlation matrix": results["corr_matrix"],
        "Performance metrics": results["metrics_df"],
        "Total metrics": results["total_df"],
        "Output dict | Portfolio Allocation": results["output_dict"]["Portfolio Allocation"],
        "Output dict | Performance Statistics": results["output_dict"]["Performance Statistics"],
        "Output dict | Factor Loadings": results["output_dict"]["Factor Loadings"],
    }

    choice = st.selectbox("Table", options=list(data_options.keys()), key="data_table_choice")
    st.dataframe(data_options[choice], width="stretch")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Validate tickers, optionally enter a current portfolio, then run the full SHARPR pipeline.")

    with st.sidebar:
        st.subheader("Settings")
        capital = st.number_input(
            "Capital",
            min_value=0.0,
            value=float(st.session_state["capital"]),
            step=1000.0,
            format="%.2f",
        )
        st.session_state["capital"] = capital
        factor_path = Path(__file__).resolve().parent / "factor_df.xlsx"
        #if factor_path.exists():
            #st.success("factor_df.xlsx found")
        #else:
            #st.info("factor_df.xlsx not found")

    st.subheader("1. Universe")
    with st.form("universe_form", clear_on_submit=False):
        assets_default = ", ".join(st.session_state["validated_assets"] or [])
        benchmark_default = st.session_state["validated_benchmark"] or ""
        rfr_default = st.session_state["validated_rfr"] or ""

        asset_text = st.text_area(
            "Asset tickers",
            value=assets_default,
            placeholder="MSFT, AAPL, NVDA",
            height=110,
        )
        col1, col2 = st.columns(2)
        with col1:
            benchmark_text = st.text_input("Benchmark ticker (optional)", value=benchmark_default, placeholder="SPY")
        with col2:
            rfr_text = st.text_input("Risk-free ticker (optional)", value=rfr_default, placeholder="BIL")
        validate_clicked = st.form_submit_button("Validate tickers", width="stretch")

    if validate_clicked:
        messages = validate_inputs(asset_text, benchmark_text, rfr_text)
        valid_assets = messages["assets"]["usable"] or []
        valid_benchmark = None
        if messages["benchmark"]["usable"]:
            valid_benchmark = messages["benchmark"]["usable"][0]
        valid_rfr = None
        if messages["risk_free"]["usable"]:
            valid_rfr = messages["risk_free"]["usable"][0]

        if len(valid_assets) == 0:
            st.session_state["ticker_messages"] = messages
            st.session_state["validated_assets"] = None
            st.session_state["validated_benchmark"] = None
            st.session_state["validated_rfr"] = None
            st.session_state["universe_signature"] = None
            reset_results_state()
            st.error("No valid asset tickers were found.")
        elif len(valid_assets) < 2:
            st.session_state["ticker_messages"] = messages
            st.session_state["validated_assets"] = valid_assets
            st.session_state["validated_benchmark"] = valid_benchmark
            st.session_state["validated_rfr"] = valid_rfr
            st.session_state["universe_signature"] = make_universe_signature(valid_assets, valid_benchmark, valid_rfr)
            reset_results_state()
            st.error("At least two valid asset tickers are needed for the full app.")
        else:
            new_signature = make_universe_signature(valid_assets, valid_benchmark, valid_rfr)
            universe_changed = new_signature != st.session_state["universe_signature"]

            st.session_state["ticker_messages"] = messages
            st.session_state["validated_assets"] = valid_assets
            st.session_state["validated_benchmark"] = valid_benchmark
            st.session_state["validated_rfr"] = valid_rfr
            st.session_state["universe_signature"] = new_signature

            if universe_changed:
                st.session_state["weights_editor_df"] = make_weights_editor_df(valid_assets)
                st.session_state["has_current_portfolio"] = "No"
                reset_results_state()

            st.success("Ticker validation complete.")

    display_ticker_messages(st.session_state.get("ticker_messages", {}))

    if st.session_state["validated_assets"]:
        st.subheader("2. Current portfolio")
        with st.form("portfolio_form", clear_on_submit=False):
            portfolio_choice = st.radio(
                "Do you have a current portfolio?",
                options=["No", "Yes"],
                index=0 if st.session_state["has_current_portfolio"] == "No" else 1,
                horizontal=True,
            )

            edited_df = st.session_state["weights_editor_df"]
            if portfolio_choice == "Yes":
                edited_df = st.data_editor(
                    st.session_state["weights_editor_df"],
                    width="stretch",
                    hide_index=True,
                    num_rows="fixed",
                    disabled=["Ticker"],
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, step=0.01, format="%.6f"),
                    },
                    key="weights_editor_widget",
                )

            submitted = st.form_submit_button("Submit portfolio and run analysis", width="stretch")

        if portfolio_choice == "Yes":
            preview_df = normalize_weights_editor_df(edited_df, st.session_state["validated_assets"])
            current_sum = pd.to_numeric(preview_df["Weight"], errors="coerce").fillna(0.0).sum()
            st.write(f"Current weight sum: **{current_sum:.6f}**")

        if submitted:
            try:
                st.session_state["has_current_portfolio"] = portfolio_choice
                st.session_state["weights_editor_df"] = normalize_weights_editor_df(edited_df, st.session_state["validated_assets"])

                weights_dict = None
                if portfolio_choice == "Yes":
                    weights_dict, _, normalized_df = validate_current_portfolio(
                        st.session_state["weights_editor_df"],
                        st.session_state["validated_assets"],
                    )
                    st.session_state["weights_editor_df"] = normalized_df

                p_signature = make_portfolio_signature(
                    has_current_portfolio=(portfolio_choice == "Yes"),
                    weights_dict=weights_dict or {},
                    capital=st.session_state["capital"],
                    factor_file_state=factor_file_state(),
                )
                compute_signature = (st.session_state["universe_signature"], p_signature)

                if st.session_state["results"] is not None and compute_signature == st.session_state["last_compute_signature"]:
                    st.success("Using cached session results.")
                else:
                    with st.spinner("Running SHARPR analytics..."):
                        results = run_pipeline(
                            asset_tickers=st.session_state["validated_assets"],
                            benchmark_ticker=st.session_state["validated_benchmark"],
                            risk_free_ticker=st.session_state["validated_rfr"],
                            starting_weights=weights_dict,
                            capital=st.session_state["capital"],
                        )
                    st.session_state["results"] = results
                    st.session_state["portfolio_signature"] = p_signature
                    st.session_state["last_compute_signature"] = compute_signature
                    st.success("Analysis complete.")
            except Exception as exc:
                st.session_state["last_error"] = str(exc)
                st.error(str(exc))

    if st.session_state.get("last_error") and st.session_state["results"] is None:
        st.warning(st.session_state["last_error"])

    results = st.session_state.get("results")
    if results is not None:
        st.subheader("3. Results")
        tabs = st.tabs(
            [
                "Overview",
                "Returns",
                "Allocation",
                "Frontier",
                "Factors",
                "Simulation",
                "Episodes",
                "Scorecard",
                "Data",
            ]
        )

        with tabs[0]:
            render_overview(results)
        with tabs[1]:
            render_returns_tab(results)
        with tabs[2]:
            render_allocation_tab(results, st.session_state["capital"])
        with tabs[3]:
            render_frontier_tab(results)
        with tabs[4]:
            render_factors_tab(results)
        with tabs[5]:
            render_simulation_tab(results)
        with tabs[6]:
            render_episodes_tab(results)
        with tabs[7]:
            render_scorecard_tab(results)
        with tabs[8]:
            render_data_tab(results)


if __name__ == "__main__":
    main()
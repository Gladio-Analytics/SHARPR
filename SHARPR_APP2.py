import pandas as pd
import streamlit as st

import core as backend
import viz

from SHARPR_APP import (
    DEFAULT_CAPITAL,
    DEFAULT_FACTORS,
    DEFAULT_FORWARD_HORIZONS,
    make_weights_editor_df,
    cached_fetch_prices,
    cached_fetch_benchmark,
    cached_fetch_risk_free_rate,
    load_factor_df,
    factor_file_state,
    validate_inputs,
    display_ticker_messages,
    normalize_weights_editor_df,
    validate_current_portfolio,
    render_overview,
    render_allocation_tab,
    render_frontier_tab,
    render_factors_tab,
    render_simulation_tab,
    render_scorecard_tab,
    render_data_tab,
)

APP_TITLE = "SHARPR Portfolio Analytics (v2)"

# (internal key, label shown to the user, backend optimizer column name or None for the two special modes)
GOALS = [
    ("max_diversification", "Maximize Diversification", "Max. Diversification"),
    ("max_sharpe", "Maximize Risk-Adjusted Excess Return", "Max. Sharpe Ratio"),
    ("min_variance", "Minimize Volatility", "Min. Variance"),
    ("risk_parity", "Balance Risk Equally", "Risk Parity"),
    ("min_es", "Minimize Extreme Losses", "Min. Expected Shortfall"),
    ("min_drawdown", "Minimize Drawdowns", "Min. Drawdown"),
    ("compare_all", "I want to compare all options", None),
    ("compare_current_vs_benchmark", "I want to compare my Current Portfolio with the Benchmark", None),
]
GOAL_METHOD_BY_KEY = {key: method for key, _, method in GOALS}
GOAL_LABEL_BY_KEY = {key: label for key, label, _ in GOALS}
GOAL_RADIO_KEY = "goal_radio"
STRATEGY_NAMES = [m for _, _, m in GOALS if m is not None]


def init_state():
    defaults = {
        "capital": DEFAULT_CAPITAL,
        "benchmark_ticker_text": "SPY",
        "rfr_ticker_text": "BIL",
        "validated_assets": None,
        "validated_benchmark": None,
        "validated_rfr": None,
        "ticker_messages": {},
        "use_formation_date": False,
        "formation_date": None,
        "universe_signature": None,
        "prepared": None,
        "prepared_signature": None,
        "has_current_portfolio": "No",
        "current_weights_editor_df": pd.DataFrame(columns=["Ticker", "Weight"]),
        "has_custom_portfolio": "No",
        "custom_weights_editor_df": pd.DataFrame(columns=["Ticker", "Weight"]),
        "portfolio_signature": None,
        "last_compute_signature": None,
        "results": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_prepared_state():
    st.session_state["prepared"] = None
    st.session_state["prepared_signature"] = None
    reset_results_state()


def reset_results_state():
    st.session_state["portfolio_signature"] = None
    st.session_state["last_compute_signature"] = None
    st.session_state["results"] = None
    st.session_state["last_error"] = None


def make_universe_signature(asset_tickers, benchmark_ticker, risk_free_ticker, formation_date):
    return (
        tuple(asset_tickers) if asset_tickers is not None else tuple(),
        "" if benchmark_ticker is None else str(benchmark_ticker),
        "" if risk_free_ticker is None else str(risk_free_ticker),
        "" if formation_date is None else str(formation_date),
    )


def make_portfolio_signature(current_weights, custom_weights, capital, factor_file_state_value):
    cap = round(float(capital), 8)

    def _norm(w):
        if not w:
            return None
        return tuple(sorted((str(k), round(float(v), 10)) for k, v in dict(w).items()))

    return (_norm(current_weights), _norm(custom_weights), cap, factor_file_state_value)


def prepare_universe(asset_tickers, benchmark_ticker=None, risk_free_ticker=None, formation_date=None):
    benchmark = cached_fetch_benchmark(benchmark_ticker) if benchmark_ticker else None
    risk_free_rate = cached_fetch_risk_free_rate(risk_free_ticker) if risk_free_ticker else None
    prices = cached_fetch_prices(tuple(asset_tickers))
    prices = backend.ffill_prices(prices)

    benchmark_ret = backend.to_returns(benchmark)
    risk_free_ret = backend.to_returns(risk_free_rate)
    asset_returns = backend.to_returns(prices)
    asset_returns, benchmark_ret, risk_free_ret = backend.cut_complete_asset_sample(
        asset_returns, benchmark_ret, risk_free_ret
    )

    if formation_date is not None:
        cutoff = pd.Timestamp(formation_date)
        asset_returns = asset_returns.loc[asset_returns.index >= cutoff]
        if asset_returns.empty:
            raise ValueError("Portfolio formation date leaves no return observations. Pick an earlier date.")
        asset_returns, benchmark_ret, risk_free_ret = backend.cut_complete_asset_sample(
            asset_returns, benchmark_ret, risk_free_ret
        )

    cov_matrix = backend.get_cov_matrix(asset_returns)
    corr_matrix = backend.get_corr_matrix(asset_returns)
    sample_means = backend.get_sample_means(asset_returns)
    mean_of_sample_means = backend.get_mean_of_sample_means(asset_returns)

    styles = viz.build_style_registry(
        ticker_names=list(asset_returns.columns),
        portfolio_names=[],
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
        "styles": styles,
    }


def run_full_pipeline(prepared, asset_tickers, current_weights=None, custom_weights=None, capital=DEFAULT_CAPITAL):
    """Always computes every optimizer + every downstream table. Goal-based filtering
    happens afterward, in subset_results_for_goal() — this function never changes based
    on what the user picks in the goal selector."""
    prices = prepared["prices"]
    benchmark_ret = prepared["benchmark"]
    risk_free_ret = prepared["risk_free_rate"]
    asset_returns = prepared["asset_returns"]
    cov_matrix = prepared["cov_matrix"]
    sample_means = prepared["sample_means"]

    factor_df, factor_warning = load_factor_df()

    current_portfolio_weights = None
    current_portfolio_return = None
    if current_weights is not None:
        current_portfolio_weights = backend.create_current_portfolio_weights(
            starting_weights=current_weights,
            returns=asset_returns,
        )
        current_portfolio_return = backend.calculate_drifting_portfolio_return(
            current_portfolio_weights, asset_returns, "Current Portfolio"
        )

    custom_portfolio_weights = None
    custom_portfolio_return = None
    if custom_weights is not None:
        custom_portfolio_weights = backend.create_current_portfolio_weights(
            starting_weights=custom_weights,
            returns=asset_returns,
        )
        custom_portfolio_return = backend.calculate_drifting_portfolio_return(
            custom_portfolio_weights, asset_returns, "Custom Portfolio"
        )

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
    risk_contribution_dispersion = backend.risk_contribution_dispersion_all(rc_pct_all)
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
            returns_all, factor_df, factors=DEFAULT_FACTORS, use_rf=True,
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
        metrics_df=metrics_df, stats=stats, prefer="stats",
    )

    # Computed and kept in `results` for completeness, but there is intentionally no
    # "Episodes" tab in this app's UI.
    episodes = backend.drawdown_episodes_panel(returns_all, top_n=5, rank_by="Depth")
    episodes_fmt = backend.format_episodes_panel(episodes)
    rallies = backend.recovery_rallies_panel(returns_all, top_n=5, rank_by="Recovery Return")
    rallies_fmt = backend.format_recovery_rallies_panel(rallies)
    bullruns = backend.bullrun_episodes_panel(
        returns_all, top_n=5, dd_threshold=0.10, rank_by="Bull Return",
    )

    summary = backend.active_metrics_summary(
        returns_all=returns_all, benchmark=benchmark_ret, annualization=252,
    )
    active_all = backend.active_return_series_all(
        returns_all=returns_all, benchmark=benchmark_ret, risk_free=risk_free_ret,
    )
    effective_bets = backend.effective_n_bets_all(weights_all, cov_matrix, normalize_weights=True)
    effective_holdings = backend.effective_n_holdings_all(weights_all, normalize=True)
    total_plus_summary = backend.add_to_calc_stats(
        total_df=total_df,
        summary=summary,
        effective_holdings=effective_holdings,
        effective_bets=effective_bets,
        risk_contribution_dispersion=risk_contribution_dispersion,
        summary_row_prefix="",
    )

    weights_df, dollars_df, delta_df, panel_df = backend.weights_to_dollar_panel(
        weights_all, capital=capital,
    )

    output_dict = backend.make_portfolio_output_dict(panel_df, total_plus_summary, loadings_wide)
    excel_bytes, excel_filename = backend.export_output_dict_to_excel(
        output_dict=output_dict, tickers=asset_tickers, for_streamlit=True,
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
        "corr_matrix": prepared["corr_matrix"],
        "sample_means": sample_means,
        "mean_of_sample_means": prepared["mean_of_sample_means"],
        "current_portfolio_weights": current_portfolio_weights,
        "current_portfolio_return": current_portfolio_return,
        "custom_portfolio_weights": custom_portfolio_weights,
        "custom_portfolio_return": custom_portfolio_return,
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
        "risk_contribution_dispersion": risk_contribution_dispersion,
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


def _is_reference_name(name):
    low = str(name).strip().lower()
    return ("benchmark" in low) or ("risk-free" in low) or ("risk free" in low) or ("riskfree" in low)


def available_goal_keys(results):
    """Excludes 'compare Current vs Benchmark' unless a Current Portfolio was entered."""
    has_current = results.get("current_portfolio_return") is not None
    keys = []
    for key, _, _ in GOALS:
        if key == "compare_current_vs_benchmark" and not has_current:
            continue
        keys.append(key)
    return keys


def subset_results_for_goal(results, goal_key, capital, asset_tickers):
    """Pure display filter over an already-fully-computed `results` dict — no optimizer
    or fetch calls happen here, only cheap pandas selection plus re-packaging the a few
    small formatting/export helpers so every tab (including the Excel download) stays
    internally consistent with the current goal selection."""
    has_current = results.get("current_portfolio_return") is not None
    has_custom = results.get("custom_portfolio_return") is not None

    if goal_key == "compare_all":
        keep_strategies = list(STRATEGY_NAMES)
    elif goal_key == "compare_current_vs_benchmark":
        keep_strategies = []
    else:
        method_name = GOAL_METHOD_BY_KEY.get(goal_key)
        keep_strategies = [method_name] if method_name in results["weights_all"].columns else []

    keep_portfolios = list(keep_strategies)
    if has_current:
        keep_portfolios.append("Current Portfolio")
    if has_custom:
        keep_portfolios.append("Custom Portfolio")

    def filt_cols(df):
        if df is None or df.empty:
            return df
        keep = [c for c in df.columns if c in keep_portfolios or _is_reference_name(c)]
        return df[keep]

    def filt_rows(df):
        if df is None or df.empty:
            return df
        keep = [r for r in df.index if r in keep_portfolios or _is_reference_name(r)]
        return df.loc[keep]

    def filt_series(s):
        if s is None:
            return None
        keep = [i for i in s.index if i in keep_portfolios]
        return s.loc[keep]

    out = dict(results)
    out["weights_all"] = filt_cols(results["weights_all"])
    out["portfolio_returns"] = filt_cols(results["portfolio_returns"])
    out["returns_all"] = filt_cols(results["returns_all"])
    out["metrics_df"] = filt_rows(results["metrics_df"])
    out["rc_pct_all"] = filt_cols(results["rc_pct_all"])
    out["sc_pct_all_heur"] = filt_cols(results["sc_pct_all_heur"])
    out["loadings_wide"] = filt_cols(results["loadings_wide"])
    out["stats"] = filt_cols(results["stats"])
    out["total_df"] = filt_rows(results["total_df"])
    out["summary"] = filt_rows(results["summary"]) if results.get("summary") is not None else None
    out["effective_holdings"] = filt_series(results.get("effective_holdings"))
    out["effective_bets"] = filt_series(results.get("effective_bets"))
    out["risk_contribution_dispersion"] = filt_series(results.get("risk_contribution_dispersion"))

    out["total_plus_summary"] = backend.add_to_calc_stats(
        total_df=out["total_df"],
        summary=out["summary"],
        effective_holdings=out["effective_holdings"],
        effective_bets=out["effective_bets"],
        risk_contribution_dispersion=out["risk_contribution_dispersion"],
        summary_row_prefix="",
    )

    weights_df, dollars_df, delta_df, panel_df = backend.weights_to_dollar_panel(
        out["weights_all"], capital=capital,
    )
    out["weights_df"] = weights_df
    out["dollars_df"] = dollars_df
    out["delta_df"] = delta_df
    out["panel_df"] = panel_df

    out["output_dict"] = backend.make_portfolio_output_dict(
        panel_df, out["total_plus_summary"], out["loadings_wide"]
    )
    out["excel_bytes"], out["excel_filename"] = backend.export_output_dict_to_excel(
        output_dict=out["output_dict"], tickers=asset_tickers, for_streamlit=True,
    )

    out["styles"] = viz.build_style_registry(
        ticker_names=list(results["asset_returns"].columns),
        portfolio_names=list(out["returns_all"].columns),
    )

    return out


def weights_editor_block(key_prefix, label, tickers):
    """Shared Yes/No + editable-weights-table pattern, used for both the
    Current Portfolio and the Custom Portfolio."""
    has_key = f"has_{key_prefix}_portfolio"
    df_key = f"{key_prefix}_weights_editor_df"

    choice = st.radio(
        f"Do you have a {label.lower()}?",
        options=["No", "Yes"],
        index=0 if st.session_state[has_key] == "No" else 1,
        horizontal=True,
        key=f"{key_prefix}_choice_radio",
    )

    edited_df = st.session_state[df_key]
    if choice == "Yes":
        st.write(f"**{label} weights**")
        edited_df = st.data_editor(
            st.session_state[df_key],
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=["Ticker"],
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, step=0.01, format="%.6f"),
            },
            key=f"{key_prefix}_weights_editor_widget",
        )
        preview_df = normalize_weights_editor_df(edited_df, tickers)
        current_sum = pd.to_numeric(preview_df["Weight"], errors="coerce").fillna(0.0).sum()
        st.caption(f"{label} weight sum: **{current_sum:.6f}**")

    return choice, edited_df


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption(
        "Step 1: fetch and preview a universe of tickers. "
        "Step 2: optionally enter current/custom portfolio weights and run the full analysis. "
        "Then pick how you want to improve your portfolio to focus the results."
    )

    with st.sidebar:
        st.subheader("Settings")
        benchmark_ticker_text = st.text_input(
            "Benchmark ticker", value=st.session_state["benchmark_ticker_text"], placeholder="SPY"
        )
        st.session_state["benchmark_ticker_text"] = benchmark_ticker_text

        rfr_ticker_text = st.text_input(
            "Risk-free ticker", value=st.session_state["rfr_ticker_text"], placeholder="BIL"
        )
        st.session_state["rfr_ticker_text"] = rfr_ticker_text

        capital = st.number_input(
            "Capital",
            min_value=0.0,
            value=float(st.session_state["capital"]),
            step=1000.0,
            format="%.2f",
        )
        st.session_state["capital"] = capital

        use_formation_date = st.checkbox(
            "Truncate history to a start date (portfolio formation date)",
            value=st.session_state["use_formation_date"],
        )
        formation_date = None
        if use_formation_date:
            formation_date = st.date_input(
                "Portfolio formation date",
                value=st.session_state["formation_date"] or (pd.Timestamp.today() - pd.Timedelta(days=365)),
            )

    st.subheader("1. Universe & data")
    with st.form("universe_form", clear_on_submit=False):
        assets_default = ", ".join(st.session_state["validated_assets"] or [])
        asset_text = st.text_area(
            "Asset tickers",
            value=assets_default,
            placeholder="MSFT, AAPL, NVDA",
            height=110,
        )
        validate_clicked = st.form_submit_button("Validate & fetch", width="stretch")

    if validate_clicked:
        st.session_state["use_formation_date"] = use_formation_date
        st.session_state["formation_date"] = formation_date if use_formation_date else None

        messages = validate_inputs(asset_text, benchmark_ticker_text, rfr_ticker_text)
        valid_assets = messages["assets"]["usable"] or []
        valid_benchmark = messages["benchmark"]["usable"][0] if messages["benchmark"]["usable"] else None
        valid_rfr = messages["risk_free"]["usable"][0] if messages["risk_free"]["usable"] else None

        st.session_state["ticker_messages"] = messages
        st.session_state["validated_assets"] = valid_assets or None
        st.session_state["validated_benchmark"] = valid_benchmark
        st.session_state["validated_rfr"] = valid_rfr

        if len(valid_assets) < 2:
            st.session_state["universe_signature"] = None
            reset_prepared_state()
            st.error("At least two valid asset tickers are needed.")
        else:
            new_signature = make_universe_signature(
                valid_assets, valid_benchmark, valid_rfr, st.session_state["formation_date"]
            )
            universe_changed = new_signature != st.session_state["universe_signature"]
            st.session_state["universe_signature"] = new_signature

            if universe_changed:
                st.session_state["current_weights_editor_df"] = make_weights_editor_df(valid_assets)
                st.session_state["custom_weights_editor_df"] = make_weights_editor_df(valid_assets)
                st.session_state["has_current_portfolio"] = "No"
                st.session_state["has_custom_portfolio"] = "No"
                reset_prepared_state()

            try:
                with st.spinner("Fetching data..."):
                    prepared = prepare_universe(
                        asset_tickers=valid_assets,
                        benchmark_ticker=valid_benchmark,
                        risk_free_ticker=valid_rfr,
                        formation_date=st.session_state["formation_date"],
                    )
                st.session_state["prepared"] = prepared
                st.session_state["prepared_signature"] = new_signature
                st.success("Data ready.")
            except Exception as exc:
                st.session_state["prepared"] = None
                st.session_state["prepared_signature"] = None
                st.error(f"Could not prepare data: {exc}")

    display_ticker_messages(st.session_state.get("ticker_messages", {}))

    prepared = st.session_state.get("prepared")
    if prepared is not None:
        st.plotly_chart(
            viz.plot_return_panel(
                asset_returns=prepared["asset_returns"],
                benchmark=prepared["benchmark"],
                risk_free_rate=prepared["risk_free_rate"],
                cumulative=True,
                title="Cumulative Returns of Portfolio Assets",
                style_registry=prepared["styles"],
            ),
            width="stretch",
            key="preview_return_panel",
        )
        st.plotly_chart(
            viz.plot_return_correlation_heatmap(
                prepared["asset_returns"],
                benchmark=prepared["benchmark"],
                risk_free_rate=prepared["risk_free_rate"],
            ),
            width="stretch",
            key="preview_heatmap",
        )

    if prepared is not None:
        st.subheader("2. Portfolio")
        tickers = st.session_state["validated_assets"]

        with st.form("portfolio_form", clear_on_submit=False):
            current_choice, current_edited_df = weights_editor_block("current", "Current Portfolio", tickers)
            st.divider()
            custom_choice, custom_edited_df = weights_editor_block("custom", "Custom Portfolio", tickers)
            submitted = st.form_submit_button("Compute optimal portfolios", width="stretch")

        if submitted:
            try:
                st.session_state["has_current_portfolio"] = current_choice
                st.session_state["current_weights_editor_df"] = normalize_weights_editor_df(current_edited_df, tickers)
                st.session_state["has_custom_portfolio"] = custom_choice
                st.session_state["custom_weights_editor_df"] = normalize_weights_editor_df(custom_edited_df, tickers)

                current_weights = None
                if current_choice == "Yes":
                    current_weights, _, normalized_df = validate_current_portfolio(
                        st.session_state["current_weights_editor_df"], tickers,
                    )
                    st.session_state["current_weights_editor_df"] = normalized_df

                custom_weights = None
                if custom_choice == "Yes":
                    custom_weights, _, normalized_df = validate_current_portfolio(
                        st.session_state["custom_weights_editor_df"], tickers,
                    )
                    st.session_state["custom_weights_editor_df"] = normalized_df

                p_signature = make_portfolio_signature(
                    current_weights=current_weights,
                    custom_weights=custom_weights,
                    capital=st.session_state["capital"],
                    factor_file_state_value=factor_file_state(),
                )
                compute_signature = (st.session_state["prepared_signature"], p_signature)

                if st.session_state["results"] is not None and compute_signature == st.session_state["last_compute_signature"]:
                    st.success("Using cached session results.")
                else:
                    with st.spinner("Running full SHARPR analysis (all optimizers)..."):
                        results = run_full_pipeline(
                            prepared=prepared,
                            asset_tickers=tickers,
                            current_weights=current_weights,
                            custom_weights=custom_weights,
                            capital=st.session_state["capital"],
                        )
                    st.session_state["results"] = results
                    st.session_state["portfolio_signature"] = p_signature
                    st.session_state["last_compute_signature"] = compute_signature
                    st.session_state[GOAL_RADIO_KEY] = None
                    st.success("Analysis complete.")
            except Exception as exc:
                st.session_state["last_error"] = str(exc)
                st.error(str(exc))

    if st.session_state.get("last_error") and st.session_state["results"] is None:
        st.warning(st.session_state["last_error"])

    results = st.session_state.get("results")
    if results is not None:
        st.subheader("3. How do you want to improve your portfolio?")
        available_keys = available_goal_keys(results)

        if st.session_state.get(GOAL_RADIO_KEY) not in available_keys:
            st.session_state[GOAL_RADIO_KEY] = None

        chosen_key = st.radio(
            "Goal",
            options=available_keys,
            index=None,
            format_func=lambda k: GOAL_LABEL_BY_KEY[k],
            key=GOAL_RADIO_KEY,
        )

        if chosen_key is None:
            st.info("Pick a goal above to see results.")
        else:
            subset = subset_results_for_goal(
                results,
                chosen_key,
                capital=st.session_state["capital"],
                asset_tickers=st.session_state["validated_assets"],
            )

            st.subheader("4. Results")
            tabs = st.tabs(
                [
                    "Overview",
                    "Returns",
                    "Allocation",
                    "Frontier",
                    "Factors",
                    "Simulation",
                    "Scorecard",
                    "Portfolio Improvements",
                    "Data",
                ]
            )

            with tabs[0]:
                render_overview(subset)
            with tabs[1]:
                fig_portfolios = viz.plot_cum_returns_and_drawdowns(
                    subset["returns_all"],
                    title="Cumulative Returns and Drawdowns of Portfolios",
                    style_registry=subset["styles"],
                )
                st.plotly_chart(fig_portfolios, width="stretch", key="results_cum_returns_dd")

                default_forward_portfolio = (
                    "Max. Sharpe Ratio" if "Max. Sharpe Ratio" in subset["returns_all"].columns
                    else str(subset["returns_all"].columns[0])
                )
                fig_forward = viz.fig_forward_horizon_distribution(
                    subset["returns_all"],
                    portfolio=default_forward_portfolio,
                    horizons=DEFAULT_FORWARD_HORIZONS,
                )
                st.plotly_chart(fig_forward, width="stretch", key="results_forward_horizon")
            with tabs[2]:
                render_allocation_tab(subset, st.session_state["capital"])
            with tabs[3]:
                render_frontier_tab(subset)
            with tabs[4]:
                render_factors_tab(subset)
            with tabs[5]:
                render_simulation_tab(subset)
            with tabs[6]:
                render_scorecard_tab(subset)
            with tabs[7]:
                comparisons, comparisons_extended = viz.build_improvement_comparisons(
                    subset["total_plus_summary"], viz.IMPROVEMENT_METRICS
                )
                if not comparisons_extended:
                    st.info(
                        "No optimizer strategy is included in the current selection "
                        "(you picked 'Compare Current vs Benchmark'). Pick a specific "
                        "improvement goal, or 'I want to compare all options', to see this chart."
                    )
                else:
                    fig_improvement = viz.plot_improvement_comparison(comparisons_extended)
                    st.plotly_chart(fig_improvement, width="stretch", key="results_improvement_comparison")
            with tabs[8]:
                render_data_tab(subset)


if __name__ == "__main__":
    main()

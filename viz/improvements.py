import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


IMPROVEMENT_METRICS = {

    "Max. Diversification": {
        "primary": [
            "Diversification Ratio",
        ],
        "secondary": [
            "Effective Holdings",
            "Effective Bets",
            "Annualized Volatility",
            "Max. Drawdown",
            "Expected Shortfall",
        ],
        "cost": [
            "Annualized Mean Return",
            "Sharpe Ratio",
            "Portfolio Turnover",
            "Tracking Error",
        ],
    },

    "Max. Sharpe Ratio": {
        "primary": [
            "Sharpe Ratio",
        ],
        "secondary": [
            "Annualized Mean Return",
            "Annualized Volatility",
            "Sortino Ratio",
            "Max. Drawdown",
            "Expected Shortfall",
        ],
        "cost": [
            "Effective Holdings",
            "Effective Bets",
            "Portfolio Turnover",
            "Tracking Error",
        ],
    },

    "Min. Variance": {
        "primary": [
            "Annualized Volatility",
        ],
        "secondary": [
            "Semi-Deviation",
            "Expected Shortfall",
            "Max. Drawdown",
            "Ulcer Index",
            "Risk of Ruin (q/p)^15",
        ],
        "cost": [
            "Annualized Mean Return",
            "Sharpe Ratio",
            "Effective Holdings",
            "Portfolio Turnover",
        ],
    },

    "Risk Parity": {
        "primary": [
            "Risk Contribution Dispersion",
        ],
        "secondary": [
            "Diversification Ratio",
            "Effective Holdings",
            "Effective Bets",
            "Annualized Volatility",
            "Max. Drawdown",
        ],
        "cost": [
            "Annualized Mean Return",
            "Sharpe Ratio",
            "Portfolio Turnover",
            "Tracking Error",
        ],
    },

    "Min. Expected Shortfall": {
        "primary": [
            "Expected Shortfall",
        ],
        "secondary": [
            "Conditional VaR (5%)",
            "Historical VaR (5%)",
            "Worst Day",
            "Semi-Deviation",
            "Max. Drawdown",
        ],
        "cost": [
            "Annualized Mean Return",
            "Sharpe Ratio",
            "Annualized Volatility",
            "Portfolio Turnover",
        ],
    },

    "Min. Drawdown": {
        "primary": [
            "Max. Drawdown",
        ],
        "secondary": [
            "Ulcer Index",
            "Average Drawdown",
            "Longest Underwater Period",
            "Average Underwater Days",
            "Expected Shortfall",
        ],
        "cost": [
            "Annualized Mean Return",
            "Sharpe Ratio",
            "Annualized Volatility",
            "Portfolio Turnover",
        ],
    },
}


def build_improvement_comparisons(total_plus_summary, improvement_metrics):
    out = {}
    out_extended = {}

    benchmark_cols = [
        c for c in total_plus_summary.columns
        if "benchmark" in str(c).lower()
    ]

    benchmark_col = benchmark_cols[0] if len(benchmark_cols) > 0 else None

    current_col = (
        "Current Portfolio"
        if "Current Portfolio" in total_plus_summary.columns
        else None
    )

    custom_col = (
        "Custom Portfolio"
        if "Custom Portfolio" in total_plus_summary.columns
        else None
    )

    for portfolio, config in improvement_metrics.items():

        if portfolio not in total_plus_summary.columns:
            continue

        improvement = config["primary"] + config["secondary"]
        tradeoff = config["cost"]
        metrics = improvement + tradeoff
        metrics = [m for m in metrics if m in total_plus_summary.index]

        cols = []
        if current_col is not None:
            cols.append(current_col)
        if custom_col is not None:
            cols.append(custom_col)
        if benchmark_col is not None:
            cols.append(benchmark_col)
        cols.append(portfolio)
        cols = list(dict.fromkeys(cols))

        df = total_plus_summary.loc[metrics, cols].copy()

        df_extended = df.copy()
        df_extended["Metric Type"] = [
            "Improvement" if metric in improvement else "Tradeoff"
            for metric in df_extended.index
        ]

        out[portfolio] = df
        out_extended[portfolio] = df_extended

    return out, out_extended


def plot_improvement_comparison(improvement_comparisons_extended):

    strategies = list(improvement_comparisons_extended.keys())

    if len(strategies) == 0:
        raise ValueError("improvement_comparisons_extended is empty.")

    strategy_info = {}
    max_improvement = 0
    max_tradeoff = 0

    for strategy in strategies:
        df = improvement_comparisons_extended[strategy].copy()
        metric_type = df["Metric Type"]
        values = df.drop(columns="Metric Type")
        primary_metric = df.index[0]

        improvement_metrics = [m for m in df.index[1:] if metric_type.loc[m] == "Improvement"]
        tradeoff_metrics = [m for m in df.index[1:] if metric_type.loc[m] == "Tradeoff"]

        strategy_info[strategy] = {
            "df": df,
            "values": values,
            "primary": primary_metric,
            "improvement": improvement_metrics,
            "tradeoff": tradeoff_metrics,
        }

        max_improvement = max(max_improvement, len(improvement_metrics))
        max_tradeoff = max(max_tradeoff, len(tradeoff_metrics))

    n_metric_rows = max(max_improvement, max_tradeoff)
    n_rows = 2 + n_metric_rows

    row_heights = [0.24] + [0.08] + [0.68 / n_metric_rows] * n_metric_rows
    specs = [[{"colspan": 2}, None]] + [[None, None]] + [[{}, {}] for _ in range(n_metric_rows)]

    fig = make_subplots(
        rows=n_rows, cols=2, specs=specs, row_heights=row_heights,
        vertical_spacing=0.065, horizontal_spacing=0.15,
    )

    strategy_trace_indices = {}

    for strategy_i, strategy in enumerate(strategies):
        info = strategy_info[strategy]
        values = info["values"]
        primary_metric = info["primary"]
        improvement_metrics = info["improvement"]
        tradeoff_metrics = info["tradeoff"]
        visible = strategy_i == 0
        strategy_trace_indices[strategy] = []

        primary_values = pd.to_numeric(values.loc[primary_metric], errors="coerce").dropna()
        fig.add_trace(
            go.Bar(
                x=primary_values.values, y=primary_values.index, orientation="h",
                text=[f"{x:.3f}" for x in primary_values.values],
                textposition="outside", showlegend=False, visible=visible,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
        strategy_trace_indices[strategy].append(len(fig.data) - 1)

        for i, metric in enumerate(improvement_metrics):
            metric_values = pd.to_numeric(values.loc[metric], errors="coerce").dropna()
            fig.add_trace(
                go.Scatter(
                    x=metric_values.values, y=metric_values.index, mode="markers+text",
                    text=[f"{x:.3f}" for x in metric_values.values],
                    textposition="middle right", marker=dict(size=10),
                    showlegend=False, visible=visible,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ),
                row=i + 3, col=1,
            )
            strategy_trace_indices[strategy].append(len(fig.data) - 1)

        for i, metric in enumerate(tradeoff_metrics):
            metric_values = pd.to_numeric(values.loc[metric], errors="coerce").dropna()
            fig.add_trace(
                go.Scatter(
                    x=metric_values.values, y=metric_values.index, mode="markers+text",
                    text=[f"{x:.3f}" for x in metric_values.values],
                    textposition="middle right", marker=dict(size=10),
                    showlegend=False, visible=visible,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ),
                row=i + 3, col=2,
            )
            strategy_trace_indices[strategy].append(len(fig.data) - 1)

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=True, zeroline=False)

    fig.update_layout(
        height=420 + 165 * n_metric_rows,
        margin=dict(l=110, r=80, t=150, b=50),
        title=dict(text="Portfolio Improvement Comparison", x=0.5),
        hovermode="closest",
    )

    def make_annotations(strategy):
        info = strategy_info[strategy]
        primary_metric = info["primary"]
        improvement_metrics = info["improvement"]
        tradeoff_metrics = info["tradeoff"]
        annotations = []

        primary_subplot = fig.get_subplot(1, 1)
        annotations.append(dict(
            text=f"<b>{primary_metric}</b>", x=0.5,
            y=primary_subplot.yaxis.domain[1] + 0.015,
            xref="paper", yref="paper", xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(size=17),
        ))

        first_left = fig.get_subplot(3, 1)
        first_right = fig.get_subplot(3, 2)
        section_y = max(first_left.yaxis.domain[1], first_right.yaxis.domain[1]) + 0.045

        annotations.append(dict(
            text="<b>IMPROVEMENT METRICS</b>", x=first_left.xaxis.domain[0], y=section_y,
            xref="paper", yref="paper", xanchor="left", yanchor="bottom",
            showarrow=False, font=dict(size=13),
        ))
        annotations.append(dict(
            text="<b>TRADE-OFF METRICS</b>", x=first_right.xaxis.domain[0], y=section_y,
            xref="paper", yref="paper", xanchor="left", yanchor="bottom",
            showarrow=False, font=dict(size=13),
        ))

        for i, metric in enumerate(improvement_metrics):
            subplot = fig.get_subplot(i + 3, 1)
            annotations.append(dict(
                text=f"<b>{metric}</b>", x=subplot.xaxis.domain[0],
                y=subplot.yaxis.domain[1] + 0.012,
                xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                showarrow=False, font=dict(size=14),
            ))

        for i, metric in enumerate(tradeoff_metrics):
            subplot = fig.get_subplot(i + 3, 2)
            annotations.append(dict(
                text=f"<b>{metric}</b>", x=subplot.xaxis.domain[0],
                y=subplot.yaxis.domain[1] + 0.012,
                xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                showarrow=False, font=dict(size=14),
            ))

        return annotations

    initial_strategy = strategies[0]
    fig.update_layout(annotations=make_annotations(initial_strategy))

    buttons = []
    for strategy in strategies:
        visible = [False] * len(fig.data)
        for trace_i in strategy_trace_indices[strategy]:
            visible[trace_i] = True
        buttons.append(dict(
            label=strategy, method="update",
            args=[{"visible": visible}, {"annotations": make_annotations(strategy)}],
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", direction="down", x=0.0, y=1.11,
            xanchor="left", yanchor="top", buttons=buttons, showactive=True,
        )]
    )

    fig.add_annotation(
        text="<b>Optimization Strategy:</b>", x=0.0, y=1.135,
        xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=13),
    )

    return fig

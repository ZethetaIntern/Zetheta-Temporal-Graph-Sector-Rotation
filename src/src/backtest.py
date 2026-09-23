import numpy as np
import pandas as pd

def calculate_performance_metrics(portfolio_returns, risk_free_rate=0.06):
    """
    Computes key quantitative finance metrics: Annualized Return,
    Annualized Volatility, Sharpe Ratio, and Maximum Drawdown.
    """
    # Daily to Annualized scaling (assuming 252 trading days)
    ann_return = np.mean(portfolio_returns) * 252
    ann_volatility = np.std(portfolio_returns) * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (ann_return - risk_free_rate) / (ann_volatility + 1e-8)

    # Maximum Drawdown
    cumulative = (1 + pd.Series(portfolio_returns)).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Annualized Return": f"{ann_return * 100:.2f}%",
        "Annualized Volatility": f"{ann_volatility * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Maximum Drawdown": f"{max_drawdown * 100:.2f}%"
    }

def run_backtest(returns_df, weight_history):
    """
    Backtests the rotation strategy over historical sector returns.
    returns_df: DataFrame of daily log returns for each sector
    weight_history: Array or list of weight allocations over time
    """
    # Align lengths and compute strategy daily returns
    strat_returns = (returns_df.values[1:] * weight_history[:-1]).sum(axis=1)

    # Benchmark: Equal-weighted portfolio
    benchmark_returns = returns_df.values[1:].mean(axis=1)

    strat_metrics = calculate_performance_metrics(strat_returns)
    bench_metrics = calculate_performance_metrics(benchmark_returns)

    print("--- STRATEGY PERFORMANCE ---")
    for k, v in strat_metrics.items():
        print(f"{k}: {v}")

    print("\n--- BENCHMARK (EQUALLY WEIGHTED) ---")
    for k, v in bench_metrics.items():
        print(f"{k}: {v}")

    return strat_returns, benchmark_returns

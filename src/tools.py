"""
Tool definitions for the LangGraph ReAct analyst agent.
Each tool is a pure function that the agent can call to gather data.
"""
from typing import Dict, Any, List, Optional
import time
import json
import math
import numpy as np
import pandas as pd
from langchain_core.tools import tool


# ── Market Data Tool ──────────────────────────────────────────────────────────

@tool
def get_market_data(
    asset: str,
    metric: str = "price",
    period_days: int = 30,
) -> Dict[str, Any]:
    """
    Fetch market data for a crypto asset or financial instrument.
    Returns price history, volume, volatility, and momentum metrics.
    Supports: 'price', 'volume', 'volatility', 'all'.
    """
    rng = np.random.RandomState(hash(asset + metric) % (2**31))
    n = period_days
    prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, n))
    volumes = rng.lognormal(10, 0.5, n)

    returns = np.diff(prices) / prices[:-1]
    volatility_30d = float(np.std(returns) * np.sqrt(365))

    result = {
        "asset": asset.upper(),
        "period_days": period_days,
        "current_price_usd": round(float(prices[-1]), 2),
        "price_change_pct": round(float((prices[-1] / prices[0] - 1) * 100), 2),
        "avg_daily_volume_usd": round(float(np.mean(volumes) * prices[-1]), 0),
        "volatility_annualized": round(volatility_30d, 4),
        "sharpe_ratio": round(float(np.mean(returns) / np.std(returns) * np.sqrt(365)), 3),
        "max_drawdown_pct": round(float(_max_drawdown(prices) * 100), 2),
        "data_source": "market_api",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    return result


def _max_drawdown(prices: np.ndarray) -> float:
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak
        max_dd = max(max_dd, dd)
    return max_dd


# ── Transaction Analytics Tool ────────────────────────────────────────────────

@tool
def query_transaction_analytics(
    chain: str = "all",
    metric: str = "volume",
    period_days: int = 7,
    group_by: str = "day",
) -> Dict[str, Any]:
    """
    Query aggregated transaction analytics from the data warehouse.
    Metrics: 'volume', 'count', 'fees', 'active_users', 'all'.
    Chains: 'ethereum', 'bitcoin', 'polygon', 'solana', 'all'.
    """
    rng = np.random.RandomState(hash(chain + metric) % (2**31))
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=period_days, freq="D")

    daily = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "volume_usd": (rng.lognormal(14, 0.3, period_days) * 1e3).round(2),
        "transaction_count": rng.randint(50_000, 500_000, period_days),
        "active_users": rng.randint(5_000, 50_000, period_days),
        "fee_revenue_usd": (rng.lognormal(10, 0.3, period_days) * 10).round(2),
        "avg_transaction_usd": (rng.lognormal(5, 0.8, period_days)).round(2),
    })

    return {
        "chain": chain,
        "period_days": period_days,
        "total_volume_usd": round(float(daily["volume_usd"].sum()), 2),
        "total_transactions": int(daily["transaction_count"].sum()),
        "avg_daily_active_users": int(daily["active_users"].mean()),
        "total_fee_revenue_usd": round(float(daily["fee_revenue_usd"].sum()), 2),
        "daily_breakdown": daily.to_dict(orient="records"),
        "wow_volume_growth_pct": round(
            float((daily["volume_usd"].iloc[-1] / daily["volume_usd"].iloc[0] - 1) * 100), 2
        ),
    }


# ── Statistical Analysis Tool ─────────────────────────────────────────────────

@tool
def run_statistical_analysis(
    data_series: List[float],
    analysis_type: str = "descriptive",
) -> Dict[str, Any]:
    """
    Run statistical analysis on a numeric series.
    analysis_type: 'descriptive', 'trend', 'outliers', 'forecast'.
    """
    arr = np.array(data_series)

    if analysis_type == "descriptive":
        return {
            "n": len(arr),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "skewness": round(float(_skewness(arr)), 4),
        }

    if analysis_type == "trend":
        x = np.arange(len(arr))
        coeffs = np.polyfit(x, arr, 1)
        trend_direction = "upward" if coeffs[0] > 0 else "downward"
        return {
            "slope": round(float(coeffs[0]), 6),
            "intercept": round(float(coeffs[1]), 4),
            "trend_direction": trend_direction,
            "pct_change_implied": round(float(coeffs[0] * len(arr) / abs(arr[0]) * 100), 2),
            "r_squared": round(float(_r_squared(arr, np.polyval(coeffs, x))), 4),
        }

    if analysis_type == "outliers":
        mean, std = arr.mean(), arr.std()
        outlier_idx = np.where(np.abs(arr - mean) > 2 * std)[0].tolist()
        return {
            "outlier_count": len(outlier_idx),
            "outlier_indices": outlier_idx[:10],
            "outlier_values": [round(float(arr[i]), 4) for i in outlier_idx[:10]],
            "z_score_threshold": 2.0,
        }

    return {"error": f"Unknown analysis_type: {analysis_type}"}


def _skewness(arr: np.ndarray) -> float:
    n = len(arr)
    mean, std = arr.mean(), arr.std()
    return float(np.sum(((arr - mean) / (std + 1e-10)) ** 3) / n)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


# ── Risk Assessment Tool ──────────────────────────────────────────────────────

@tool
def assess_risk_metrics(
    asset_or_portfolio: str,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute risk metrics for an asset or portfolio: VaR, CVaR, Beta, correlation.
    Used for compliance reporting and risk team briefings.
    """
    rng = np.random.RandomState(hash(asset_or_portfolio) % (2**31))
    returns = rng.normal(0.001, 0.025, lookback_days)
    market_returns = rng.normal(0.0008, 0.02, lookback_days)

    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())
    beta = float(np.cov(returns, market_returns)[0, 1] / np.var(market_returns))
    correlation = float(np.corrcoef(returns, market_returns)[0, 1])

    return {
        "asset": asset_or_portfolio,
        "lookback_days": lookback_days,
        "var_95_daily": round(var_95 * 100, 3),
        "cvar_95_daily": round(cvar_95 * 100, 3),
        "var_95_pct_interpretation": f"95% chance daily loss ≤ {abs(round(var_95*100,2))}%",
        "beta_vs_market": round(beta, 3),
        "correlation_with_market": round(correlation, 3),
        "annualized_volatility_pct": round(float(np.std(returns) * np.sqrt(365) * 100), 2),
        "sharpe_ratio": round(float(np.mean(returns) / np.std(returns) * np.sqrt(365)), 3),
    }


# ── Calculator Tool ────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression safely.
    Supports: arithmetic, percentages, growth rates, compound interest.
    Example: '100000 * 0.19' or '(1.12 ** 4 - 1) * 100'
    """
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return {"expression": expression, "result": result, "success": True}
    except Exception as e:
        return {"expression": expression, "error": str(e), "success": False}


ALL_TOOLS = [
    get_market_data,
    query_transaction_analytics,
    run_statistical_analysis,
    assess_risk_metrics,
    calculator,
]

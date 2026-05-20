"""Unit tests for the LangGraph analyst agent components."""
import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.tools import (
    get_market_data, query_transaction_analytics,
    run_statistical_analysis, assess_risk_metrics, calculator,
)
from src.report_generator import synthesize_report, _infer_title, _extract_sections
from src.memory import SessionMemory, PersistentMemory
from src.state import INITIAL_STATE


# ── Tools ─────────────────────────────────────────────────────────────────────

def test_get_market_data():
    result = get_market_data.invoke({"asset": "ethereum", "period_days": 30})
    assert "current_price_usd" in result
    assert "volatility_annualized" in result
    assert result["asset"] == "ETHEREUM"
    assert result["current_price_usd"] > 0


def test_get_market_data_deterministic():
    r1 = get_market_data.invoke({"asset": "bitcoin", "period_days": 30})
    r2 = get_market_data.invoke({"asset": "bitcoin", "period_days": 30})
    assert r1["current_price_usd"] == r2["current_price_usd"]


def test_query_transaction_analytics():
    result = query_transaction_analytics.invoke({"chain": "ethereum", "period_days": 7})
    assert "total_volume_usd" in result
    assert "total_transactions" in result
    assert result["period_days"] == 7
    assert len(result["daily_breakdown"]) == 7


def test_run_statistical_analysis_descriptive():
    data = [float(x) for x in range(1, 11)]
    result = run_statistical_analysis.invoke({"data_series": data, "analysis_type": "descriptive"})
    assert result["mean"] == pytest.approx(5.5, rel=0.01)
    assert result["n"] == 10


def test_run_statistical_analysis_trend():
    data = [float(i) * 1.05 for i in range(1, 21)]
    result = run_statistical_analysis.invoke({"data_series": data, "analysis_type": "trend"})
    assert "slope" in result
    assert result["trend_direction"] == "upward"


def test_run_statistical_analysis_outliers():
    data = [1.0, 1.1, 0.9, 1.0, 100.0, 1.1, 0.8]
    result = run_statistical_analysis.invoke({"data_series": data, "analysis_type": "outliers"})
    assert result["outlier_count"] >= 1


def test_assess_risk_metrics():
    result = assess_risk_metrics.invoke({"asset_or_portfolio": "ETH", "lookback_days": 30})
    assert "var_95_daily" in result
    assert "cvar_95_daily" in result
    assert "beta_vs_market" in result
    assert "sharpe_ratio" in result


def test_calculator_basic():
    result = calculator.invoke({"expression": "100000 * 0.19"})
    assert result["success"] is True
    assert result["result"] == pytest.approx(19000.0)


def test_calculator_compound_interest():
    result = calculator.invoke({"expression": "(1.12 ** 4 - 1) * 100"})
    assert result["success"] is True
    assert result["result"] == pytest.approx(57.35, rel=0.01)


def test_calculator_invalid_expression():
    result = calculator.invoke({"expression": "import os; os.system('rm -rf /')"})
    assert result["success"] is False


def test_calculator_math_functions():
    result = calculator.invoke({"expression": "sqrt(144)"})
    assert result["success"] is True
    assert result["result"] == pytest.approx(12.0)


# ── Report generator ──────────────────────────────────────────────────────────

def test_infer_title_ethereum():
    title = _infer_title("Analyze ethereum market performance")
    assert "Ethereum" in title


def test_infer_title_risk():
    title = _infer_title("What are the risk metrics for this portfolio?")
    assert "Risk" in title


def test_synthesize_report_empty_context():
    report = synthesize_report(
        query="What is the market doing?",
        data_context={},
        messages=[],
        citations=[],
    )
    assert isinstance(report, str)
    assert len(report) > 100
    assert "Executive Summary" in report


def test_synthesize_report_with_data():
    market_data = get_market_data.invoke({"asset": "ethereum"})
    risk_data = assess_risk_metrics.invoke({"asset_or_portfolio": "ETH"})
    txn_data = query_transaction_analytics.invoke({"chain": "ethereum"})

    report = synthesize_report(
        query="Ethereum analysis",
        data_context={
            "get_market_data": market_data,
            "assess_risk_metrics": risk_data,
            "query_transaction_analytics": txn_data,
        },
        messages=[],
        citations=["get_market_data (tool call at iteration 1)"],
    )
    assert "ETHEREUM" in report or "Ethereum" in report
    assert "Risk" in report
    assert "Transaction" in report


# ── Memory ────────────────────────────────────────────────────────────────────

def test_session_memory_add_and_get():
    session = SessionMemory("test_session_001")
    from langchain_core.messages import HumanMessage
    session.add_messages([HumanMessage(content="Test query")])
    messages = session.get_messages()
    assert len(messages) == 1


def test_session_memory_log_tool_call():
    session = SessionMemory("test_session_002")
    session.log_tool_call("get_market_data", {"asset": "eth"}, {"price": 3000}, elapsed_ms=120.5)
    summary = session.summarize()
    assert summary["tool_calls"] == 1
    assert "get_market_data" in summary["tools_used"]


def test_session_memory_save_report():
    session = SessionMemory("test_session_003")
    session.save_report("test query", "test report", {"key": "value"})
    history = session.get_report_history()
    assert len(history) == 1
    assert history[0]["query"] == "test query"


def test_persistent_memory_save_load(tmp_path):
    memory = PersistentMemory(storage_dir=str(tmp_path))
    session = SessionMemory("persist_test_001")
    session.save_report("ethereum analysis", "## Report content", {})
    memory.save_session(session)

    loaded = memory.load_session("persist_test_001")
    assert loaded is not None
    assert loaded["session_id"] == "persist_test_001"
    assert len(loaded["reports"]) == 1


def test_persistent_memory_list_sessions(tmp_path):
    memory = PersistentMemory(storage_dir=str(tmp_path))
    for i in range(3):
        s = SessionMemory(f"session_{i:03d}")
        memory.save_session(s)
    sessions = memory.list_sessions()
    assert len(sessions) == 3


# ── State ──────────────────────────────────────────────────────────────────────

def test_initial_state_defaults():
    state = dict(INITIAL_STATE)
    assert state["iteration"] == 0
    assert state["final_report"] is None
    assert state["messages"] == []
    assert state["citations"] == []

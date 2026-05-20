import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.state import AnalystState, INITIAL_STATE
from src.tools import ALL_TOOLS
from src.report_generator import synthesize_report


SYSTEM_PROMPT = """You are a senior financial analyst specializing in blockchain and crypto markets.
Your job is to produce a comprehensive analyst report using the tools available to you.

Approach:
1. GATHER — use get_market_data and query_transaction_analytics to collect metrics
2. ANALYZE — use run_statistical_analysis and assess_risk_metrics for insights
3. CALCULATE — use calculator for precise financial computations
4. SYNTHESIZE — compile findings into a structured analyst report

Be specific with numbers. Flag risks prominently. Write for both technical and non-technical readers.
When you've gathered enough data (typically 3-5 tool calls), stop calling tools and write the report."""

MAX_ITERATIONS = 10


def _llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0).bind_tools(ALL_TOOLS)


def agent_node(state: AnalystState, llm=None) -> AnalystState:
    if llm is None:
        llm = _llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(msgs)
    return {**state, "messages": [response], "iteration": state["iteration"] + 1}


def tool_node_fn(state: AnalystState) -> AnalystState:
    result = ToolNode(ALL_TOOLS).invoke(state)

    ctx = dict(state["data_context"])
    citations = list(state["citations"])

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                out = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                out = msg.content
            ctx[msg.name] = out
            citations.append(f"{msg.name} (iteration {state['iteration']})")

    return {**state, **result, "data_context": ctx, "citations": citations}


def report_node(state: AnalystState) -> AnalystState:
    report = synthesize_report(
        query=state["query"],
        data_context=state["data_context"],
        messages=state["messages"],
        citations=state["citations"],
    )
    return {**state, "final_report": report}


def should_continue(state: AnalystState) -> Literal["tools", "report", "__end__"]:
    if state["iteration"] >= MAX_ITERATIONS:
        return "report"

    last = state["messages"][-1] if state["messages"] else None
    if last is None:
        return "__end__"

    if isinstance(last, AIMessage):
        return "tools" if last.tool_calls else "report"

    return "__end__"


def build_agent_graph(model: str = "gpt-4o-mini") -> StateGraph:
    llm = _llm(model)

    def _agent(state):
        return agent_node(state, llm=llm)

    g = StateGraph(AnalystState)
    g.add_node("agent", _agent)
    g.add_node("tools", tool_node_fn)
    g.add_node("report", report_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "report": "report", "__end__": END})
    g.add_edge("tools", "agent")
    g.add_edge("report", END)

    return g.compile()


def run_analyst_agent(query: str, model: str = "gpt-4o-mini", verbose: bool = True) -> dict:
    agent = build_agent_graph(model=model)
    state = {**INITIAL_STATE, "query": query, "messages": [HumanMessage(content=query)]}

    if verbose:
        print(f"\n[Agent] query: {query[:80]}...")

    final = agent.invoke(state)

    if verbose:
        print(f"\n[Agent] done in {final['iteration']} iterations")
        print(f"[Agent] tools used: {list(final['data_context'].keys())}")
        print(f"\n{'='*60}\n{final.get('final_report', 'no report generated')}")

    return {
        "query": query,
        "report": final.get("final_report"),
        "data_context": final.get("data_context", {}),
        "iterations": final.get("iteration", 0),
        "citations": final.get("citations", []),
    }

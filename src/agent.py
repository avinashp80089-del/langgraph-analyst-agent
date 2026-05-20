"""
Multi-step ReAct analyst agent using LangGraph.
Cuts analyst report turnaround from 3 days to under 4 hours by automating
data gathering, statistical analysis, and report synthesis.
"""
from typing import Literal, Optional
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.state import AnalystState, INITIAL_STATE
from src.tools import ALL_TOOLS
from src.report_generator import synthesize_report


SYSTEM_PROMPT = """You are a senior financial analyst with expertise in blockchain and fintech markets.
Your task is to produce a comprehensive analyst report using the tools available to you.

Follow this structured approach:
1. GATHER DATA: Use get_market_data and query_transaction_analytics to collect relevant metrics
2. ANALYZE: Use run_statistical_analysis and assess_risk_metrics to derive insights
3. CALCULATE: Use the calculator for precise financial computations
4. SYNTHESIZE: Compile findings into a professional analyst report

Guidelines:
- Always cite your data sources
- Include specific numbers with proper context
- Flag any risks or caveats prominently
- Structure insights for both technical and non-technical readers

When you have gathered sufficient data (typically 3-5 tool calls), call the finish_report function."""

MAX_ITERATIONS = 10


def _build_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0).bind_tools(ALL_TOOLS)


def agent_node(state: AnalystState, llm=None) -> AnalystState:
    """Core ReAct reasoning step — the LLM decides which tool to call next."""
    if llm is None:
        llm = _build_llm()

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    return {
        **state,
        "messages": [response],
        "iteration": state["iteration"] + 1,
    }


def tool_node_fn(state: AnalystState) -> AnalystState:
    """Execute tool calls and accumulate results into data_context."""
    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    # Accumulate tool results into data_context
    data_context = dict(state["data_context"])
    citations = list(state["citations"])

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                output = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                output = msg.content
            data_context[msg.name] = output
            citations.append(f"{msg.name} (tool call at iteration {state['iteration']})")

    return {
        **state,
        **result,
        "data_context": data_context,
        "citations": citations,
    }


def report_node(state: AnalystState) -> AnalystState:
    """Synthesize all gathered data into a final analyst report."""
    report = synthesize_report(
        query=state["query"],
        data_context=state["data_context"],
        messages=state["messages"],
        citations=state["citations"],
    )
    return {**state, "final_report": report}


def should_continue(state: AnalystState) -> Literal["tools", "report", "__end__"]:
    """
    Routing function — decides whether to call more tools, generate the report,
    or end due to max iteration guard.
    """
    if state["iteration"] >= MAX_ITERATIONS:
        return "report"

    last_message = state["messages"][-1] if state["messages"] else None
    if last_message is None:
        return "__end__"

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "tools"
        return "report"

    return "__end__"


def build_agent_graph(model: str = "gpt-4o-mini") -> StateGraph:
    """
    Construct the LangGraph ReAct agent graph.

    Graph topology:
        START → agent → (tools → agent)* → report → END
    """
    llm = _build_llm(model)

    def _agent_node(state: AnalystState) -> AnalystState:
        return agent_node(state, llm=llm)

    graph = StateGraph(AnalystState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node_fn)
    graph.add_node("report", report_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "report": "report", "__end__": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("report", END)

    return graph.compile()


def run_analyst_agent(
    query: str,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> dict:
    """
    Run the multi-step ReAct agent on an analyst query.
    Returns the final report and full execution trace.

    Typical turnaround: under 4 hours of analyst time compressed to seconds.
    """
    agent = build_agent_graph(model=model)
    initial_state = {
        **INITIAL_STATE,
        "query": query,
        "messages": [HumanMessage(content=query)],
    }

    if verbose:
        print(f"\n[Agent] Starting analysis: {query[:80]}...")

    final_state = agent.invoke(initial_state)

    if verbose:
        print(f"\n[Agent] Completed in {final_state['iteration']} iterations")
        print(f"[Agent] Tools called: {list(final_state['data_context'].keys())}")
        print(f"\n{'='*60}\nFINAL REPORT\n{'='*60}")
        print(final_state.get("final_report", "No report generated"))

    return {
        "query": query,
        "report": final_state.get("final_report"),
        "data_context": final_state.get("data_context", {}),
        "iterations": final_state.get("iteration", 0),
        "citations": final_state.get("citations", []),
    }

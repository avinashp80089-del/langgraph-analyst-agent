"""Agent state definition for the LangGraph ReAct analyst agent."""
from typing import Annotated, TypedDict, Optional, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage


class AnalystState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    Holds the full message history, accumulated context, and report draft.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    data_context: Dict[str, Any]        # results fetched by tool calls
    report_sections: List[Dict[str, str]]  # accumulated report sections
    citations: List[str]                # sources used
    iteration: int                      # tracks ReAct loop depth (max_iterations guard)
    final_report: Optional[str]         # set when report is complete


class ToolResult(TypedDict):
    tool_name: str
    input: Dict[str, Any]
    output: Any
    elapsed_ms: float
    success: bool
    error: Optional[str]


INITIAL_STATE: AnalystState = {
    "messages": [],
    "query": "",
    "data_context": {},
    "report_sections": [],
    "citations": [],
    "iteration": 0,
    "final_report": None,
}

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AnalystState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    data_context: Dict[str, Any]
    report_sections: List[Dict[str, str]]
    citations: List[str]
    iteration: int
    final_report: Optional[str]


INITIAL_STATE: AnalystState = {
    "messages": [],
    "query": "",
    "data_context": {},
    "report_sections": [],
    "citations": [],
    "iteration": 0,
    "final_report": None,
}

"""
Conversation memory management for multi-turn analyst sessions.
Supports short-term (in-session) and long-term (persisted) memory.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class SessionMemory:
    """In-memory store for a single analyst session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self._messages: List[BaseMessage] = []
        self._report_history: List[Dict[str, Any]] = []
        self._tool_call_log: List[Dict[str, Any]] = []

    def add_messages(self, messages: List[BaseMessage]):
        self._messages.extend(messages)

    def get_messages(self, last_n: Optional[int] = None) -> List[BaseMessage]:
        if last_n:
            return self._messages[-last_n:]
        return list(self._messages)

    def log_tool_call(self, tool_name: str, inputs: Dict, output: Any, elapsed_ms: float):
        self._tool_call_log.append({
            "tool": tool_name,
            "inputs": inputs,
            "output": output,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def save_report(self, query: str, report: str, data_context: Dict):
        self._report_history.append({
            "query": query,
            "report": report,
            "data_context": data_context,
            "generated_at": datetime.utcnow().isoformat(),
        })

    def get_report_history(self) -> List[Dict[str, Any]]:
        return list(self._report_history)

    def summarize(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self._messages),
            "reports_generated": len(self._report_history),
            "tool_calls": len(self._tool_call_log),
            "tools_used": list({e["tool"] for e in self._tool_call_log}),
        }

    def clear(self):
        self._messages.clear()
        self._tool_call_log.clear()


class PersistentMemory:
    """
    File-backed memory store — persists session summaries and report history
    across agent restarts. Analogous to LangChain's ConversationSummaryBufferMemory
    but tuned for the analyst use case.
    """

    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: SessionMemory):
        path = self.storage_dir / f"{session.session_id}.json"
        data = {
            **session.summarize(),
            "reports": session.get_report_history(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[Memory] Session {session.session_id} saved to {path}")

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self.storage_dir / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        summaries = []
        for path in sorted(self.storage_dir.glob("*.json")):
            with open(path) as f:
                summaries.append(json.load(f))
        return summaries

    def get_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve past reports most relevant to the current query.
        Simple keyword matching — swap in a vector similarity search in production.
        """
        all_sessions = self.list_sessions()
        query_tokens = set(query.lower().split())

        scored = []
        for session in all_sessions:
            for report in session.get("reports", []):
                report_tokens = set(report.get("query", "").lower().split())
                overlap = len(query_tokens & report_tokens)
                if overlap > 0:
                    scored.append((overlap, report))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

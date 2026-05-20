import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage


class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self._messages: List[BaseMessage] = []
        self._reports: List[Dict[str, Any]] = []
        self._tool_log: List[Dict[str, Any]] = []

    def add_messages(self, messages: List[BaseMessage]):
        self._messages.extend(messages)

    def get_messages(self, last_n: Optional[int] = None) -> List[BaseMessage]:
        return self._messages[-last_n:] if last_n else list(self._messages)

    def log_tool_call(self, tool_name: str, inputs: Dict, output: Any, elapsed_ms: float):
        self._tool_log.append({
            "tool": tool_name,
            "inputs": inputs,
            "output": output,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def save_report(self, query: str, report: str, data_context: Dict):
        self._reports.append({
            "query": query,
            "report": report,
            "data_context": data_context,
            "generated_at": datetime.utcnow().isoformat(),
        })

    def get_report_history(self):
        return list(self._reports)

    def summarize(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self._messages),
            "reports_generated": len(self._reports),
            "tool_calls": len(self._tool_log),
            "tools_used": list({e["tool"] for e in self._tool_log}),
        }

    def clear(self):
        self._messages.clear()
        self._tool_log.clear()


class PersistentMemory:
    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: SessionMemory):
        path = self.storage_dir / f"{session.session_id}.json"
        data = {**session.summarize(), "reports": session.get_report_history()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[Memory] saved {session.session_id} → {path}")

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self.storage_dir / f"{session_id}.json"
        return json.load(open(path)) if path.exists() else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(self.storage_dir.glob("*.json")):
            with open(p) as f:
                out.append(json.load(f))
        return out

    def get_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword overlap — swap in vector similarity in prod."""
        query_tokens = set(query.lower().split())
        scored = []
        for session in self.list_sessions():
            for report in session.get("reports", []):
                overlap = len(query_tokens & set(report.get("query", "").lower().split()))
                if overlap > 0:
                    scored.append((overlap, report))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

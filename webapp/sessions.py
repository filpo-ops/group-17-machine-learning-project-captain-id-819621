"""In-memory session store for the webapp.

Each upload (or demo load) creates a session keyed by a short opaque ID. The
session holds the user's DataFrame plus the cached final state once the pipeline
has run. Cleared when the server restarts; that's by design — no persistence,
no auth, no DB.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Session:
    """One in-flight (or completed) pipeline run."""
    id: str
    name: str
    df: pd.DataFrame
    final_state: Optional[Dict[str, Any]] = None     # populated after run
    html_report: Optional[str] = None                # populated after run
    correction_log_json: Optional[str] = None        # cached download

    def is_ready(self) -> bool:
        return self.final_state is not None and self.html_report is not None


class SessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, df: pd.DataFrame, name: str) -> Session:
        sid = secrets.token_urlsafe(8)
        sess = Session(id=sid, name=name, df=df)
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def get(self, sid: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(sid)

    def update(self, sid: str, **fields: Any) -> None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return
            for k, v in fields.items():
                setattr(sess, k, v)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


# Singleton store used by the app
store = SessionStore()

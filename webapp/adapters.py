"""Adapter: pipeline final_state (Python) → JSON shape expected by the React frontend.

The React `data.js` shape was designed for the Claude Design prototype mock data;
this module translates our actual `final_state` dict (from `run_quality_pipeline`)
into the same shape so the frontend code can stay unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

# Constant referenced by the frontend's "ISO-8000 weighted" copy in the score card.
RELIABILITY_WEIGHTS_DICT = {
    "completeness": 0.30,
    "consistency":  0.25,
    "validity":     0.20,
    "uniqueness":   0.15,
    "accuracy":     0.10,
}

# Provider name shown in the UI top bar / report.
PROVIDER_NAME = "DeepSeek-Chat (V3)"


def _df_preview(df: pd.DataFrame, n: int = 8) -> List[Dict[str, Any]]:
    """Sanitize a small DataFrame slice to JSON-safe dicts."""
    if df is None or df.empty:
        return []
    return df.head(n).fillna("").astype(str).to_dict(orient="records")


def state_to_react_payload(final_state: Dict[str, Any], html_report: str) -> Dict[str, Any]:
    """Build the JSON the React frontend consumes after a pipeline run completes.

    Mirrors the shape of `data.js → RESULTS` from the Claude Design prototype, plus
    a few extra fields used by the live timeline (provider, dataset_name).
    """
    issues = final_state.get("issues", [])
    log    = final_state.get("correction_log", [])

    # Assign 1-based ids so the React side can use them as React keys
    issues_out = [
        {
            "id":           idx + 1,
            "tool":         iss.get("tool"),
            "issue_type":   iss.get("issue_type"),
            "severity":     iss.get("severity"),
            "columns":      iss.get("columns") or [],
            "row_count":    iss.get("row_count"),
            "message":      iss.get("message", ""),
            "sample_rows":  iss.get("sample_rows", []),
        }
        for idx, iss in enumerate(issues)
    ]

    log_out = [
        {
            "id":             idx + 1,
            "agent":          entry.get("agent", "—"),
            "action":         entry.get("action"),
            "column":         entry.get("column") or (entry.get("columns")[0] if entry.get("columns") else None),
            "rows_affected":  entry.get("rows_affected", 0),
            "rationale":      entry.get("rationale") or entry.get("reason") or "",
            "applied":        bool(entry.get("applied", False)),
        }
        for idx, entry in enumerate(log)
    ]

    raw_df   = final_state.get("raw_df")
    fixed_df = final_state.get("fixed_df", raw_df)

    # Post-remediation re-audit: only present if node_re_audit ran.
    # The frontend uses these to show a before/after delta on the score card.
    post_score   = final_state.get("post_reliability_score")
    post_sub     = final_state.get("post_sub_scores") or {}
    post_sev     = final_state.get("post_severity_breakdown") or {}
    rem_score    = final_state.get("remediation_score")
    rem_weighted = final_state.get("remediation_score_weighted")

    return {
        "dataset_name":       final_state.get("dataset_name"),
        "provider":           PROVIDER_NAME,
        "reliability_score":  float(final_state.get("reliability_score", 0)),
        "sub_scores":         dict(final_state.get("sub_scores", {})),
        "weights":            RELIABILITY_WEIGHTS_DICT,
        "severity_breakdown": dict(final_state.get("severity_breakdown", {})),
        "issues":             issues_out,
        "correction_log":     log_out,
        "audit_trail":        list(final_state.get("audit_trail", [])),
        "fixed_preview":      _df_preview(fixed_df, n=8),
        "raw_shape":          list(raw_df.shape) if raw_df is not None else None,
        "fixed_shape":        list(fixed_df.shape) if fixed_df is not None else None,
        "html_report":        html_report,
        # Post-remediation reliability (None if re_audit didn't run for some reason)
        "post_reliability_score":  float(post_score) if post_score is not None else None,
        "post_sub_scores":         dict(post_sub),
        "post_severity_breakdown": dict(post_sev),
        # Resolution-rate metrics (complementary to reliability_score):
        # how much of the detected work the pipeline actually closed.
        "remediation_score":          float(rem_score) if rem_score is not None else None,
        "remediation_score_weighted": float(rem_weighted) if rem_weighted is not None else None,
    }


def state_snapshot_to_event(node_name: str, snapshot: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    """Build a per-node SSE event payload from an accumulated state snapshot.

    Used by the SSE endpoint after each `stream_quality_pipeline` yield.
    Returns a small dict (≪ full state) to keep SSE messages compact.
    """
    audit_trail = snapshot.get("audit_trail") or []
    last_msg    = audit_trail[-1] if audit_trail else ""
    return {
        "node":     node_name,
        "status":   "done",
        "elapsed":  round(elapsed, 2),
        "message":  last_msg,
    }

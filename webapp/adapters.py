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


def _values_equivalent(a: str, b: str) -> bool:
    """Compare two stringified cell values with tolerance for cosmetic dtype
    promotion. Returns True if the values are semantically equal — i.e. equal
    as strings, OR equal as floats within a small tolerance.

    This prevents int→float promotion from being flagged as a "fix" (e.g. raw
    `"72"` vs fixed `"72.0"` should NOT be marked as modified). Promotion is
    a routine side-effect of pandas inserting `NaN` into an integer column.
    """
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (ValueError, TypeError):
        return False


def _diff_preview(raw_df: pd.DataFrame, fixed_df: pd.DataFrame, n: int = 8) -> List[Dict[str, Any]]:
    """Build a JSON-safe preview of `fixed_df.head(n)` with each cell that
    differs from the corresponding cell in `raw_df.head(n)` suffixed by `*`.

    The frontend's `FixedPreview` component highlights any cell whose string
    value ends with `*` — that's how the user sees, at a glance, which values
    the pipeline modified. Without this comparison the legend "* = value
    modified by remediation" is dead text.

    Comparison is done on the STRING representation (with numeric tolerance,
    see `_values_equivalent`) because remediation often changes dtypes
    (e.g. `"29.99 EUR"` → `29.99`), and we want a "this changed" signal
    regardless of the underlying type — but NOT for purely cosmetic changes
    like int → float promotion.
    """
    if fixed_df is None or fixed_df.empty:
        return []
    f = fixed_df.head(n).fillna("").astype(str)
    if raw_df is None or raw_df.empty:
        return f.to_dict(orient="records")
    r = raw_df.head(n).fillna("").astype(str)

    # Schema may have shifted (renamed / dropped columns) so we mark only on
    # the columns present in the fixed frame; new columns are always marked,
    # missing columns are skipped.
    #
    # Heuristic against dedup-induced false positives: when remediation drops
    # a duplicate row, every subsequent fixed row is positionally compared
    # against a different raw row — so naive cell-by-cell diff would mark ALL
    # columns as modified. If more than ~70 % of a row's cells differ, we
    # assume a row reorder and emit no asterisks. Real fixes typically touch
    # 1–3 cells per row.
    REORDER_THRESHOLD = 0.7

    out: List[Dict[str, Any]] = []
    for i in range(len(f)):
        row_diffs = []
        for col in f.columns:
            f_val = f.iat[i, f.columns.get_loc(col)]
            if i < len(r) and col in r.columns:
                r_val = r.iat[i, r.columns.get_loc(col)]
                row_diffs.append((col, f_val, not _values_equivalent(f_val, r_val)))
            else:
                row_diffs.append((col, f_val, True))

        diff_ratio = sum(1 for _, _, d in row_diffs if d) / max(1, len(row_diffs))
        suppress_marks = diff_ratio > REORDER_THRESHOLD

        out.append({
            col: f"{val}*" if (changed and not suppress_marks) else val
            for col, val, changed in row_diffs
        })
    return out


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
        "fixed_preview":      _diff_preview(raw_df, fixed_df, n=8),
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

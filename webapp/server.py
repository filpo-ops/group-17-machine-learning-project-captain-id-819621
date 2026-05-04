"""FastAPI backend for the multi-agent data quality demo.

Endpoints:
    GET  /                          → serve static/index.html
    POST /upload                    → accept CSV, create session, return preview
    POST /demo                      → load synthetic orders.csv as demo, return preview
    GET  /run/{session_id}          → SSE stream of pipeline events (Phase 3)
    GET  /report/{session_id}       → serve final HTML report (Phase 6)
    GET  /download/fixed/{sid}      → fixed_<name>.csv (Phase 6)
    GET  /download/log/{sid}        → correction_log JSON (Phase 6)
    GET  /benchmark                 → return evaluation_results_v2.json (Phase 6)

Run with:
    # Dev (auto-reload, watches only source — NOT .venv, which would loop forever):
    uvicorn webapp.server:app --reload --reload-dir webapp --reload-dir agents --port 8000

    # Demo / no-reload:
    uvicorn webapp.server:app --port 8000
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict
from collections import OrderedDict

# Make `agents.pipeline` importable when the webapp is launched from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool
from webapp.sessions import store as session_store
from webapp.sessions import Session
from webapp.adapters import state_to_react_payload, state_snapshot_to_event

# ─── Application ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agents for Data Quality",
    description="Multi-agent LLM pipeline (LUISS ML 2025/26 — Reply Group 17)",
    version="1.0.0",
)

# Permissive CORS — same-origin in production, but useful during dev when serving the
# frontend separately (e.g. live-reload tools).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "webapp" / "static"
DEMO_CSV   = PROJECT_ROOT / "agents" / "data" / "synthetic" / "orders.csv"
BENCHMARK_JSON_V2  = PROJECT_ROOT / "agents" / "data" / "benchmark" / "evaluation_results_v2.json"
BENCHMARK_JSON_V1  = PROJECT_ROOT / "agents" / "data" / "benchmark" / "evaluation_results.json"

# ─── Replay cache ─────────────────────────────────────────────────────────────
# LRU keyed by SHA-1 hash of the dataframe content. When the user clicks "Run"
# on a CSV the system has seen before (typical demo scenario: hit "Try with the
# synthetic orders sample" twice), we serve the cached payload instead of
# burning LLM tokens.
_REPLAY_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_REPLAY_CACHE_MAX = 8


def _replay_key(df) -> str:
    """Hash the df for stable cache lookup; ~10× faster than to_csv+sha1 on 5k rows."""
    return hashlib.sha1(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def _replay_get(key: str) -> Dict[str, Any] | None:
    if key in _REPLAY_CACHE:
        _REPLAY_CACHE.move_to_end(key)
        return _REPLAY_CACHE[key]
    return None


def _replay_put(key: str, payload: Dict[str, Any]) -> None:
    _REPLAY_CACHE[key] = payload
    _REPLAY_CACHE.move_to_end(key)
    while len(_REPLAY_CACHE) > _REPLAY_CACHE_MAX:
        _REPLAY_CACHE.popitem(last=False)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _classify_dtype(s: pd.Series) -> str:
    """Classify a column into exactly one of: numeric / date / string.

    Every column lands in exactly one bucket. `is_object_dtype` alone misses
    modern pandas extension dtypes (`string`, `category`, `bool`), so we use a
    positive-test cascade: dates first, then strict numeric (excluding bool),
    then everything else falls into 'string'.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numeric"
    return "string"  # object, string, category, bool, etc.


def _df_preview_payload(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Build the JSON payload the frontend expects for the dataset-preview screen."""
    n_rows, n_cols = df.shape
    type_counts = {"numeric": 0, "string": 0, "date": 0}
    for col in df.columns:
        type_counts[_classify_dtype(df[col])] += 1
    # Coerce preview rows to JSON-safe primitives
    preview = df.head(8).fillna("").astype(str).to_dict(orient="records")
    return {
        "filename": f"{name}.csv",
        "size":     f"{int(df.memory_usage(deep=True).sum() / 1024)} KB",
        "rows":     int(n_rows),
        "cols":     int(n_cols),
        "types":    type_counts,
        "columns":  list(df.columns),
        "preview":  preview,
    }


def _read_csv_safely(buf: bytes, name: str) -> pd.DataFrame:
    """Parse a CSV buffer into a DataFrame; raises HTTP 400 on parse errors or empty file.

    No row cap: large CSVs are allowed at the cost of slower pipeline runs.
    """
    try:
        df = pd.read_csv(io.BytesIO(buf))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV `{name}`: {e}") from e
    if df.empty:
        raise HTTPException(status_code=400, detail=f"CSV `{name}` is empty.")
    return df


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the React frontend (single static HTML page)."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>webapp/static/index.html missing</h1>"
            "<p>Run Phase 4 of the implementation plan to materialize the frontend.</p>",
            status_code=503,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept a CSV upload, create a session, return the preview payload."""
    contents = await file.read()
    name = Path(file.filename or "upload").stem
    df = _read_csv_safely(contents, name=name)
    sess = session_store.create(df=df, name=name)
    return {"session_id": sess.id, **_df_preview_payload(df, name)}


@app.post("/demo")
async def load_demo() -> Dict[str, Any]:
    """Load the synthetic `orders.csv` demo dataset and return its preview payload.

    The synthetic orders dataset is intentionally neutral with respect to the
    NoiPA fixtures the pipeline was tuned on: 500 rows × 13 columns of
    e-commerce data with anomalies seeded in the first 8 rows for showcase
    visibility plus moderate scattering across the rest (~13 % rows touched).
    """
    if not DEMO_CSV.exists():
        raise HTTPException(status_code=500, detail=f"Demo dataset missing: {DEMO_CSV}")
    df = pd.read_csv(DEMO_CSV)
    sess = session_store.create(df=df, name="orders")
    return {"session_id": sess.id, **_df_preview_payload(df, "orders")}


@app.get("/cache/{session_id}")
async def replay_cache_lookup(session_id: str) -> Dict[str, Any]:
    """If the dataset hash is already in the replay cache, return the payload directly.

    Used by the frontend to offer a "Replay last run (instant)" button when the
    user runs the same demo dataset twice — no LLM calls, no waiting.
    """
    sess = session_store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id `{session_id}`.")
    payload = _replay_get(_replay_key(sess.df))
    if payload is None:
        raise HTTPException(status_code=404, detail="No cached run for this dataset.")
    # Re-attach session_id so frontend can use it for downloads
    return {**payload, "session_id": session_id, "cached": True}


@app.get("/run/{session_id}")
async def run_pipeline_stream(session_id: str) -> EventSourceResponse:
    """SSE stream of pipeline events for the live timeline.

    Emits one `node_done` event per pipeline node + a final `complete` event
    carrying the full React payload. The frontend uses an `EventSource`
    listener to drive the per-stage timeline UI.
    """
    sess = session_store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id `{session_id}`.")

    async def event_generator() -> AsyncIterator[Dict[str, Any]]:
        # Lazy import keeps cold-start fast
        from agents.pipeline import (
            stream_quality_pipeline,
            render_quality_report,
        )

        t0 = time.time()
        last_state: Dict[str, Any] | None = None

        # The pipeline is sync (LLM calls block on network) — drive it from a
        # threadpool so each `next(gen)` doesn't freeze the event loop. Without
        # this, a single in-flight pipeline blocks all other HTTP/SSE traffic.
        _SENTINEL = object()
        gen = stream_quality_pipeline(sess.df, name=sess.name)

        def _next_or_sentinel():
            try:
                return next(gen)
            except StopIteration:
                return _SENTINEL

        try:
            while True:
                item = await run_in_threadpool(_next_or_sentinel)
                if item is _SENTINEL:
                    break
                node_name, snapshot = item
                last_state = snapshot
                event_payload = state_snapshot_to_event(node_name, snapshot, time.time() - t0)
                yield {"event": "node_done", "data": json.dumps(event_payload)}

            # All nodes done — render the report and emit the complete payload
            if last_state is None:
                yield {"event": "error", "data": json.dumps({"message": "Pipeline produced no events."})}
                return

            html = await run_in_threadpool(render_quality_report, last_state)
            session_store.update(session_id, final_state=last_state, html_report=html)

            payload = state_to_react_payload(last_state, html)
            # Store in replay cache so future runs of the same dataset are instant
            _replay_put(_replay_key(sess.df), payload)
            yield {"event": "complete", "data": json.dumps(payload, default=str)}

        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": f"{type(exc).__name__}: {exc}"})}

    return EventSourceResponse(
        event_generator(),
        ping=15,                                            # heartbeat to keep Safari/proxies happy
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/results/{session_id}")
async def run_pipeline_sync(session_id: str) -> Dict[str, Any]:
    """Run the full pipeline synchronously and return the React payload.

    Used for testing before SSE streaming is wired up. Live demos should use
    `GET /run/{session_id}` (Phase 3) for the per-node timeline experience.
    """
    sess = session_store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id `{session_id}`.")
    # Lazy import to keep module-load time fast
    from agents.pipeline import run_quality_pipeline
    # Threadpool: pipeline is sync + network-blocking, mustn't freeze event loop
    final_state, html = await run_in_threadpool(
        run_quality_pipeline, sess.df, sess.name, False
    )
    session_store.update(session_id, final_state=final_state, html_report=html)
    return state_to_react_payload(final_state, html)


@app.get("/download/fixed/{session_id}")
async def download_fixed_csv(session_id: str) -> Response:
    """Return the corrected DataFrame as a CSV download."""
    sess = session_store.get(session_id)
    if sess is None or sess.final_state is None or sess.final_state.get("fixed_df") is None:
        raise HTTPException(status_code=404, detail="No fixed dataset for this session (run pipeline first).")
    fixed_df = sess.final_state["fixed_df"]
    csv_bytes = fixed_df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="fixed_{sess.name}.csv"'},
    )


@app.get("/download/log/{session_id}")
async def download_correction_log(session_id: str) -> Response:
    """Return the correction_log as a JSON download."""
    sess = session_store.get(session_id)
    if sess is None or sess.final_state is None:
        raise HTTPException(status_code=404, detail="No correction log for this session (run pipeline first).")
    log = sess.final_state.get("correction_log", [])
    return Response(
        content=json.dumps(log, indent=2, default=str).encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="correction_log_{sess.name}.json"'},
    )


@app.get("/download/report/{session_id}")
async def download_html_report(session_id: str) -> Response:
    """Return the rendered HTML report as a downloadable file."""
    sess = session_store.get(session_id)
    if sess is None or sess.html_report is None:
        raise HTTPException(status_code=404, detail="No HTML report for this session (run pipeline first).")
    return Response(
        content=sess.html_report.encode("utf-8"),
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="report_{sess.name}.html"'},
    )


@app.get("/download/bundle/{session_id}")
async def download_bundle(session_id: str) -> Response:
    """Return all artifacts (fixed CSV + correction log + HTML report) as a single ZIP."""
    import zipfile
    sess = session_store.get(session_id)
    if sess is None or sess.final_state is None or sess.html_report is None:
        raise HTTPException(status_code=404, detail="No artifacts to bundle (run pipeline first).")
    fixed_df = sess.final_state.get("fixed_df")
    if fixed_df is None:
        raise HTTPException(status_code=500, detail="Internal: fixed_df missing from final_state.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"fixed_{sess.name}.csv", fixed_df.to_csv(index=False))
        z.writestr(
            f"correction_log_{sess.name}.json",
            json.dumps(sess.final_state.get("correction_log", []), indent=2, default=str),
        )
        z.writestr(f"report_{sess.name}.html", sess.html_report)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="bundle_{sess.name}.zip"'},
    )


@app.get("/benchmark")
async def benchmark_results() -> Dict[str, Any]:
    """Return the v2 benchmark JSON (preferred) or fall back to v1."""
    import json
    if BENCHMARK_JSON_V2.exists():
        return {"version": "v2", **json.loads(BENCHMARK_JSON_V2.read_text())}
    if BENCHMARK_JSON_V1.exists():
        return {"version": "v1", **json.loads(BENCHMARK_JSON_V1.read_text())}
    raise HTTPException(status_code=404, detail="No benchmark results found.")


# ─── Static file mount (must come AFTER explicit routes) ──────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

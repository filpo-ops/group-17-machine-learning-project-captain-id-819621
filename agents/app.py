"""Agents for Data Quality — Streamlit demo (LUISS ML 2025/26 · Reply Group 17).

Two modes via the sidebar:
  1. Run pipeline on a CSV — live multi-agent execution with per-node timeline
  2. Benchmark — one-glance view of the deterministic-layer validation
"""
from pathlib import Path
import json, os, sys, time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent
BENCHMARK_DIR = APP_DIR / "data" / "benchmark"
IMAGES_DIR = APP_DIR / "images"
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Agents for Data Quality", page_icon="🛠️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Lightweight CSS polish ────────────────────────────────────────────────
st.markdown("""
<style>
.hero{padding:28px 32px;background:linear-gradient(135deg,#0a6e2c,#3aa860);
      color:#fff;border-radius:14px;margin-bottom:20px;
      box-shadow:0 4px 16px rgba(10,110,44,0.18)}
.hero h1{margin:0;font-size:34px;font-weight:700;letter-spacing:-0.5px}
.hero .tagline{font-size:17px;margin-top:8px;color:rgba(255,255,255,0.95)}
.hero .meta{font-size:13px;margin-top:10px;color:rgba(255,255,255,0.75)}

.stage-card{padding:12px 16px;border-radius:8px;margin:6px 0;
            border-left:4px solid #ddd;background:#fafafa;font-family:-apple-system,sans-serif}
.stage-done{border-left-color:#0a6e2c;background:#f4faf6}
.stage-running{border-left-color:#d4ad28;background:#fffdf3}
.stage-pending{border-left-color:#ddd;background:#fafafa;color:#aaa}

.kpi{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px;text-align:center}
.kpi .l{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px}
.kpi .v{font-size:28px;font-weight:700;color:#0a6e2c;margin:6px 0}
</style>
""", unsafe_allow_html=True)


# ── Pipeline boot (cached across the session) ─────────────────────────────
@st.cache_resource(show_spinner="Booting pipeline (one-time)...")
def boot_pipeline():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None, "DEEPSEEK_API_KEY non trovata in .env"
    nb_data = json.loads((APP_DIR / "Main.ipynb").read_text(encoding="utf-8"))
    glb = {"__builtins__": __builtins__, "display": lambda x: None,
           "__file__": str(APP_DIR / "Main.ipynb")}
    os.chdir(APP_DIR)
    for i, c in enumerate(nb_data["cells"]):
        if c["cell_type"] != "code": continue
        raw = "".join(c["source"])
        first_line = raw.split("\n", 1)[0]
        if raw.startswith("%%writefile") or first_line.startswith("# Phase 8"):
            continue
        src = "\n".join(l for l in raw.split("\n") if not l.strip().startswith(("!","%")))
        try: exec(src, glb, glb)
        except Exception as e:
            return None, f"boot error in cell {i}: {type(e).__name__}: {e}"
    return glb, None


def gauge(value, label="Reliability", height=260):
    color = "#0a6e2c" if value >= 70 else ("#d4ad28" if value >= 40 else "#c0392b")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":"/100", "font":{"size":40}},
        gauge={"axis":{"range":[0,100],"tickwidth":1},
               "bar":{"color":color,"thickness":0.75},
               "steps":[{"range":[0,40],"color":"#fff0ee"},
                        {"range":[40,70],"color":"#fff7d6"},
                        {"range":[70,100],"color":"#e8f5ed"}]},
        title={"text":label, "font":{"size":14, "color":"#666"}}))
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def radar(sub_scores, height=320):
    labels = list(sub_scores.keys())
    values = [sub_scores[k] for k in labels]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself",
                                     marker=dict(color="#0a6e2c"),
                                     line=dict(color="#0a6e2c")))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                      showlegend=False, height=height,
                      margin=dict(l=40,r=40,t=20,b=20))
    return fig


def severity_bar(severity_breakdown, height=200):
    labels = ["critical","high","medium","low"]
    colors = ["#c0392b","#e67e22","#d4ad28","#95a5a6"]
    values = [severity_breakdown.get(l, 0) for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=20,b=20),
                      yaxis=dict(showgrid=False), showlegend=False)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📥 Input")
    use_demo = st.toggle("Try with NoiPA `spesa` (demo)", value=False)
    uploaded = None if use_demo else st.file_uploader("Upload a CSV", type=["csv"])
    with st.expander("⚙️ Settings"):
        max_rows = st.number_input("Max rows", 200, 50000, 5000, step=500)
        show_narrative = st.checkbox("LLM-generated executive summary", value=False,
                                      help="Adds 1 short LLM call (~150 tokens) to write the report's summary in Italian.")
    st.markdown("---")
    st.markdown("**Group 17** · LUISS ML 2025/26")
    st.markdown("Captain ID `819621` · Reply *Agents for Data Quality*")
    mode = st.radio("Mode", ["🚀 Run pipeline", "🧪 Benchmark"], index=0)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛠️ Agents for Data Quality</h1>
  <div class="tagline">A multi-agent LLM pipeline that validates, scores, and remediates any CSV — in seconds.</div>
  <div class="meta">LangGraph · DEEPSEEK v4 flash · 9 nodes (4 LLM agents + supervisor) · ~3-5k tokens / dataset</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────── Mode 1: Run pipeline ─────────────────────────
if mode == "🚀 Run pipeline":
    if uploaded is None and not use_demo:
        # Welcome / explainer
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("### How it works")
            st.markdown("""
**6 specialized agents collaborate to assess your data:**

| Stage | Type | What it does |
|---|---|---|
| 🔌 Ingest + Discover | deterministic | Loads the CSV, infers schema rules from a sample |
| 🔍 Audit | deterministic | Runs 9 quality tools, accumulates issues |
| 🔧 Schema · Completeness · Consistency · Anomaly | **4 LLM agents** | Each plans fixes for its dimension, with column context |
| ⚡ Remediation | deterministic | Applies the merged plan with atomic tools |
| 🎯 Supervisor | deterministic | Aggregates 5 sub-scores → reliability 0-100 (ISO-8000) |
            """)
            if (IMAGES_DIR / "architecture_flowchart.png").exists():
                st.image(str(IMAGES_DIR / "architecture_flowchart.png"))
        with c2:
            st.markdown("### Quick start")
            st.markdown("👈 Upload a CSV in the sidebar, **or** flip the toggle to try with a demo dataset.")
            st.info("Pre-validated on 4 NoiPA datasets (Reply test fixtures). The pipeline is **schema-agnostic** — works on any tabular CSV.")
            st.markdown("### Benchmark — deterministic layer")
            ev = BENCHMARK_DIR / "evaluation_results.json"
            if ev.exists():
                r = json.loads(ev.read_text())
                k1, k2, k3 = st.columns(3)
                k1.metric("Precision", f"{r['Precision']:.2f}")
                k2.metric("Recall", f"{r['Recall']:.2f}")
                k3.metric("F1", f"{r['F1']:.2f}")
                st.caption(f"3 error types × 4 datasets, n_each=3")
        st.stop()

    # ── Got a CSV: load + preview
    if uploaded is not None:
        df_user = pd.read_csv(uploaded).head(max_rows)
        ds_name = Path(uploaded.name).stem
    else:
        df_user = pd.read_csv(DATA_DIR / "project_data_quality" / "spesa.csv").head(max_rows)
        ds_name = "spesa"

    st.subheader(f"📊 `{ds_name}`")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{df_user.shape[0]:,}")
    cols[1].metric("Columns", df_user.shape[1])
    cols[2].metric("Numeric cols", df_user.select_dtypes(include="number").shape[1])
    cols[3].metric("String cols", df_user.select_dtypes(include="object").shape[1])
    with st.expander("Preview (first 10 rows)", expanded=False):
        st.dataframe(df_user.head(10), use_container_width=True, hide_index=True)

    if not st.button("🚀 Run multi-agent pipeline", type="primary", use_container_width=True):
        st.stop()

    glb, err = boot_pipeline()
    if err: st.error(err); st.stop()
    quality_graph = glb["quality_graph"]
    glb["datasets"][ds_name] = df_user

    # ── Live timeline of the 9 nodes via graph.stream() ───────────────────
    PIPELINE_STAGES = ["ingest", "discover", "audit", "schema",
                        "completeness", "consistency", "anomaly",
                        "remediation", "supervisor"]
    STAGE_LABELS = {
        "ingest":       "🔌 Ingest — loading dataframe",
        "discover":     "🔬 Discover — inferring schema rules",
        "audit":        "🔍 Audit — 9 deterministic tools running",
        "schema":       "🤖 Schema agent (LLM) — validity dimension",
        "completeness": "🤖 Completeness agent (LLM) — completeness dimension",
        "consistency":  "🤖 Consistency agent (LLM) — consistency + uniqueness",
        "anomaly":      "🤖 Anomaly agent (LLM) — accuracy dimension",
        "remediation":  "⚡ Remediation — applying fixes deterministically",
        "supervisor":   "🎯 Supervisor — aggregating reliability score",
    }

    timeline_box = st.container()
    placeholders = {s: timeline_box.empty() for s in PIPELINE_STAGES}
    for s in PIPELINE_STAGES:
        placeholders[s].markdown(f'<div class="stage-card stage-pending">⏸️ {STAGE_LABELS[s]}</div>', unsafe_allow_html=True)

    # Stream the graph and update the UI on every node completion.
    # graph.stream() emits only node-update events, NOT the initial input — so we
    # seed `state` with dataset_name and raw_df ourselves.
    state = {"dataset_name": ds_name, "raw_df": df_user.copy()}
    t0 = time.time()
    try:
        for event in quality_graph.stream({"dataset_name": ds_name, "raw_df": df_user.copy()}):
            for node_name, update in event.items():
                # Merge update into accumulated state (mirrors LangGraph reducers)
                for k, v in update.items():
                    if isinstance(state.get(k), list) and isinstance(v, list):
                        state[k] = state[k] + v
                    elif isinstance(state.get(k), dict) and isinstance(v, dict):
                        state[k] = {**state[k], **v}
                    else:
                        state[k] = v
                # Render done state for this node
                tag = state.get("audit_trail", [""])[-1] if state.get("audit_trail") else ""
                elapsed = time.time() - t0
                if node_name in placeholders:
                    placeholders[node_name].markdown(
                        f'<div class="stage-card stage-done">✅ <strong>{STAGE_LABELS[node_name]}</strong> · {elapsed:.1f}s<br>'
                        f'<span style="color:#666;font-size:13px">{tag}</span></div>',
                        unsafe_allow_html=True)
    except Exception as e:
        st.exception(e); st.stop()

    if show_narrative:
        with st.spinner("Generating narrative summary..."):
            state = glb["add_llm_narrative"](state)

    # Ensure raw_df present for the renderer
    if "raw_df" not in state: state["raw_df"] = df_user.copy()
    html = glb["render_quality_report"](state)

    st.success(f"Pipeline complete in {time.time()-t0:.1f}s · {len(state.get('issues',[]))} issues · {sum(1 for e in state.get('correction_log',[]) if e.get('applied'))} corrections applied.")

    # ── Results dashboard ────────────────────────────────────────────────
    score = state["reliability_score"]
    sub = state["sub_scores"]
    n_issues = len(state.get("issues", []))
    n_applied = sum(1 for e in state.get("correction_log", []) if e.get("applied"))

    g1, g2 = st.columns([1, 1])
    g1.plotly_chart(gauge(score, "Reliability score"), use_container_width=True)
    g2.plotly_chart(radar(sub, height=260), use_container_width=True)

    k = st.columns(5)
    for i, (lbl, val) in enumerate(sub.items()):
        k[i].markdown(f'<div class="kpi"><div class="l">{lbl}</div><div class="v">{val:.0f}</div></div>',
                      unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Report", "🔍 Issues", "🔧 Corrections", "📁 Fixed CSV", "🛤️ Audit trail"])

    with tab1:
        st.components.v1.html(html, height=900, scrolling=True)
        st.download_button("⬇️ Download HTML report", html.encode(),
                           file_name=f"report_{ds_name}.html", mime="text/html")

    with tab2:
        sev = state.get("severity_breakdown", {})
        c1, c2 = st.columns([1, 2])
        c1.plotly_chart(severity_bar(sev), use_container_width=True)
        if n_issues > 0:
            issues_df = pd.DataFrame([{
                "tool": i["tool"], "type": i["issue_type"], "severity": i["severity"],
                "columns": ", ".join(i.get("columns", []) or []),
                "rows": i.get("row_count"), "message": (i.get("message") or "")[:140]
            } for i in state["issues"]])
            c2.dataframe(issues_df, use_container_width=True, hide_index=True, height=400)
        else:
            c2.success("No issues — clean dataset.")

    with tab3:
        log = state.get("correction_log", [])
        if log:
            log_df = pd.DataFrame(log)
            st.dataframe(log_df, use_container_width=True, hide_index=True, height=420)
        else:
            st.info("Nothing to correct.")

    with tab4:
        fixed_df = state.get("fixed_df", df_user)
        st.dataframe(fixed_df.head(50), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download fixed CSV",
                           fixed_df.to_csv(index=False).encode(),
                           file_name=f"fixed_{ds_name}.csv", mime="text/csv")

    with tab5:
        st.code("\n".join(state.get("audit_trail", [])), language=None)
        if state.get("errors"):
            with st.expander("Warnings", expanded=False):
                st.warning("\n".join(state["errors"]))


# ─────────────────────────── Mode 2: Benchmark ────────────────────────────
elif mode == "🧪 Benchmark":
    st.subheader("Phase 4 — Synthetic Benchmark")
    st.caption("Validation of the deterministic Phase 3 layer on 3 representative error types × 4 NoiPA datasets, n_each=3.")
    ev = BENCHMARK_DIR / "evaluation_results.json"
    if not ev.exists():
        st.error(f"Missing: {ev}. Run the notebook (Phase 4) to generate it."); st.stop()
    r = json.loads(ev.read_text())
    cols = st.columns(3)
    cols[0].plotly_chart(gauge(r["Precision"]*100, "Precision"), use_container_width=True)
    cols[1].plotly_chart(gauge(r["Recall"]*100, "Recall"), use_container_width=True)
    cols[2].plotly_chart(gauge(r["F1"]*100, "F1"), use_container_width=True)
    st.caption(f"TP={r['TP']} · FP={r['FP']} · FN={r['FN']}")

    st.markdown("---")
    if (IMAGES_DIR / "architecture_flowchart.png").exists():
        st.image(str(IMAGES_DIR / "architecture_flowchart.png"), caption="Pipeline architecture")
    if (IMAGES_DIR / "detection_heatmap.png").exists():
        st.image(str(IMAGES_DIR / "detection_heatmap.png"), caption="Benchmark metrics")

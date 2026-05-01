# Agents for Data Quality

**LUISS — Machine Learning A.A. 2025/26 · Reply Whitehall**
Group 17 — Ludovica De Biase, Giuseppe Catrambone, Filippo Lombardo (captain ID 819621)

![Architecture](agents/images/architecture_flowchart.png)

## [Section 1] Introduction

The system takes a raw CSV dataset as input (with anomalies typical of NoiPA public data: disguised nulls, currency symbols, heterogeneous date formats, out-of-range values, duplicate rows, cross-column logic violations) and produces two outputs:

1. a **corrected CSV** in which the anomalies have been resolved automatically through deterministic tools;
2. a **Quality Report (HTML)** with a 0–100 reliability score, per-category breakdown, list of detected issues and log of the actions applied.

The system is designed around the principle of **"determinism-first with surgical LLM"**: the deterministic layer captures every anomaly that can be expressed as a rule; four specialized LLM agents (one per reliability dimension — Schema, Completeness, Consistency, Anomaly) intervene only where the decision requires contextual reasoning, picking the fix action from a closed enum of atomic tools for their own category. This choice flips the classic "LLM-first" approach in which the model is the main engine: the rationale is twofold — **efficiency** (~3–4k tokens per dataset, against tens of thousands for an agent-everywhere approach) and **reliability** (the deterministic layer is validated by F1 on a synthetic benchmark, the LLM is verifiable through a JSON plan with a closed enum of actions and a rule-based fallback if the model fails).

## [Section 2] Methods

### Architecture
The pipeline is a LangGraph `StateGraph` with **10 nodes (4 LLM + 6 deterministic)**, single-iteration with a deterministic post-fix re-audit:

```
ingest → discover → audit → schema(LLM) → completeness(LLM) → consistency(LLM) → anomaly(LLM) → remediation → re_audit → supervisor
```

- **ingest** loads the DataFrame into the shared state.
- **discover** inspects a sample of the df and dynamically populates the validation rules (`EXPECTED_SCHEMAS`, `MANDATORY_COLUMNS`, `FORMAT_RULES`, `NUMERIC_RULES`). **Nothing is hardcoded against specific datasets** — the pipeline works on any CSV.
- **audit** runs 9 deterministic tools (Schema, Completeness, Sparse, Format, Categorical Variants, Numeric Validity, IQR Outliers, Duplicates, Cross-Column) and accumulates issues in a standardized JSON format.
- **4 LLM analysis agents** (Schema / Completeness / Consistency / Anomaly): each receives its own slice of issues (filtered by `issue_type`), makes **a single LLM call** with a closed enum of allowed actions for its category, and returns: (a) a JSON plan, (b) a 0–1 sub-score for its reliability dimension. Token budget per agent: 500–1000. Total per dataset: ~3–4k tokens. Rule-based deterministic fallback if the LLM fails or returns invalid JSON.
- **remediation** applies the consolidated plan with atomic tools (`impute_median`, `impute_mode`, `clip_iqr`, `drop_duplicates`, `normalize_dates`, `strip_currency`, `cast_numeric`, `drop_unexpected_columns`, `normalize_categorical`, `ignore`). A pre-flight guard against `col=None` (no more silent `KeyError`s) is in place. Each application produces a log entry with `agent`, `action`, `rationale` — and on failure, an explicit `reason` (column missing, etc.).
- **re_audit** (deterministic, zero LLM): re-runs the same 9 tools on `fixed_df` and recomputes sub-scores and severity post-remediation. Without this node, the reliability score reflects only the pre-fix state; with it, the UI shows a true before/after.
- **supervisor** is **deterministic** (zero LLM calls): aggregates the 5 sub-scores using the standard ISO-8000 weights (`completeness 30%, consistency 25%, validity 20%, uniqueness 15%, accuracy 10%`) and produces **two** 0–100 reliability scores — pre-fix (from the LLM sub-scores on the initial audit) and post-fix (from the deterministic sub-scores on `fixed_df`). The delta is the visible value of the pipeline.

### Sparsity-aware scoring
Columns with >95% missing values (e.g. `note_operatore`, `flag_rischio` in ALLARMI) are treated as *structurally dead*: imputing them would introduce noise, so the LLM agents correctly choose `ignore` and these issues do not penalise the completeness sub-score. Threshold controlled by `_DEAD_COLUMN_THRESHOLD` in `agents/pipeline.py`.

### Technology stack
| Component | Choice |
|---|---|
| Agent orchestration | LangGraph |
| LLM backbone | DeepSeek-Chat (V3) via `langchain-openai` (OpenAI-compatible API) |
| Webapp | FastAPI + Server-Sent Events + React 18 (Babel-standalone CDN, no build step) |
| Deterministic layer | pandas + numpy + scipy |
| Reporting | Jinja2 → self-contained HTML (PDF via browser print) |
| Language | Python 3.10+ |

### Design exploration: paths we tried before settling

The choices above did not emerge on the first attempt. We report the three experimental branches we walked (and the traces are still visible in the `feature/agents-data-quality`, `ollamacolab`, `deepseek` branches of this repo) because they illustrate the concrete trade-offs behind the final system.

**LLM provider — three attempts, one final choice.**

| Approach | Branch | Pros | Cons | Outcome |
|---|---|---|---|---|
| **Groq** (`llama-3.3-70b-versatile` via `langchain-groq`) | `feature/agents-data-quality`, early Main commits | very low latency (token streaming), suitably-sized model | aggressive rate-limits on the free tier, recurring blocks during multi-agent runs (4 close-spaced calls), API keys suspended without notice in shared dev environments | abandoned after the mid-check |
| **Ollama + Qwen on Google Colab** | `ollamacolab` | local LLM, zero cost, zero rate-limits, independence from external providers | Colab disconnects erratically (sessions interrupted mid-run), complex tunneling to expose Ollama from Colab to the local notebook, Qwen at the size we can keep in RAM shows lower quality than llama-3.3-70b on the structured-output task | abandoned (infrastructural fragility, sub-optimal quality) |
| **DeepSeek-Chat (V3)** via `langchain-openai` | `deepseek` → `Main` | OpenAI-compatible API (drop-in via `ChatOpenAI`), reasoning quality comparable on short-form structured-output tasks, negligible price, no rate-limit issues for our 4 calls/run | dependency on an external provider (mitigated by the rule-based deterministic fallback on every agent) | final choice |

**Take-away.** The pipeline's added value does not lie in any single provider — every agent has a rule-based fallback that works even with the LLM disabled (tested: +24.6 reliability points post-fix on `ALLARMI.csv` even with the pure fallback path). The provider is swappable in ~5 lines of code in `agents/pipeline.py`.

**Demo UI — three iterations.**

| Iteration | Pros | Cons | Outcome |
|---|---|---|---|
| **Streamlit** (original Phase 7, today in `legacy/streamlit/app.py`) | quick to write, materialised from the notebook with `%%writefile` | limited layout for multi-step pipeline visualization, no live node-by-node streaming, prototype look-and-feel | archived as fallback |
| **React + FastAPI v1** (commit `33d0142 demo html implementation`) | live 9-node timeline via SSE, single score, downloads of the 4 artefacts (CSV, log, report, bundle) | pre-fix score only (frustrating: the user would see 54/100 even after applying fixes that reduced issues by 70%), `FixedPreview` with hardcoded NoiPA mock columns | replaced |
| **React + FastAPI v2** (current, after Claude Design v2) | 10-node timeline grouped into 3 phases (det → llm → det), **before/after side-by-side** score with reveal animation, sub-scores with per-dimension delta, dynamic-column `FixedPreview`, 5 selectable palettes in dev mode | none significant identified | final choice |

The key architectural decision in the **v1 → v2 transition** was the introduction of the deterministic `re_audit` node: without it, the reliability score could not show an honest delta because it remained anchored to the pre-remediation audit.

### Reproducing the environment
```bash
python -m venv .venv
source .venv/bin/activate                   # macOS/Linux
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=sk-..." > .env       # provider key required for the LLM agents
jupyter lab agents/main.ipynb
```
The notebook is fully self-contained for the scientific pipeline: all the code — data loading, deterministic tools, v2 benchmark, LangGraph graph definition, execution, report generation — lives in `agents/main.ipynb`. Code cells are separated by text cells that explain *what* and *why*.

For the **webapp demo** (FastAPI + React, interactive frontend derived from Claude Design):
```bash
uvicorn webapp.server:app --port 8000
# then open: http://localhost:8000
```
The webapp runs the pipeline live on the uploaded CSV (or on the NoiPA `spesa` demo dataset), shows a 10-node timeline with SSE streaming, a score card with **before/after reliability** (e.g. `48.0 → 73.0 (+25.0)`), severity breakdown with per-level deltas (e.g. `high: 15 → 7 (-8)`), a detailed correction log and a download of the corrected CSV.

> ⚠️ Do not use uvicorn's `--reload` during a demo: reload destroys in-memory sessions and the user gets `404 Unknown session_id` between `/upload` and `/run/{sid}`.

The `agents/pipeline.py` module (extracted from `main.ipynb`) exposes the runtime API used by the webapp: `run_quality_pipeline()`, `stream_quality_pipeline()`, `render_quality_report()`, `quality_graph`, `RELIABILITY_WEIGHTS`. CLI smoke test: `python -m agents.pipeline`.

> *The earlier Streamlit demo is archived in `legacy/streamlit/app.py` as a fallback; the webapp replaces it in every flow.*

## [Section 3] Experimental Design

**Purpose.** Validate the deterministic layer (Phase 3) with a synthetic benchmark, before building the multi-agent pipeline on top of it. The logic is simple: if the functions producing the facts the LLM agents reason on are not reliable, neither is the whole pipeline.

**Baseline.** *No-op detector* (detects 0 anomalies → Precision undefined, Recall=0). A working system must clearly outperform this reference.

**Evaluation Metrics.** Precision, Recall, F1 computed at the `(dataset, error_type)` pair level, comparing injected pairs (deterministic ground truth) against detected ones. Three error types — one categorical (`disguised_null`), one numerical (`iqr_outlier`), one structural (`exact_duplicate`) — are enough to cover the typical anomaly classes of a public CSV.

## [Section 4] Results

### Deterministic layer — synthetic benchmark (Phase 4)

Run with `random.seed(42)`, `n_each=3` injections per error_type, on a 500-row sample per dataset. **3 representative error types** (one categorical, one numerical, one structural):

![Benchmark metrics](agents/images/detection_heatmap.png)

| Metric | Value |
|---|---|
| **Global F1** | 1.00 |
| Global Precision | 1.00 |
| Global Recall | 1.00 |
| TP / FP / FN | 12 / 0 / 0 |

| error_type | detected? | issue_types that capture it |
|---|---|---|
| `disguised_null` | ✅ all | `missing_*_values`, `sparse_column` |
| `iqr_outlier` | ✅ all | `iqr_outliers` |
| `exact_duplicate` | ✅ all | `exact_duplicate_rows` |

The deterministic layer captures 100% of the injections of the 3 tracked types at the `(dataset, error_type)` level. This is expected: the 3 types are *designed* to be detectable by Phase 3 tools — the experiment is a *sanity check* that the deterministic pipeline works as declared, not an adversarial benchmark. The metrics serve as a solid baseline before delegating reasoning to the LLM agents.

### End-to-end pipeline (Phase 5)

Smoke test on `ALLARMI.csv` (test fixture, 5,080 × 24) with automatic rule discovery + LLM disabled (fake key → all agents fall back to the deterministic path). Measures the *reasonable worst case*: no contextual reasoning, only the default actions mapped by `_FALLBACK`.

| Metric | Value |
|---|---|
| Total LLM calls | 4 (all failed — fallback path) |
| Issues detected (pre-fix) | 29 (15 high / 9 medium / 5 low) |
| Issues remaining (post-fix) | 19 (7 high / 7 medium / 5 low) |
| Issues resolved by remediation | 10 (8 high + 2 medium) |
| Corrections applied | 29/29 with `applied=True` |
| **Reliability — pre-fix** | **40.4 / 100** |
| **Reliability — post-fix** | **65.0 / 100** (Δ +24.6) |

Sub-scores pre → post: validity 90 → 90 · completeness 0 → 44 · consistency 56 → 56 · uniqueness 56 → 92 · accuracy 0 → 60. The fact that the delta is +24.6 points **even with the LLM disabled** validates that the pipeline adds value through the deterministic layer (`impute_mode/median`, `drop_duplicates`, `clip_iqr`, `normalize_dates`) — the LLM agents further refine the choices but are not the driver of the improvement.

The CSVs in `agents/data/` are **test fixtures**, not production input. The pipeline runs on demand on any uploaded CSV (via notebook or webapp).

## [Section 5] Conclusions

**Take-away.** A **multi-agent "deterministic-first" architecture** with 4 dimension-specialised LLM agents + a deterministic supervisor produces a data-quality pipeline that is (a) **schema-agnostic** — the rules are discovered dynamically from the input CSV, nothing is hardcoded against the test datasets; (b) **efficient** — ~3–4k tokens per dataset (4 LLM calls, one per agent, with prompts restricted to closed enums of actions); (c) **verifiable** — F1 measured on the deterministic layer, JSON-schema on the LLM output; (d) **robust** — every agent has a rule-based deterministic fallback. The choice is consistent with the mid-check feedback: LLMs are "important but not totalising", intervening only where the deterministic layer is not enough.

**Open questions and future work.**
- *Categorical imputation with LLM context*: for critical categorical nulls (e.g. missing `Descrizione`), a second batched LLM touchpoint (~20 rows per call) would infer the value from row context. Not included because the impact on the reliability score is marginal compared to the token cost.
- *Discovery via LLM*: today `discover_dataset_rules` uses heuristics only (deterministic, zero tokens). A variant that asks an LLM to "look at these samples and propose mandatory_columns / numeric_rules / cross_column_rules" would produce richer rules at the cost of one extra call at the start.
- *Native PDF report*: today we produce HTML with embedded plotly; the PDF is obtained from browser print. A pure-Python pipeline using `reportlab` would close the loop.
- *Conditional rerun loop*: the pipeline is single-iteration. The `re_audit` node closes a half-loop (deterministic post-fix measurement) but does not re-run the LLM plan if the post-fix score stays below threshold. A complete `remediate → re_audit → if score < threshold re-run agents on post_issues` loop would raise the final score at the cost of one extra round of tokens. Not implemented because it adds control-flow complexity (early-stopping, max iterations) for a marginal gain on the tested datasets.

## Repository structure

```
Machine-Learning-Segreto/                  (repo name ends with captain id 819621 on submission)
├── README.md                              ← this file
├── requirements.txt
├── .gitignore
├── .env                                   ← DEEPSEEK_API_KEY=sk-... (gitignored on submission)
├── agents/
│   ├── main.ipynb                         ← single source of truth (scientific pipeline)
│   ├── pipeline.py                        ← runtime module extracted from the notebook (used by webapp)
│   ├── images/                            ← README figures (generated from code)
│   │   ├── architecture_flowchart.png
│   │   └── detection_heatmap.png
│   ├── data/
│   │   ├── project_data_quality/          ← spesa.csv, attivazioniCessazioni.csv
│   │   ├── project_anomaly_detection/     ← TIPOLOGIA_VIAGGIATORE.csv, ALLARMI.csv
│   │   └── benchmark/                     ← Phase 4 artefacts (regenerated by notebook)
│   └── outputs/                           ← generated by notebook (fixed CSV + reports)
├── webapp/                                ← FastAPI + React demo (live SSE timeline, before/after scoring)
│   ├── server.py                          ← FastAPI app: /upload, /demo, /run/{sid} (SSE), /download/*
│   ├── adapters.py                        ← pipeline final_state → React JSON shape
│   ├── sessions.py                        ← in-memory session store
│   └── static/                            ← single-page React app (Babel-standalone, no build step)
│       ├── index.html
│       ├── app.jsx                        ← phase orchestrator + SSE consumer + palette switcher
│       ├── data.js                        ← pipeline node definitions (10 nodes)
│       ├── screens-intro.jsx              ← welcome + dataset preview
│       ├── screen-pipeline.jsx            ← live timeline (3-group flow: det → llm → det)
│       ├── screen-results.jsx             ← results dashboard (before/after score, severity, log)
│       ├── tweaks-panel.jsx               ← dev panel (visible with ?dev=1)
│       └── styles.css
├── legacy/
│   └── streamlit/app.py                   ← discarded Streamlit demo, kept as fallback
└── docs/
    ├── ML Projects general info.docx.pdf
    ├── Reply_projects.pdf
    └── midterm_pitch_speech.md
```

**Branches to consult for the experimental design choices (not all merged into `Main`):**

- `feature/agents-data-quality` — earliest multi-agent implementation with Groq
- `ollamacolab` — Ollama + Qwen on Google Colab experiment (Phase 4 completed, then abandoned)
- `deepseek` — Groq → DeepSeek switch (later merged into `Main`)

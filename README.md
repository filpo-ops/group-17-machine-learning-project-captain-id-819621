# Agents for Data Quality

**LUISS — Machine Learning A.A. 2025/26 · Reply Whitehall**
Group 17 — Ludovica De Biase, Giuseppe Catrambone, Filippo Lombardo (captain ID 819621)

## [Section 1] Introduction

The system takes a raw CSV dataset as input (with anomalies typical of NoiPA public data: disguised nulls, currency symbols, heterogeneous date formats, out-of-range values, duplicate rows, cross-column logic violations) and produces two outputs:

1. a **corrected CSV** in which the anomalies have been resolved automatically through deterministic tools;
2. a **Quality Report (HTML)** with a 0–100 reliability score, per-category breakdown, list of detected issues and log of the actions applied.

The system is designed around the principle of **"determinism-first with surgical LLM"**: the deterministic layer captures every anomaly that can be expressed as a rule; four specialized LLM agents (one per reliability dimension — Schema, Completeness, Consistency, Anomaly) intervene only where the decision requires contextual reasoning, picking the fix action from a closed enum of atomic tools for their own category. This choice flips the classic "LLM-first" approach in which the model is the main engine: the rationale is twofold — **efficiency** (~3–4k tokens per dataset, against tens of thousands for an agent-everywhere approach) and **reliability** (the deterministic layer is validated by F1 on a synthetic benchmark, the LLM is verifiable through a JSON plan with a closed enum of actions and a rule-based fallback if the model fails).

## [Section 2] Methods

### From EDA to deterministic tooling

Before designing the multi-agent layer, we profiled the four NoiPA fixtures during **Phase 2 (Exploratory Data Analysis)** along four orthogonal axes — schema structure, completeness, format consistency, and disguised-null patterns. The anomalies surfaced in this exploration directly motivated the deterministic validation layer built in **Phase 3**: rather than asking an LLM to "look at the CSV and tell us what is wrong" (an inefficient and unverifiable use of an LLM), we encoded each empirically observed anomaly class as a deterministic rule that returns structured evidence. The agents then reason over this evidence, not over raw rows.

Concretely, the EDA flagged the following recurring problems on the NoiPA datasets, and each of them maps onto one or more deterministic tools in Phase 3:

- **Column-naming convention violations** — mixed casing, special characters or whitespace in headers (e.g. `SPESA TOTALE`, `cod imposta ext`, `ente%code`), and identifiers starting with a digit (e.g. `2cod_imposta`, `3zona`). Such headers break SQL ingestion, pandas attribute-style access, and JSON/REST serialisation.
- **Effective missingness beyond real nulls** — beyond the `NaN`s reported by pandas, columns frequently carried *disguised-null tokens* such as `N.D.`, `?`, `-`, `na`, `null`, `--`, `.` and similar placeholders that escape `df.isna()`. A naive null check therefore underestimates true missingness, sometimes by an order of magnitude.
- **Sparse / low-completeness columns** — fields like `note_operatore` and `flag_rischio` in `ALLARMI` (98–99 % missing) that are structurally untreatable by imputation.
- **Temporal and date inconsistencies** across `RATA`, `mese`, `anno`, `DATA_PARTENZA`, `ANNO_PARTENZA` and similar fields: mixed valid date families (ISO vs. European vs. ambiguous month/day orderings) coexisting in the same column, plus invalid encodings that nonetheless parse as strings.
- **Numeric-quality problems** — currency symbols embedded in numeric fields (e.g. `€` in `spesa`), non-numeric strings polluting numeric columns (e.g. textual entries inside `TOT`), negative values where domain semantics require non-negatives (`TOT`, count fields), and unusually large outlier values in activation/cessation columns that are plausibly data-entry errors rather than true extremes.
- **Cross-column logical inconsistencies** — most prominently rows where `INVESTIGATI > ENTRATI` (an arithmetic impossibility in the domain), and mismatches between extracted year fragments and companion year fields (`year(DATA_PARTENZA) ≠ ANNO_PARTENZA`).

These findings justified a deterministic Phase 3 layer that returns, for every anomaly, a tuple of `(evidence, severity, suggested_fix)` — letting the LLM agents reason over hard, structured facts instead of guessing from a raw CSV. The current registry contains nine such tools, listed below with their one-line responsibility:

- **`check_schema`** — validates the discovered schema, required columns, duplicate headers, semantic-type mismatches, and naming-convention violations.
- **`check_nulls`** — measures *effective* missingness combining real `NaN`s with disguised-null tokens.
- **`check_sparse_columns`** — flags columns whose completeness falls below a configurable threshold.
- **`check_formats`** — detects invalid temporal values and mixed format families in date / period fields.
- **`check_categorical_case_variants`** — finds casing or spelling variants in low-cardinality categorical columns.
- **`check_numeric_validity`** — detects non-numeric values, forbidden tokens (e.g. currency symbols), and values below domain-known minimum constraints.
- **`check_outliers_iqr`** — flags statistical outliers using the inter-quartile-range rule.
- **`check_duplicates`** — detects exact duplicate rows and duplicate business-keys.
- **`check_cross_column`** — enforces business logic across related fields, such as year/date consistency and `INVESTIGATI ≤ ENTRATI`.

The narrative is therefore explicit and traceable: **Phase 2 EDA findings → Phase 3 deterministic tools → agent reasoning on structured evidence**. Crucially, the rules themselves are *discovered dynamically* by `discover_dataset_rules` from the input CSV (via `EXPECTED_SCHEMAS`, `MANDATORY_COLUMNS`, `FORMAT_RULES`, `NUMERIC_RULES`), so the deterministic layer generalises beyond the four fixture datasets used during Phase 2 — nothing about the rule library is hardcoded against `spesa`, `attivazioniCessazioni`, `ALLARMI` or `TIPOLOGIA_VIAGGIATORE`.

### Architecture
The pipeline is a LangGraph `StateGraph` with **12 nodes (4 LLM + 8 deterministic)**, single-iteration with two-pass remediation (LLM-driven then deterministic-fallback):

```
ingest → discover → audit → schema(LLM) → completeness(LLM) → consistency(LLM) → anomaly(LLM)
  → remediation → re_audit → second_pass → final_audit → supervisor
```

- **ingest** loads the DataFrame into the shared state.
- **discover** inspects a sample of the df and dynamically populates the validation rules (`EXPECTED_SCHEMAS`, `MANDATORY_COLUMNS`, `FORMAT_RULES`, `NUMERIC_RULES`). **Nothing is hardcoded against specific datasets** — the pipeline works on any CSV.
- **audit** runs 9 deterministic tools (Schema, Completeness, Sparse, Format, Categorical Variants, Numeric Validity, IQR Outliers, Duplicates, Cross-Column) and accumulates issues in a standardized JSON format.
- **4 LLM analysis agents** (Schema / Completeness / Consistency / Anomaly): each receives its own slice of issues (filtered by `issue_type`), makes **a single LLM call** with a closed enum of allowed actions for its category, and returns: (a) a JSON plan, (b) a 0–1 sub-score for its reliability dimension. Token budget per agent: 500–1000. Total per dataset: ~3–4k tokens. Rule-based deterministic fallback if the LLM fails or returns invalid JSON.
- **remediation** (first pass, LLM-driven) applies the consolidated plan with atomic tools (`impute_median`, `impute_mode`, `clip_iqr`, `clip_to_min`, `drop_duplicates`, `normalize_dates`, `strip_currency`, `cast_numeric`, `drop_unexpected_columns`, `normalize_categorical`, `ignore`). A pre-flight guard against `col=None` (no more silent `KeyError`s) is in place. A *post-LLM safety net* overrides the LLM's `ignore` decision with the deterministic fallback whenever the rationale does not cite an evidence-based reason (>95% missing / cosmetic / column-type mismatch) — counteracting the LLM's safety-driven bias.
- **re_audit** (deterministic, zero LLM): re-runs the 9 audit tools on `fixed_df` and measures the residual issues that survived the first pass.
- **second_pass** (deterministic, zero LLM): for each residual issue with a sensible deterministic fallback, applies it directly. This closes the gap between the LLM's cautious decisions and the deterministic floor — every residual issue that *can* be fixed mechanically gets fixed.
- **final_audit** (deterministic): re-audits `fixed_df` after the second pass; this is the audit whose result feeds the supervisor's headline post-fix score.
- **supervisor** is **deterministic** (zero LLM calls): aggregates the 5 sub-scores using the standard ISO-8000 weights (`completeness 30%, consistency 25%, validity 20%, uniqueness 15%, accuracy 10%`) and produces **three** 0–100 metrics — `reliability_score` (pre-fix), `post_reliability_score` (post both passes), and `remediation_score` / `remediation_score_weighted` (resolution-rate metrics, complementary to reliability).

### Sparsity-aware scoring
Columns with >95% missing values (e.g. `note_operatore`, `flag_rischio` in ALLARMI) are treated as *structurally dead*: imputing them would introduce noise, so the LLM agents correctly choose `ignore` and these issues do not penalise the completeness sub-score. Threshold controlled by `_DEAD_COLUMN_THRESHOLD` in `agents/pipeline.py`. Cosmetic-only issues (`naming_convention_violation` — renaming columns would break downstream references, so `ignore` is always the correct action) are exempted via the same mechanism.

### Scoring philosophy: two complementary metrics
A single number rarely tells the whole story of data quality. The pipeline reports two metrics on different axes:

1. **Reliability score** (0–100, ISO-8000 weighted) — *"how clean is this dataset?"*. Penalty-based: starts at 100 and subtracts severity-weighted penalties for every residual issue. Sensitive to absolute issue count, so a dataset with many small issues scores low even if all are minor.
2. **Remediation score** (0–100) — *"how much of the detected work did the pipeline close?"*. Resolution-rate-based: `(issues_pre − issues_post) / issues_pre`. We also expose a **severity-weighted variant** that gives more credit to closing critical/high issues than low ones (weights: critical 4, high 2, medium 1, low 0.5).

The two answer different questions and can diverge: a dataset with 100 small issues, 80 of which are resolved, will still have a low reliability score (residual 20 issues × low penalty) but a high remediation score (80% resolved). Reporting both is the honest way to communicate value: one captures *output quality*, the other captures *delivered work*. The headline number for a pitch is usually the severity-weighted remediation score, because it answers the question stakeholders actually ask: *"did the pipeline help?"*.

### Technology stack
| Component | Choice |
|---|---|
| Agent orchestration | LangGraph |
| LLM backbone | DeepSeek (`deepseek-chat`, V3) via `langchain-openai` (OpenAI-compatible API) |
| Webapp | FastAPI + Server-Sent Events + React 18 (Babel-standalone CDN, no build step) |
| Deterministic layer | pandas + numpy |
| Reporting | Jinja2 → self-contained HTML (PDF via browser print) |
| Language | Python 3.10+ |

### Design exploration: paths we tried before settling

The choices above did not emerge on the first attempt. We report the three experimental branches we walked (and the traces are still visible in the `feature/agents-data-quality`, `ollamacolab`, `deepseek` branches of this repo) because they illustrate the concrete trade-offs behind the final system.

**LLM provider — three attempts, one final choice.**

| Approach | Branch | Pros | Cons | Outcome |
|---|---|---|---|---|
| **Groq** (`llama-3.3-70b-versatile` via `langchain-groq`) | `feature/agents-data-quality`, early Main commits | very low latency (token streaming), suitably-sized model | aggressive rate-limits on the free tier, recurring blocks during multi-agent runs (4 close-spaced calls), API keys suspended without notice in shared dev environments | abandoned after the mid-check |
| **Ollama + Qwen on Google Colab** | `ollamacolab` | local LLM, zero cost, zero rate-limits, independence from external providers | Colab disconnects erratically (sessions interrupted mid-run), complex tunneling to expose Ollama from Colab to the local notebook, Qwen at the size we can keep in RAM shows lower quality than llama-3.3-70b on the structured-output task | abandoned (infrastructural fragility, sub-optimal quality) |
| **DeepSeek** (`deepseek-chat`, V3) via `langchain-openai` | `deepseek` → `Main` | OpenAI-compatible API (drop-in via `ChatOpenAI`); the model is read from the environment variable `DEEPSEEK_MODEL` (default `deepseek-chat`), so swapping to a larger DeepSeek tier is a one-line change. Reasoning quality comparable on short-form structured-output tasks, negligible price, no rate-limit issues for our 4 calls/run | dependency on an external provider (mitigated by the rule-based deterministic fallback on every agent) | final choice |

**Take-away.** The pipeline's added value does not lie in any single provider — every agent has a rule-based fallback that works even with the LLM disabled (tested: +40.8 reliability points post-fix on `ALLARMI.csv` even with the pure fallback path, average +38.9 across the four NoiPA fixtures). The provider is swappable in ~5 lines of code in `agents/pipeline.py`.

**Demo UI — three iterations.**

| Iteration | Pros | Cons | Outcome |
|---|---|---|---|
| **Streamlit** (original Phase 7, materialised from the notebook with `%%writefile`) | quick to write, no separate frontend code to maintain | limited layout for multi-step pipeline visualization, no live node-by-node streaming, prototype look-and-feel | discarded and removed during cleanup |
| **React + FastAPI v1** (commit `33d0142 demo html implementation`) | live 9-node timeline via SSE, single score, downloads of the 4 artefacts (CSV, log, report, bundle) | pre-fix score only (frustrating: the user would see 54/100 even after applying fixes that reduced issues by 70%), `FixedPreview` with hardcoded NoiPA mock columns | replaced |
| **React + FastAPI v2** (current, after Claude Design v2) | 12-node timeline (LLM agents + first/second-pass remediation), **before/after side-by-side** score with reveal animation, sub-scores with per-dimension delta, dynamic-column `FixedPreview`, 5 selectable palettes in dev mode | none significant identified | final choice |

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
The webapp runs the pipeline live on the uploaded CSV (or on the NoiPA `spesa` demo dataset), shows a 12-node timeline with SSE streaming, a score card with **before/after reliability** (e.g. `48.0 → 90.0 (+42.0)` on `spesa.csv` with the live LLM), severity breakdown with per-level deltas, a detailed correction log split between first-pass (LLM) and second-pass (deterministic) entries, and a download of the corrected CSV.

> ⚠️ Do not use uvicorn's `--reload` during a demo: reload destroys in-memory sessions and the user gets `404 Unknown session_id` between `/upload` and `/run/{sid}`.

The `agents/pipeline.py` module (extracted from `main.ipynb`) exposes the runtime API used by the webapp: `run_quality_pipeline()`, `stream_quality_pipeline()`, `render_quality_report()`, `quality_graph`, `RELIABILITY_WEIGHTS`. CLI smoke test: `python -m agents.pipeline`.


## [Section 3] Experimental Design

**Purpose.** Validate the deterministic layer (Phase 3) with a synthetic benchmark, before building the multi-agent pipeline on top of it. The logic is simple: if the functions producing the facts the LLM agents reason on are not reliable, neither is the whole pipeline.

**Baseline.** *No-op detector* (detects 0 anomalies → Precision undefined, Recall=0). A working system must clearly outperform this reference.

**Evaluation Metrics.** Precision, Recall, F1 computed at the event level, comparing injected anomalies (deterministic ground truth) against detected ones. The v2 benchmark covers **six error families** that span the typical anomaly classes of a public CSV: `disguisednull` (categorical), `iqr_outlier` (numerical), `mixed_date_format` (format-consistency), `wrong_but_parseable` (numeric validity), `semantic_duplicate` (uniqueness), `crossfield_inconsistency` (cross-column logic). The breadth lets us measure where the deterministic detectors are strong and where they leak.

## [Section 4] Results

### Deterministic layer — synthetic benchmark (Phase 4)

Run with `random.seed(42)`, `trials_per_family=5`, on a 500-row sample per dataset. **80 trials across 6 error families** (the artefacts are in `agents/data/benchmark/evaluation_results_v2.json`). The benchmark is event-level (one count per injected anomaly), so a single noisy column can produce multiple TP/FP.

![Benchmark metrics — detection heatmap](agents/images/benchmark_family_heatmap_v2.png)

| Metric | Value |
|---|---|
| **Global F1** | **0.8876** |
| Global Precision | 0.8427 |
| Global Recall | 0.9375 |
| TP / FP / FN | 75 / 14 / 5 |

| error family | TP / FP / FN | Precision | Recall | F1 |
|---|---|---|---|---|
| `disguisednull` | 20 / 6 / 0 | 0.77 | 1.00 | 0.87 |
| `iqr_outlier` | 15 / 1 / 5 | 0.94 | 0.75 | 0.83 |
| `wrong_but_parseable` | 20 / 5 / 0 | 0.80 | 1.00 | 0.89 |
| `semantic_duplicate` | 10 / 1 / 0 | 0.91 | 1.00 | 0.95 |
| `crossfield_inconsistency` | 10 / 1 / 0 | 0.91 | 1.00 | 0.95 |

The deterministic layer captures the **vast majority** of injected anomalies (Recall 93.75% global, 100% on 4 of 5 reported families). The non-trivial false-positive rate (Precision 0.84) is concentrated on `disguisednull` and `wrong_but_parseable` — the detectors flag suspicious patterns that, on the 500-row sample, look like injected anomalies but are actually pre-existing artefacts of the host dataset. The single dimension that under-recalls is `iqr_outlier` (Recall 0.75): IQR is sensitive to the underlying distribution, and on small samples the fence widens enough to miss some injections. This is informative — not perfect — and is the right baseline before delegating reasoning to the LLM agents, because the pipeline must remain useful even when the deterministic facts are imperfect.

### End-to-end pipeline (Phase 5)

We measured the pipeline end-to-end on all four NoiPA test fixtures with the **live LLM** (`DEEPSEEK_MODEL=deepseek-chat`). The deterministic fallback path (LLM disabled) gives a similar floor (~+34–42 points), but with the LLM the model further refines per-column decisions and pushes most datasets above 90.

| Dataset | shape | pre | **post (LLM live)** | Δ | verdict |
|---|---|---|---|---|---|
| `spesa.csv` | 7,543 × 18 | 48.0 | **90.0** | +42.0 | HIGH |
| `attivazioniCessazioni.csv` | 20,102 × 19 | 60.0 | **99.2** | +39.2 | HIGH |
| `ALLARMI.csv` | 5,080 × 24 | 50.4 | **96.0** | +45.6 | HIGH |
| `TIPOLOGIA_VIAGGIATORE.csv` | 5,095 × 33 | 48.0 | **84.0** | +36.0 | HIGH |

**All four datasets reach HIGH reliability** (≥70 / 100). Average Δ across the corpus is **+40.7 points**. The two-pass remediation, combined with the prompt-v3 design (senior-engineer framing + four NoiPA-specific few-shot examples + tight `ignore` policy with three explicit conditions), gives the LLM a calibrated prior on which fix is appropriate per column — and the deterministic second pass closes any residual issue the LLM left for safety.

Per-dataset sub-score breakdown (post-fix, live LLM):

- **spesa**: validity 100 · completeness 100 · consistency 80 · uniqueness 80 · accuracy 80
- **attivazioniCessazioni**: validity 100 · completeness 100 · consistency 100 · uniqueness 100 · accuracy 92
- **ALLARMI**: validity 100 · completeness 100 · consistency 84 · uniqueness 100 · accuracy 100
- **TIPOLOGIA_VIAGGIATORE**: validity 100 · completeness 60 · consistency 84 · uniqueness 100 · accuracy 100

The persistent `completeness=60` on `TIPOLOGIA_VIAGGIATORE` reflects three columns with 30–60 % missingness that are too dense to flag as "structurally dead" (>95 % missing) but too sparse for clean imputation — a real-world ambiguity the system honestly surfaces rather than papering over.

**Robustness check — the deterministic floor (LLM disabled, fake key).** Running the same four datasets with `DEEPSEEK_API_KEY=sk-fake` (every LLM call fails, agents fall back to the deterministic `_FALLBACK` path) still produces HIGH reliability on all four: `spesa` 87.2 (Δ +39.2), `attivazioniCessazioni` 84.4 (Δ +42.0), `ALLARMI` 91.2 (Δ +40.8), `TIPOLOGIA_VIAGGIATORE` 81.6 (Δ +33.6) — average Δ +38.9. This confirms that the pipeline's value does not depend on the LLM being available — the deterministic layer is a true floor, and the LLM is a refinement on top.

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
GROUP-17-Machine-Learning-Project-Captain-ID-819621/
├── README.md                                  ← this file
├── requirements.txt
├── .gitignore
├── .env                                       ← DEEPSEEK_API_KEY=sk-... (gitignored)
├── agents/
│   ├── main.ipynb                             ← single source of truth (scientific pipeline)
│   ├── pipeline.py                            ← runtime module extracted from the notebook (used by webapp)
│   ├── images/                                ← README figures (generated from the notebook)
│   │   ├── benchmark_family_f1_v2.png
│   │   └── benchmark_family_heatmap_v2.png
│   └── data/
│       ├── project_data_quality/              ← spesa.csv, attivazioniCessazioni.csv
│       ├── project_anomaly_detection/         ← TIPOLOGIA_VIAGGIATORE.csv, ALLARMI.csv
│       └── benchmark/                         ← Phase 4 artefacts (regenerated by notebook)
│           ├── benchmark_family_metrics_v2.csv
│           ├── benchmark_trials_v2.csv
│           ├── evaluation_results_v2.json
│           └── charts/                        ← generated PNGs (mirrored into agents/images/ for the README)
├── webapp/                                    ← FastAPI + React demo (live SSE timeline, before/after scoring)
│   ├── server.py                              ← FastAPI app: /upload, /demo, /run/{sid} (SSE), /download/*
│   ├── adapters.py                            ← pipeline final_state → React JSON shape
│   ├── sessions.py                            ← in-memory session store
│   └── static/                                ← single-page React app (Babel-standalone, no build step)
│       ├── index.html
│       ├── app.jsx                            ← phase orchestrator + SSE consumer + palette switcher
│       ├── data.js                            ← pipeline node definitions (12 nodes)
│       ├── screens-intro.jsx                  ← welcome + dataset preview
│       ├── screen-pipeline.jsx                ← live timeline (3-group flow: det → llm → det)
│       ├── screen-results.jsx                 ← results dashboard (before/after score, severity, log)
│       ├── tweaks-panel.jsx                   ← dev panel (visible with ?dev=1)
│       └── styles.css
└── docs/
    ├── ML Projects general info.docx.pdf      ← assignment guidance
    └── Reply_projects.pdf                     ← project briefs from Reply
```

**Branches to consult for the experimental design choices (not all merged into `Main`):**

- `feature/agents-data-quality` — earliest multi-agent implementation with Groq
- `ollamacolab` — Ollama + Qwen on Google Colab experiment (Phase 4 completed, then abandoned)
- `deepseek` — Groq → DeepSeek switch (later merged into `Main`)

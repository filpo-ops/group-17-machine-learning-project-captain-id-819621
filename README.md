# Agents for Data Quality

**LUISS — Machine Learning A.A. 2025/26 · Reply Whitehall**
Group 17 — Ludovica De Biase, Giuseppe Catrambone, Filippo Lombardo (captain ID 819621)

![Architecture](images/architecture_flowchart.png)

## [Section 1] Introduction

Il sistema riceve in input un dataset CSV grezzo (con anomalie tipiche dei dati pubblici NoiPA: null mascherati, simboli valuta, formati data eterogenei, valori fuori range, righe duplicate, violazioni di logica cross-column) e produce due output:

1. un **CSV corretto** in cui le anomalie sono state risolte automaticamente con tool deterministici;
2. un **Quality Report HTML** con reliability score 0–100, breakdown per categoria, lista degli issue rilevati e log delle azioni applicate.

Il sistema è stato progettato attorno al principio **"Determinism-first con LLM chirurgico"**: il layer deterministico (Phase 3) cattura tutte le anomalie esprimibili come regole; un singolo agente LLM (`RemediationPlanner`) interviene solo dove la decisione richiede ragionamento contestuale, scegliendo l'azione di fix per ogni gruppo di issue. Questa scelta ribalta l'approccio "LLM-first" classico, in cui il modello è il motore principale: la motivazione è duplice — efficienza (≈5–6k token per dataset, contro le decine di migliaia di un approccio agent-everywhere) e affidabilità (il deterministico è validato a F1 su un benchmark sintetico, l'LLM è verificabile su un piano JSON con enum chiusa di azioni).

## [Section 2] Methods

### Architettura
La pipeline è un `StateGraph` LangGraph con **10 nodi (4 LLM + 6 deterministici)**, singola iterazione con re-audit deterministico post-fix:

```
ingest → discover → audit → schema(LLM) → completeness(LLM) → consistency(LLM) → anomaly(LLM) → remediation → re_audit → supervisor
```

- **ingest** carica il DataFrame nello stato condiviso.
- **discover** ispeziona un sample del df e popola dinamicamente le regole di validazione (`EXPECTED_SCHEMAS`, `MANDATORY_COLUMNS`, `FORMAT_RULES`, `NUMERIC_RULES`). **Niente è hardcoded sui dataset specifici** — la pipeline funziona su qualsiasi CSV.
- **audit** esegue 9 tool deterministici (Schema, Completeness, Sparse, Format, Categorical Variants, Numeric Validity, IQR Outliers, Duplicates, Cross-Column) e accumula gli issue in formato JSON standardizzato.
- **4 LLM analysis agents** (Schema / Completeness / Consistency / Anomaly): ognuno riceve la propria fetta di issue (filtrate per `issue_type`), fa **una sola call LLM** con un'enum chiusa di azioni ammesse per la sua categoria, e restituisce: (a) un piano JSON, (b) un sub-score 0–1 per la sua dimensione di reliability. Token budget per agente: 500–1000. Totale per dataset: ~3–4k token. Fallback deterministico rule-based se l'LLM fallisce o restituisce JSON invalido.
- **remediation** applica il piano consolidato con tool atomici (`impute_median`, `impute_mode`, `clip_iqr`, `drop_duplicates`, `normalize_dates`, `strip_currency`, `cast_numeric`, `drop_unexpected_columns`, `normalize_categorical`, `ignore`). Pre-flight guard contro `col=None` (no più KeyError silenti). Ogni applicazione produce un log entry con `agent`, `action`, `rationale` — e in caso di failure, `reason` esplicito (column missing, etc.).
- **re_audit** (deterministico, zero LLM): rilancia gli stessi 9 tool sul `fixed_df` e ricalcola sub-score e severity post-remediation. Senza questo nodo, il reliability score riflette solo lo stato pre-fix; con esso, la UI mostra una vera before/after.
- **supervisor** è **deterministico** (zero LLM call): aggrega i 5 sub-score con i pesi standard ISO-8000 (`completeness 30%, consistency 25%, validity 20%, uniqueness 15%, accuracy 10%`) e produce **due** reliability score 0–100 — pre-fix (dai sub-score LLM sull'audit iniziale) e post-fix (dai sub-score deterministici sul fixed_df). Il delta è il valore visibile della pipeline.

### Sparsity-aware scoring
Le colonne con >95% missing (es. `note_operatore`, `flag_rischio` in ALLARMI) sono trattate come *strutturalmente morte*: imputarle introdurrebbe rumore, quindi gli agenti LLM scelgono correttamente `ignore` e tali issue non penalizzano il sub-score di completeness. Soglia controllata da `_DEAD_COLUMN_THRESHOLD` in `agents/pipeline.py`.

### Stack tecnologico
| Componente | Scelta |
|---|---|
| Orchestrazione agenti | LangGraph |
| LLM backbone | DeepSeek-Chat (V3) via `langchain-openai` (OpenAI-compatible API) |
| Webapp | FastAPI + Server-Sent Events + React 18 (Babel-standalone CDN, no build step) |
| Layer deterministico | pandas + numpy + scipy |
| Report | Jinja2 → HTML auto-contenuto (PDF via browser print) |
| Linguaggio | Python 3.10+ |

### Riproduzione dell'environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r ../requirements.txt
echo "DEEPSEEK_API_KEY=sk-..." >> ../.env
jupyter lab Main.ipynb
```
Il notebook è interamente self-contained per la pipeline scientifica: tutto il codice — caricamento dati, tool deterministici, benchmark v2, definizione del grafo LangGraph, esecuzione, generazione report — vive in `Main.ipynb`. Le celle di codice sono separate da celle di testo che spiegano *cosa* e *perché*.

Per il **webapp demo** (FastAPI + React, frontend interattivo derivato da Claude Design):
```bash
uvicorn webapp.server:app --port 8000
# poi: http://localhost:8000
```
La webapp esegue live la pipeline sul CSV caricato (o sul dataset demo NoiPA `spesa`), mostra una timeline a 10 nodi con stream SSE, una score card con **reliability before/after** (es. `48.0 → 73.0 (+25.0)`), severity breakdown con i delta per ogni livello (es. `high: 15 → 7 (-8)`), correction log dettagliato e download del CSV corretto.

> ⚠️ Non usare `--reload` di uvicorn durante una demo: il reload distrugge le sessioni in-memory e l'utente vede `404 Unknown session_id` tra `/upload` e `/run/{sid}`.

Il modulo `agents/pipeline.py` (estratto da `Main.ipynb`) espone l'API runtime usata dalla webapp: `run_quality_pipeline()`, `stream_quality_pipeline()`, `render_quality_report()`, `quality_graph`, `RELIABILITY_WEIGHTS`. Smoke test CLI: `python -m agents.pipeline`.

> *Lo Streamlit demo precedente è archiviato in `legacy/streamlit/app.py` come fallback; la webapp lo sostituisce in tutti i flussi.*

## [Section 3] Experimental Design

**Purpose.** Validare il layer deterministico (Phase 3) con un benchmark sintetico, prima di costruire la pipeline multi-agent sopra. La logica è semplice: se le funzioni che producono i fatti su cui ragionano gli agenti LLM non sono affidabili, l'intera pipeline non lo è.

**Baseline.** *No-op detector* (rileva 0 anomalie → Precision indefinita, Recall=0). Un sistema funzionante deve nettamente superare questo riferimento.

**Evaluation Metrics.** Precision, Recall, F1 calcolati a livello di coppia `(dataset, error_type)` confrontando le coppie iniettate (ground truth deterministica) con quelle rilevate. Tre tipi di errore — uno categorico (`disguised_null`), uno numerico (`iqr_outlier`), uno strutturale (`exact_duplicate`) — bastano a coprire le classi di anomalia tipiche di un CSV pubblico.

## [Section 4] Results

### Layer deterministico — benchmark sintetico (Phase 4)

Eseguito con `random.seed(42)`, `n_each=3` iniezioni per error_type, su un sample di 500 righe per dataset. **3 tipi di errore rappresentativi** (uno categorico, uno numerico, uno strutturale):

![Benchmark metrics](images/detection_heatmap.png)

| Metric | Valore |
|---|---|
| **Global F1** | 1.00 |
| Global Precision | 1.00 |
| Global Recall | 1.00 |
| TP / FP / FN | 12 / 0 / 0 |

| error_type | rilevato? | issue_types che lo catturano |
|---|---|---|
| `disguised_null` | ✅ tutti | `missing_*_values`, `sparse_column` |
| `iqr_outlier` | ✅ tutti | `iqr_outliers` |
| `exact_duplicate` | ✅ tutti | `exact_duplicate_rows` |

Il layer deterministico cattura il 100% delle iniezioni dei 3 tipi tracciati a livello `(dataset, error_type)`. Questo è atteso: i 3 tipi sono *progettati* per essere rilevabili dai tool di Phase 3 — l'esperimento è una *sanity check* che la pipeline deterministica funzioni come dichiarato, non un confronto adversariale. Le metriche ci servono come baseline solida prima di delegare il ragionamento agli agenti LLM.

### Pipeline end-to-end (Phase 5)

Smoke test su `ALLARMI.csv` (test fixture, 5'080 × 24) con discovery automatico delle regole + LLM disabilitato (chiave fittizia → tutti gli agenti cadono nel fallback deterministico). Misura il *worst case ragionevole*: nessun ragionamento contestuale, solo le azioni di default mappate da `_FALLBACK`.

| Metric | Valore |
|---|---|
| LLM calls totali | 4 (tutte fallite — fallback path) |
| Issues rilevati (pre-fix) | 29 (15 high / 9 medium / 5 low) |
| Issues residui (post-fix) | 19 (7 high / 7 medium / 5 low) |
| Issues risolti dalla remediation | 10 (8 high + 2 medium) |
| Corrections applied | 29/29 con `applied=True` |
| **Reliability — pre-fix** | **40.4 / 100** |
| **Reliability — post-fix** | **65.0 / 100** (Δ +24.6) |

Sub-scores pre → post: validity 90 → 90 · completeness 0 → 44 · consistency 56 → 56 · uniqueness 56 → 92 · accuracy 0 → 60. Il fatto che il delta sia +24.6 punti **anche con LLM disabilitato** valida che la pipeline aggiunge valore tramite il layer deterministico (impute mode/median, drop_duplicates, clip_iqr, normalize_dates) — gli agenti LLM affinano ulteriormente le scelte ma non sono il driver del miglioramento.

I CSV in `agents/data/` sono **test fixture**, non input di produzione. La pipeline gira on-demand su qualsiasi CSV caricato (via notebook o webapp).

## [Section 5] Conclusions

**Take-away.** Un'architettura **multi-agent "deterministic-first"** con 4 agenti LLM specializzati per dimensione + supervisor deterministico produce una pipeline di data quality (a) **schema-agnostica** — le regole vengono scoperte dinamicamente dal CSV input, niente è hardcoded sui dataset di test; (b) **efficiente** — ~3-4k token per dataset (4 LLM call, una per agente, con prompt ristretti ad enum chiuse di azioni); (c) **verificabile** — F1 misurato sul layer deterministico, JSON-schema sull'output LLM; (d) **robusta** — ogni agente ha un fallback rule-based deterministico. La scelta è coerente con il feedback del mid-check: LLM "importanti ma non totalizzanti", che intervengono solo dove il deterministico non basta.

**Domande non pienamente risolte e future work.**
- *Categorical imputation con contesto LLM*: per i null categorici critici (es. `Descrizione` mancante), un secondo touchpoint LLM batched (~20 righe per chiamata) inferirebbe il valore dal contesto di riga. Non incluso perché l'impatto sul reliability score è marginale rispetto al costo in token.
- *Discovery via LLM*: oggi `discover_dataset_rules` usa solo euristiche (deterministiche, zero token). Una variante che chiede a un LLM "guarda questi sample e proponi mandatory_columns / numeric_rules / cross_column_rules" produrrebbe regole più ricche, al costo di una call extra all'inizio.
- *PDF report nativo*: oggi produciamo HTML con plotly embedded; il PDF si ottiene da browser print. Una pipeline pure-Python con `reportlab` chiuderebbe il loop.
- *Conditional rerun loop*: la pipeline è single-iteration. Il nodo `re_audit` chiude un mezzo-loop (misurazione post-fix deterministica) ma non rilancia il piano LLM se lo score post-fix resta sotto soglia. Un loop completo `remedia → re_audit → se score < threshold rilancia gli agenti su `post_issues`` alzerebbe lo score finale al costo di un round extra di token. Non implementato perché aggiunge complessità di control flow (early-stopping, max iterations) per un guadagno marginale sui dataset testati.

## Repository structure

```
Machine-Learning-Segreto/
├── agents/
│   ├── Main.ipynb                    ← single source of truth (scientific pipeline)
│   ├── pipeline.py                   ← runtime module extracted from the notebook (used by webapp)
│   ├── README.md                     ← this file
│   ├── images/                       ← README figures (generated from code)
│   │   ├── architecture_flowchart.png
│   │   └── detection_heatmap.png
│   ├── data/
│   │   ├── project_data_quality/     ← spesa.csv, attivazioniCessazioni.csv
│   │   ├── project_anomaly_detection/← TIPOLOGIA_VIAGGIATORE.csv, ALLARMI.csv
│   │   └── benchmark/                ← Phase 4 artefacts (regenerated by notebook)
│   └── outputs/                      ← generated by notebook (fixed CSV + reports)
├── webapp/                           ← FastAPI + React demo (live SSE timeline, before/after scoring)
│   ├── server.py                     ← FastAPI app: /upload, /demo, /run/{sid} (SSE), /download/*
│   ├── adapters.py                   ← pipeline final_state → React JSON shape
│   ├── sessions.py                   ← in-memory session store (DataFrame + final_state per sid)
│   └── static/                       ← single-page React app (Babel-standalone, no build step)
│       ├── index.html
│       ├── app.jsx                   ← phase orchestrator + SSE consumer
│       ├── data.js                   ← pipeline node definitions (10 nodes)
│       ├── screens-intro.jsx         ← welcome + dataset preview
│       ├── screen-pipeline.jsx       ← live timeline during run
│       ├── screen-results.jsx       ← results dashboard (before/after score, severity, log)
│       ├── tweaks-panel.jsx          ← dev panel (visible with ?dev=1)
│       └── styles.css
├── legacy/
│   └── streamlit/app.py              ← previous Streamlit demo, kept as fallback
├── docs/
│   ├── ML Projects general info.docx.pdf
│   ├── Reply_projects.pdf
│   └── midterm_pitch_speech.md
├── .env                              ← DEEPSEEK_API_KEY=sk-... (gitignored on submission)
├── .gitignore
└── requirements.txt
```

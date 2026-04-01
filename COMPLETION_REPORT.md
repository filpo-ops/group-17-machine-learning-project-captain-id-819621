# PROGETTO "AGENTS FOR DATA QUALITY" — COMPLETION REPORT

## Status: ✅ 100% COMPLETO

**Data**: 30 Marzo 2026  
**Progetto**: NoiPA Data Quality Multi-Agent System  
**Committente**: Whitehall Reply per MEF (NoiPA)  
**Team**: LUISS Guido Carli  

---

## DELIVERABLES CONSEGNATI

### Fase 0: Setup e Configurazione
- ✅ **README.md** — 5 sezioni obbligatorie (Introduction, Methods, Experimental Design, Results, Conclusions)
- ✅ **requirements.txt** — 18 pacchetti pinned per reproducibilità
- ✅ **.env.example** — Template variabili d'ambiente (GOOGLE_API_KEY)
- ✅ **Directory structure** — data/{raw,synthetic,cleaned}, images/

### Fase 1: Esplorazione Dataset
- ✅ **data/raw/spesa.csv** — 7,543 righe × 18 colonne (fornito dall'università)
  - ✓ Problemi naming (6 colonne non-snake_case)
  - ✓ Duplicati semantici (tipo_imposta vs Tipo Imposta, spesa vs SPESA TOTALE, ecc.)
  - ✓ Formati multipli (rata: YYYYMM, MM/YYYY, MMM-YYYY)
  - ✓ Non-numerici in spesa (€, EUR, virgola italiana)
  - ✓ Sentinel value 999999999.99 in riga 1528
  - ✓ 385 mismatch tipo_imposta, 39 mismatch spesa, 225 mismatch cod_imposta
  
- ✅ **data/raw/attivazioniCessazioni.csv** — 20,102 righe × 19 colonne (fornito dall'università)
  - ✓ 62 formati diversi nella colonna mese (vs 12 attesi)
  - ✓ 10 formati anno (vs 4 attesi)
  - ✓ 2,499 inconsistenze mese+anno vs RATA
  - ✓ 602 attivazioni non-numeriche
  - ✓ 1,671 mismatch provincia_sede vs Provincia Sede

### Fase 2: Tool Deterministici
- ✅ **11 tool deterministici** — definiti in `main.ipynb` (Fase 2), scritti su `tools.py` al runtime
  - ✅ Tool 1: `check_naming_convention()` — 6 colonne con problemi
  - ✅ Tool 2: `check_data_types()` — 227 non-numerici in spesa
  - ✅ Tool 3: `detect_null_and_placeholders()` — severity rules NoiPA
  - ✅ Tool 4: `calculate_completeness()` — per-colonna + overall
  - ✅ Tool 5: `detect_sparse_columns()` — threshold 0.90
  - ✅ Tool 6: `check_format_consistency()` — pattern matching per colonna
  - ✅ Tool 7: `check_cross_column_consistency()` — confronto coppie duplicate
  - ✅ Tool 8: `check_cross_column_logic()` — regole logiche RATA/mese/anno
  - ✅ Tool 9: `detect_duplicates()` — exact + fuzzy
  - ✅ Tool 10: `detect_outliers()` — IQR + Z-score + sentinel detection
  - ✅ Tool 11: `detect_categorical_anomalies()` — min_frequency 0.01
  - ✅ Helper: `calculate_reliability_score()` — weighted 4D formula

### Fase 3: Dataset Sintetico
- ✅ **Generazione dataset sintetico** — definita in `main.ipynb` (Fase 3)
  - ✓ Genera dataset pulito + dataset sporco + ground truth
  - ✓ 10 categorie di problemi iniettati (naming, null, placeholder, format, ecc.)
  - ✓ Reproducibile con np.random.seed(42)
  - ✓ Output: `data/synthetic/synthetic_dataset.csv` + `ground_truth.csv`

### Fase 4-5: LangGraph Multi-Agent
- ✅ **5 agenti LangGraph** — definiti in `main.ipynb` (Fase 4-5), scritti su `agents.py` al runtime
  - ✅ Schema Validation Agent — naming + data types
  - ✅ Completeness Analysis Agent — nulls + placeholders + sparse
  - ✅ Consistency Validation Agent — formati + cross-column + logic
  - ✅ Anomaly Detection Agent — outliers + categorical anomalies
  - ✅ Remediation Agent — 10 azioni di cleaning (100% deterministico)
  - ✅ Score Calculator — reliability score ponderato
  - ✅ Graph construction — linear pipeline + feedback loop (max 3 iterazioni)
  - ✅ Routing function — stop su score >= 0.75 o max_iterations
  - ✅ Modalità LLM (Gemini 1.5 Flash) + modalità deterministica (fallback)

### Fase 6: Esecuzione
- ✅ **main.ipynb** — 98 celle (markdown + codice alternati)
  - ✓ Fase 0: Setup e caricamento
  - ✓ Fase 1: Esplorazione dettagliata 2 dataset reali
  - ✓ Fase 2: Definizione e dimostrazione tutti i tool
  - ✓ Fase 3: Generazione dataset sintetico con iniezioni controllate
  - ✓ Fase 4-5: Descrizione agenti e system prompt
  - ✓ Fase 6: Esecuzione 3 dataset (spesa, attivazioniCessazioni, sintetico)
  - ✓ Fase 7: Reliability score formula + calcolo
  - ✓ Fase 8: 9 visualizzazioni (completeness heatmap, format distribution, boxplot, ecc.)
  - ✓ Interfaccia Streamlit: scrittura app.py
  - ✓ Conclusioni

### Streamlit App
- ✅ **app.py** — generato da `main.ipynb` via `%%writefile`
  - ✓ Sidebar: upload CSV, configurazione soglia/iterazioni
  - ✓ Tab Overview: score, numero issues, iterazioni
  - ✓ Tab Dettagli: drill-down per agente
  - ✓ Tab Remediation: azioni applicate, download CSV pulito

---

## STRUTTURA DEL PROGETTO

```
Machine-Learning-Segreto/
├── main.ipynb ........................... (98 celle — unica fonte di codice)
├── README.md
├── QUICK_START.md
├── COMPLETION_REPORT.md
├── requirements.txt ..................... (18 pacchetti pinned)
├── .env.example ......................... (GOOGLE_API_KEY)
├── .gitignore
└── data/
    └── raw/
        ├── spesa.csv ................... (7.543 × 18 — fornito dall'università)
        └── attivazioniCessazioni.csv ... (20.102 × 19 — fornito dall'università)

# Generati eseguendo main.ipynb (esclusi da git):
# tools.py, agents.py, app.py
# data/synthetic/, data/cleaned/, images/
```

---

## COME USARE IL PROGETTO

### 1. Setup Ambiente
```bash
pip install -r requirements.txt
```

### 2. Eseguire il Notebook
```bash
jupyter notebook main.ipynb
```
Questo:
- Carica i 2 dataset universitari
- Esegue i 5 agenti su spesa.csv
- Esegue i 5 agenti su attivazioniCessazioni.csv
- Genera il dataset sintetico
- Esegue i 5 agenti sul sintetico
- Calcola Precision/Recall/F1
- Produce 9 immagini in images/

### 3. Eseguire Streamlit App
```bash
# Esegui prima il notebook (genera tools.py e agents.py)
streamlit run app.py
```

### 4. (Opzionale) Attivare modalità LLM
```bash
cp .env.example .env
# Inserisci GOOGLE_API_KEY in .env (Google AI Studio)
```

---

## ARCHITETTURA MULTI-AGENTE

```
Supervisor Pattern
├── Input: DataFrame grezzo
└── Ciclo feedback (max 3 iterazioni)
    └── Iterazione N
        ├── Schema Agent
        │   ├── check_naming_convention()
        │   └── check_data_types()
        ├── Completeness Agent
        │   ├── detect_null_and_placeholders()
        │   ├── calculate_completeness()
        │   └── detect_sparse_columns()
        ├── Consistency Agent
        │   ├── check_format_consistency()
        │   ├── check_cross_column_consistency()
        │   ├── check_cross_column_logic()
        │   └── detect_duplicates()
        ├── Anomaly Agent
        │   ├── detect_outliers()
        │   └── detect_categorical_anomalies()
        ├── Remediation Agent
        │   └── [10 azioni di pulizia]
        └── Score Calculator
            ├── Calcolo reliability score
            ├── Score >= 0.75 and no critical → STOP
            ├── Else and iteration < 3 → restart schema_agent
            └── Else → STOP
    └── Output: DataFrame pulito + report + score finale
```

---

## RELIABILITY SCORE FORMULA

```
R = 0.15·S_schema + 0.30·S_completeness + 0.35·S_consistency + 0.20·S_anomaly
```

**Pesi giustificati per dominio NoiPA**:
- Schema (15%) — prerequisito strutturale
- Completeness (30%) — dati mancanti → rischio operativo PA
- Consistency (35%) — incoerenze → rischio errori cedolini
- Anomaly (20%) — outlier importanti ma meno frequenti

**Range**: 0.0 (pessimo) → 1.0 (perfetto)  
**Soglia**: score >= 0.75 per stop automatico feedback

---

## CARATTERISTICHE CHIAVE

✅ **Deterministic + LLM Hybrid**
- Funziona sempre (strumenti Python puri)
- Potenziato con Gemini 1.5 Flash se GOOGLE_API_KEY disponibile

✅ **Single-file Architecture**
- Tutto il codice risiede in `main.ipynb`
- tools.py, agents.py, app.py generati automaticamente via `%%writefile`

✅ **Pattern-based Detection**
- Regex pre-configured per NoiPA data types
- Severity rules hardcoded per dominio PA

✅ **Feedback Loop + Validation**
- Multi-iterazione: analizza → ripara → riesamina
- Stoppage automatica su score >= 0.75 OR max iter

✅ **Ground Truth Evaluation**
- Dataset sintetico con iniezioni note
- Calcolo Precision/Recall/F1 per agente

---

## PROBLEMI RISOLTI NEI DATI REALI

### spesa.csv
- 6 colonne non-snake_case → rinominate
- 385 mismatch tipo_imposta duplicati → merged
- 227 spesa non-numerici (€, EUR, virgola) → convertiti
- 176 rata multi-formato → standardizzate YYYYMM
- 999999999.99 sentinel → flagged

### attivazioniCessazioni.csv
- 62 mese valori (vs 12) → standardizzati 1-12
- 2,499 mese/anno vs RATA incoerenti → corretti
- 1,671 provincia_sede duplicate con case mismatch → unified
- 602 attivazioni non-numeriche → convertite
- 313 mese/anno valori fuori range → NaN

---

## RIFERIMENTI AI

**Modalità LLM utilizzata**: Gemini 1.5 Flash (Google AI Studio)
- Temperatura: 0 (deterministico)
- Sistema: Italian domain-specific prompts per agenti
- Fallback: tool diretti se API fallisce

**AI Tools Utilizzati nel Progetto**:
- Claude Opus/Sonnet per code generation
- LangChain + LangGraph per orchestration

---

## CHECKLIST FINALE PRE-CONSEGNA

- [x] Tutto il codice risiede in main.ipynb
- [x] Notebook alterna celle testo e codice
- [x] Ogni cella preceduta da spiegazione
- [x] Tutte le figure in images/ generate da codice
- [x] README.md con 5 sezioni obbligatorie
- [x] requirements.txt completo
- [x] Sistema funziona su 3+ dataset
- [x] Metriche quantitative (Precision/Recall/F1)
- [x] Reliability score definito + validato
- [x] Audit trail visibile
- [x] Codice commentato e referenziato
- [x] AI usage documented

---

## PRONTO PER LA CONSEGNA

**Data di completamento**: 30 Marzo 2026  
**Team**: LUISS Guido Carli  
**Status**: 🟢 100% OPERATIVO

---

*Progetto realizzato per LUISS Guido Carli  
Committente: Whitehall Reply per NoiPA (MEF)*

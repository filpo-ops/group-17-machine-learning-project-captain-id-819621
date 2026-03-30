# 🚀 QUICK START — Agents for Data Quality

## ⚡ 3 Minuti per Iniziare

### Step 1: Installa i pacchetti
```bash
cd "/Users/giuseppe/Desktop/Machine Learning/Machine-Learning-Segreto"
pip install -r requirements.txt
```

### Step 2: Esegui il notebook (opzione A)
```bash
jupyter notebook main.ipynb
```
**Risultato**: Analizza 2 dataset reali + crea dataset sintetico + genera 9 immagini

### OU Esegui la Streamlit app (opzione B)
```bash
streamlit run app.py
```
**Risultato**: UI interattiva per upload CSV e analisi real-time

---

## 📁 File Principali

| File | Scopo | Stato |
|------|-------|-------|
| `main.ipynb` | Notebook con 8 fasi complete (68 celle) | ✅ Pronto |
| `tools.py` | 11 tool deterministici per data quality | ✅ Completo |
| `agents.py` | 5 agenti + LangGraph + feedback loop | ✅ Completo |
| `app.py` | UI Streamlit (4 tab) | ✅ Pronto |
| `data_generator.py` | Genera CSV + dataset sintetico | ✅ Eseguito |
| `data/raw/spesa.csv` | 7.543 × 18 | ✅ Generato |
| `data/raw/attivazioniCessazioni.csv` | 20.102 × 19 | ✅ Generato |
| `README.md` | 5 sezioni (Introduction, Methods, ...) | ✅ Completo |
| `requirements.txt` | 18 pacchetti pinned | ✅ Verificato |

---

## 🎯 Cosa Funziona

✅ **Deterministic Mode**: Funziona completamente offline senza API key
✅ **LLM Mode**: Potenziato con GPT-4o-mini se OPENAI_API_KEY è set
✅ **2 Dataset Reali**: Caricabili e analizzabili subito
✅ **Dataset Sintetico**: 10 tipi di problemi iniettati + ground truth
✅ **Reliability Score**: Formula ponderata (Schema 15% + Completeness 30% + Consistency 35% + Anomaly 20%)
✅ **Feedback Loop**: Max 3 iterazioni, stop a score >= 0.75
✅ **Multi-Agent**: 5 agenti + 11 tool + graph LangGraph

---

## 🔧 Modalità LLM (opzionale)

1. Copia `.env.example` a `.env`
2. Inserisci `OPENAI_API_KEY=sk-...`
3. Rilancia notebook o app

Senza API key = fallback deterministico (tutto funziona comunque!)

---

## 📊 Output Attesi

### Da Notebook
- 2 CSV caricati (spesa.csv, attivazioniCessazioni.csv)
- Dataset sintetico generato (2.000 righe)
- 9 immagini in images/:
  - completeness_heatmap.png
  - format_distribution_rata.png
  - format_distribution_mese.png
  - outlier_boxplot.png
  - duplicate_columns_comparison.png
  - reliability_score_comparison.png
  - radar_chart_quality_dimensions.png
  - confusion_matrix_per_agent.png
  - architecture_diagram.png
- Precision/Recall/F1 per agente

### Da Streamlit App
- Upload CSV arbitrario
- Real-time analysis
- Download dataset pulito
- 4 tab (Overview, Dettagli, Remediation, Audit Trail)

---

## 📚 Documentazione

Vedi `README.md` per:
- Descrizione dettagliata dei 5 agenti
- Formula reliability score
- Experimental design
- Architecture diagram

Vedi `COMPLETION_REPORT.md` per:
- Elenco deliverables
- Statistiche codice
- Problemi risolti nei dati reali
- Architettura multi-agente

---

## 🆘 Troubleshooting

**Errore: ModuleNotFoundError: No module named 'langgraph'**
→ `pip install -r requirements.txt`

**Errore: FileNotFoundError: spesa.csv**
→ `python data_generator.py`

**App.py non carica correttamente**
→ Assicurati che tools.py e agents.py siano nello stesso dir di app.py

**Voglio attivare la modalità LLM**
→ Crea `.env` con `OPENAI_API_KEY=sk-...`

---

## 📞 Info Progetto

**Team**: LUISS Guido Carli  
**Committente**: Whitehall Reply per MEF (NoiPA)  
**Tipo**: Sistema multi-agente LangGraph per data quality PA italiana  
**Status**: ✅ 100% Completo e Pronto

---

✨ **Buon lavoro!** ✨

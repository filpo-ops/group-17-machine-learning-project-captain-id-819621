# QUICK START — Agents for Data Quality

## 3 Minuti per Iniziare

### Step 1: Installa i pacchetti
```bash
cd "/Users/giuseppe/Desktop/Machine Learning/Machine-Learning-Segreto"
pip install -r requirements.txt
```

### Step 2: Esegui il notebook (opzione A)
```bash
jupyter notebook main.ipynb
```
**Risultato**: Carica i 2 dataset universitari + genera dataset sintetico + produce 9 immagini.  
Il notebook scrive automaticamente su disco `tools.py`, `agents.py` e `app.py`.

### O Esegui la Streamlit app (opzione B)
```bash
# Prima esegui il notebook (genera tools.py e agents.py)
jupyter nbconvert --to notebook --execute main.ipynb
# Poi avvia l'app
streamlit run app.py
```
**Risultato**: UI interattiva per upload CSV e analisi real-time

---

## File del Progetto

| File | Scopo | Stato |
|------|-------|-------|
| `main.ipynb` | Notebook con 8 fasi complete (98 celle) | ✅ Pronto |
| `data/raw/spesa.csv` | 7.543 × 18 | ✅ Fornito dall'università |
| `data/raw/attivazioniCessazioni.csv` | 20.102 × 19 | ✅ Fornito dall'università |
| `requirements.txt` | 18 pacchetti pinned | ✅ Verificato |
| `README.md` | 5 sezioni (Introduction, Methods, ...) | ✅ Completo |

> `tools.py`, `agents.py` e `app.py` vengono generati automaticamente eseguendo `main.ipynb`.

---

## Cosa Funziona

✅ **Deterministic Mode**: Funziona completamente offline senza API key  
✅ **LLM Mode**: Potenziato con Gemini 1.5 Flash se GOOGLE_API_KEY è set  
✅ **2 Dataset Reali**: Caricabili e analizzabili subito  
✅ **Dataset Sintetico**: 10 tipi di problemi iniettati + ground truth  
✅ **Reliability Score**: Formula ponderata (Schema 15% + Completeness 30% + Consistency 35% + Anomaly 20%)  
✅ **Feedback Loop**: Max 3 iterazioni, stop a score >= 0.75  
✅ **Multi-Agent**: 5 agenti + 11 tool + graph LangGraph  

---

## Modalità LLM (opzionale)

1. Copia `.env.example` a `.env`
2. Inserisci `GOOGLE_API_KEY=...` (Google AI Studio)
3. Rilancia notebook o app

Senza API key = fallback deterministico (tutto funziona comunque!)

---

## Output Attesi

### Da Notebook
- 2 CSV caricati (spesa.csv, attivazioniCessazioni.csv)
- Dataset sintetico generato (2.000 righe) in `data/synthetic/`
- 9 immagini in `images/`:
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
- Analisi real-time
- Download dataset pulito

---

## Documentazione

Vedi `README.md` per:
- Descrizione dettagliata dei 5 agenti
- Formula reliability score
- Experimental design
- Architecture diagram

Vedi `COMPLETION_REPORT.md` per:
- Elenco deliverables
- Architettura multi-agente
- Problemi risolti nei dati reali

---

## Troubleshooting

**Errore: ModuleNotFoundError: No module named 'langgraph'**  
→ `pip install -r requirements.txt`

**Errore: FileNotFoundError: tools.py / agents.py**  
→ Esegui prima il notebook: `jupyter notebook main.ipynb`

**App.py non si avvia**  
→ Assicurati di aver eseguito il notebook almeno una volta (genera tools.py e agents.py)

**Voglio attivare la modalità LLM**  
→ Crea `.env` con `GOOGLE_API_KEY=<la tua chiave Google AI Studio>`

---

## Info Progetto

**Team**: LUISS Guido Carli  
**Committente**: Whitehall Reply per MEF (NoiPA)  
**Tipo**: Sistema multi-agente LangGraph per data quality PA italiana  
**Status**: ✅ 100% Completo e Pronto

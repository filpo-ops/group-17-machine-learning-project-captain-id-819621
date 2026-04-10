# Discorso Mid-Term Pitch: Agents for Data Quality (NoiPA - Reply)

**Durata stimata:** 3-4 minuti

---

## 1. Introduzione e Contesto (1 minuto)

"Buongiorno a tutti. Il nostro team sta lavorando al progetto proposto da Reply: **Agents for Data Quality**, focalizzato sul caso d'uso del portale **NoiPA**. 

Come sappiamo, NoiPA gestisce costantemente dataset molto complessi provenienti da fonti eterogenee. Attualmente, il controllo di qualità e la validazione su questi dati, che contengono ad esempio informazioni demografiche o sulle spese, risultano prettamente manuali o addirittura inesistenti. 

Il nostro obiettivo finale è sviluppare un **Multi-Agent System** basato su LLM in grado di ricevere un dataset grezzo (CSV), automatizzare una catena di validazione e produrre un vero e proprio 'Quality Report'. Il report finale non solo evidenzierà le anomalie, ma offrirà dei suggerimenti di correzione generati dagli agenti e un punteggio di affidabilità sul dato."

## 2. Stato dell'Arte: Il Lavoro Svolto Finora (1.5 minuti)

"Ad oggi, in vista di questo *mid-check*, abbiamo gettato le fondamenta quantitative e architetturali del sistema.

Sappiamo bene che prima di dare libertà di ragionamento a un Modello Generativo – esponendolo al rischio di allucinazioni – c'è bisogno di fornirgli dei *Tools* deterministici rigorosi su cui basare le sue conclusioni. Pertanto, ci siamo concentrati sull'esplorazione dei dataset forniti e abbiamo costruito la nostra **Deterministic Tool Library**. 

Seguendo rigorosamente i requisiti del progetto di Reply, abbiamo codificato in Python una serie robusta di tool che coprono i controlli primari:
1. **Schema Validation**: per l'audit automatico della conformità dei nomi.
2. **Completeness Analysis**: un check avanzato dei valori mancanti che individua non solo i classici NaN, ma riconosce falsi negativi e token ambigui come '?', '-', e 'N.D.'.
3. **Consistency e Outlier Detection**: validazione cross-colonna che riconosce discrepanze di formati logici o limiti numerici impossibili.

Parallelamente, abbiamo già completato il **setup dell'infrastruttura**. Abbiamo predisposto l'environment in modo da far girare modelli ottimizzati in locale tramite Ollama, ponendo così le basi per l'architettura LLM."

## 3. Prossimi Passi: Architettura Agenti e Frontend (1 minuto)

"Per quanto riguarda la seconda metà del progetto, dedicheremo il grosso dello sviluppo al cuore del sistema: l'**Architettura Multi-Agente**. 

Nelle prossime settimane ci focalizzeremo su:
1. **Integrazione degli Agenti LLM**: Utilizzeremo frame per l'orchestrazione per trasformare le funzioni scritte finora nei vettori sensoriali e attuativi degli agenti. Struttureremo ad esempio una Supervisor Architecture, in cui uno o più agenti osserveranno i risultati estratti dai Tool e genereranno in linguaggio naturale le **Correction Suggestions** in caso di anomalie.
2. **Synthetic Benchmark**: Svilupperemo un dataset generato sinteticamente con un certo livello di 'sporcizia' controllata per testare oggettivamente il Recall e la validità del nostro Reliability Score finale.
3. **Sviluppo dell'Interfaccia Grafica**: Come suggerito dalle linee guida aziendali, raccoglieremo l'output degli agenti costruendo una piccola web-app interattiva in **Streamlit**. Questa dashboard dimostrativa sarà l'output definitivo del progetto e permetterà agli utenti (o per noi, in fase di valutazione) di caricare un CSV e visionare immediatamente il report qualitativo.

Grazie mille dell'attenzione. Siamo a disposizione se avete domande oppure curiosità tecnico-logiche sulle implementazioni attuali."

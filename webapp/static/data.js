// Mock data and pipeline definitions

const PIPELINE_NODES = [
  { id: 'ingest',       kind: 'det', label: 'Ingest',         desc: 'Load CSV, infer types' },
  { id: 'discover',     kind: 'det', label: 'Discover',       desc: 'Rule discovery from samples' },
  { id: 'audit',        kind: 'det', label: 'Audit',          desc: '9 quality tools sweep' },
  { id: 'schema',       kind: 'llm', label: 'Schema',         desc: 'Plan validity fixes' },
  { id: 'completeness', kind: 'llm', label: 'Completeness',   desc: 'Plan imputation strategy' },
  { id: 'consistency',  kind: 'llm', label: 'Consistency',    desc: 'Plan normalization' },
  { id: 'anomaly',      kind: 'llm', label: 'Anomaly',        desc: 'Reason about outliers' },
  { id: 'remediation',  kind: 'det', label: 'Remediation',    desc: 'Apply planned actions' },
  { id: 're_audit',     kind: 'det', label: 'Re-audit',       desc: 'Re-run audit on fixed data' },
  { id: 'supervisor',   kind: 'det', label: 'Supervisor',     desc: 'Compute before/after score' },
];

const NODE_OUTCOMES = {
  ingest:       'Loaded sample.csv (7,543 × 18) — 9 numeric, 6 string, 3 date',
  discover:     '14 candidate rules discovered — 11 confirmed, 3 rejected',
  audit:        '29 issues found across 4 categories (1 critical, 16 high)',
  schema:       '5 actions planned · validity score 0.90',
  completeness: '7 actions planned · completeness score 0.00 → 0.74',
  consistency:  '4 actions planned · consistency score 0.78 → 0.90',
  anomaly:      '3 actions planned · 12 outliers flagged, 9 retained',
  remediation:  '19 actions applied · 1,847 cells modified',
  supervisor:   'Reliability = 54.0 / 100 — verdict: medium',
};

const SAMPLE_CSV_PREVIEW = {
  filename: 'sample.csv',
  size: '2.1 MB',
  rows: 7543,
  cols: 18,
  types: { numeric: 9, string: 6, date: 3 },
  columns: ['id', 'codice_fiscale', 'nome', 'cognome', 'data_nascita', 'comune', 'provincia', 'spesa', 'data_pagamento', 'categoria', 'qualifica', 'importo_lordo', 'importo_netto', 'flag_attivo'],
  preview: [
    { id: '00001', codice_fiscale: 'RSSMRA80A01H501Z', nome: 'Mario',    cognome: 'Rossi',    data_nascita: '1980-01-01', comune: 'Roma',     provincia: 'RM', spesa: '1245.50',  data_pagamento: '2024-03-15', categoria: 'A1', qualifica: 'Funzionario',  importo_lordo: '38500.00', importo_netto: '24300.00', flag_attivo: 'Y' },
    { id: '00002', codice_fiscale: 'BNCLCU75M52F205X', nome: 'Lucia',    cognome: 'Bianchi',  data_nascita: '1975-08-12', comune: 'Milano',   provincia: 'MI', spesa: '',         data_pagamento: '2024-03-15', categoria: 'A2', qualifica: 'Dirigente',    importo_lordo: '52000.00', importo_netto: '32100.00', flag_attivo: 'Y' },
    { id: '00003', codice_fiscale: 'VRDGNN82L15G273M', nome: 'Giovanni', cognome: 'Verdi',    data_nascita: '1982-07-15', comune: 'Napoli',   provincia: 'NA', spesa: '892.00',   data_pagamento: '15/03/2024', categoria: 'B1', qualifica: 'Funzionario',  importo_lordo: '36200.00', importo_netto: '23000.00', flag_attivo: 'Y' },
    { id: '00004', codice_fiscale: 'NREANN90T48L219K', nome: 'Anna',     cognome: 'Neri',     data_nascita: '1990-12-08', comune: 'Torino',   provincia: 'TO', spesa: '2180.75',  data_pagamento: '2024-03-16', categoria: 'A1', qualifica: 'Tecnico',      importo_lordo: '34000.00', importo_netto: '21800.00', flag_attivo: 'Y' },
    { id: '00005', codice_fiscale: 'GLLPLA68B10D612R', nome: 'Paola',    cognome: 'Galli',    data_nascita: '1968-02-10', comune: 'Bologna',  provincia: 'BO', spesa: '—',        data_pagamento: '2024-03-16', categoria: 'C1', qualifica: 'Dirigente',    importo_lordo: '58400.00', importo_netto: '35600.00', flag_attivo: 'Y' },
    { id: '00006', codice_fiscale: 'CNTFNC85R22F839S', nome: 'Francesco',cognome: 'Conti',    data_nascita: '1985-10-22', comune: 'firenze',  provincia: 'FI', spesa: '1567.30',  data_pagamento: '2024-03-17', categoria: 'A2', qualifica: 'Funzionario',  importo_lordo: '41200.00', importo_netto: '26100.00', flag_attivo: 'Y' },
    { id: '00007', codice_fiscale: 'MNRGLI78P55I452Q', nome: 'Giulia',   cognome: 'Marino',   data_nascita: '1978-09-15', comune: 'Bari',     provincia: 'BA', spesa: '34520.00', data_pagamento: '2024-03-17', categoria: 'B1', qualifica: 'Tecnico',      importo_lordo: '32800.00', importo_netto: '21000.00', flag_attivo: 'Y' },
    { id: '00007', codice_fiscale: 'MNRGLI78P55I452Q', nome: 'Giulia',   cognome: 'Marino',   data_nascita: '1978-09-15', comune: 'Bari',     provincia: 'BA', spesa: '34520.00', data_pagamento: '2024-03-17', categoria: 'B1', qualifica: 'Tecnico',      importo_lordo: '32800.00', importo_netto: '21000.00', flag_attivo: 'Y' },
    { id: '00008', codice_fiscale: 'FRRSFN72E18H501W', nome: 'Stefano',  cognome: 'Ferrari',  data_nascita: '1972-05-18', comune: 'Genova',   provincia: 'GE', spesa: '1112.40',  data_pagamento: '2024-03-18', categoria: 'A1', qualifica: 'Funzionario',  importo_lordo: '40100.00', importo_netto: '25400.00', flag_attivo: 'N' },
  ]
};

const FIXED_PREVIEW = [
  { id: '00001', codice_fiscale: 'RSSMRA80A01H501Z', nome: 'Mario',    cognome: 'Rossi',    data_nascita: '1980-01-01', comune: 'Roma',     provincia: 'RM', spesa: '1245.50',  data_pagamento: '2024-03-15', categoria: 'A1', qualifica: 'Funzionario',  importo_lordo: '38500.00', importo_netto: '24300.00', flag_attivo: 'Y' },
  { id: '00002', codice_fiscale: 'BNCLCU75M52F205X', nome: 'Lucia',    cognome: 'Bianchi',  data_nascita: '1975-08-12', comune: 'Milano',   provincia: 'MI', spesa: '1356.20*', data_pagamento: '2024-03-15', categoria: 'A2', qualifica: 'Dirigente',    importo_lordo: '52000.00', importo_netto: '32100.00', flag_attivo: 'Y' },
  { id: '00003', codice_fiscale: 'VRDGNN82L15G273M', nome: 'Giovanni', cognome: 'Verdi',    data_nascita: '1982-07-15', comune: 'Napoli',   provincia: 'NA', spesa: '892.00',   data_pagamento: '2024-03-15*',categoria: 'B1', qualifica: 'Funzionario',  importo_lordo: '36200.00', importo_netto: '23000.00', flag_attivo: 'Y' },
  { id: '00004', codice_fiscale: 'NREANN90T48L219K', nome: 'Anna',     cognome: 'Neri',     data_nascita: '1990-12-08', comune: 'Torino',   provincia: 'TO', spesa: '2180.75',  data_pagamento: '2024-03-16', categoria: 'A1', qualifica: 'Tecnico',      importo_lordo: '34000.00', importo_netto: '21800.00', flag_attivo: 'Y' },
  { id: '00005', codice_fiscale: 'GLLPLA68B10D612R', nome: 'Paola',    cognome: 'Galli',    data_nascita: '1968-02-10', comune: 'Bologna',  provincia: 'BO', spesa: '1356.20*', data_pagamento: '2024-03-16', categoria: 'C1', qualifica: 'Dirigente',    importo_lordo: '58400.00', importo_netto: '35600.00', flag_attivo: 'Y' },
  { id: '00006', codice_fiscale: 'CNTFNC85R22F839S', nome: 'Francesco',cognome: 'Conti',    data_nascita: '1985-10-22', comune: 'Firenze*', provincia: 'FI', spesa: '1567.30',  data_pagamento: '2024-03-17', categoria: 'A2', qualifica: 'Funzionario',  importo_lordo: '41200.00', importo_netto: '26100.00', flag_attivo: 'Y' },
  { id: '00007', codice_fiscale: 'MNRGLI78P55I452Q', nome: 'Giulia',   cognome: 'Marino',   data_nascita: '1978-09-15', comune: 'Bari',     provincia: 'BA', spesa: '34520.00', data_pagamento: '2024-03-17', categoria: 'B1', qualifica: 'Tecnico',      importo_lordo: '32800.00', importo_netto: '21000.00', flag_attivo: 'Y' },
  { id: '00008', codice_fiscale: 'FRRSFN72E18H501W', nome: 'Stefano',  cognome: 'Ferrari',  data_nascita: '1972-05-18', comune: 'Genova',   provincia: 'GE', spesa: '1112.40',  data_pagamento: '2024-03-18', categoria: 'A1', qualifica: 'Funzionario',  importo_lordo: '40100.00', importo_netto: '25400.00', flag_attivo: 'N' },
];

const RESULTS = {
  reliability_score: 54.0,
  sub_scores: { validity: 90, completeness: 0, consistency: 90, uniqueness: 90, accuracy: 0 },
  weights:    { validity: 0.20, completeness: 0.30, consistency: 0.25, uniqueness: 0.15, accuracy: 0.10 },
  severity_breakdown: { critical: 1, high: 16, medium: 6, low: 6 },
  issues: [
    { id: 1,  tool: 'check_nulls',          issue_type: 'missing_mandatory_values',  severity: 'critical', columns: ['spesa'],                  row_count: 1582, message: 'Column `spesa` has 1582 effectively missing values (21.0%)', sample_rows: [2, 5, 47, 102, 318] },
    { id: 2,  tool: 'check_outliers',       issue_type: 'extreme_outlier',           severity: 'high',     columns: ['spesa'],                  row_count: 12,   message: 'Values >25× IQR detected in `spesa` (max 34,520.00)', sample_rows: [7, 891, 2104] },
    { id: 3,  tool: 'check_dates',          issue_type: 'inconsistent_format',       severity: 'high',     columns: ['data_pagamento'],         row_count: 84,   message: 'Mixed date formats: ISO and DD/MM/YYYY in `data_pagamento`', sample_rows: [3, 412] },
    { id: 4,  tool: 'check_duplicates',     issue_type: 'duplicate_keys',            severity: 'high',     columns: ['id'],                     row_count: 47,   message: '47 duplicate `id` values detected', sample_rows: [7, 8] },
    { id: 5,  tool: 'check_casing',         issue_type: 'inconsistent_casing',       severity: 'medium',   columns: ['comune'],                 row_count: 213,  message: 'Mixed casing in `comune` (e.g. "firenze" vs "Firenze")', sample_rows: [6] },
    { id: 6,  tool: 'check_referential',    issue_type: 'orphan_value',              severity: 'high',     columns: ['categoria'],              row_count: 8,    message: '8 rows reference unknown `categoria` codes', sample_rows: [4521] },
    { id: 7,  tool: 'check_codice_fiscale', issue_type: 'invalid_checksum',          severity: 'high',     columns: ['codice_fiscale'],         row_count: 23,   message: '23 codice fiscale entries fail checksum validation', sample_rows: [88, 412] },
    { id: 8,  tool: 'check_ranges',         issue_type: 'out_of_range',              severity: 'medium',   columns: ['importo_netto'],          row_count: 4,    message: '`importo_netto` exceeds `importo_lordo` in 4 rows', sample_rows: [1203] },
    { id: 9,  tool: 'check_dates',          issue_type: 'future_date',               severity: 'low',      columns: ['data_pagamento'],         row_count: 2,    message: '2 payment dates are in the future', sample_rows: [3344] },
    { id: 10, tool: 'check_enum',           issue_type: 'invalid_enum_value',        severity: 'medium',   columns: ['flag_attivo'],            row_count: 5,    message: '`flag_attivo` contains values outside {Y, N}', sample_rows: [921] },
  ],
  correction_log: [
    { id: 1, agent: 'Completeness', action: 'impute_median',     column: 'spesa',           rows_affected: 1582, rationale: '21% missing on critical column; median preserves distribution shape' },
    { id: 2, agent: 'Anomaly',      action: 'cap_p99',           column: 'spesa',           rows_affected: 12,   rationale: 'Winsorize at p99 to limit extreme values without dropping rows' },
    { id: 3, agent: 'Consistency',  action: 'normalize_dates',   column: 'data_pagamento',  rows_affected: 84,   rationale: 'Coerce DD/MM/YYYY into ISO 8601 to match dominant format' },
    { id: 4, agent: 'Schema',       action: 'drop_duplicates',   column: 'id',              rows_affected: 47,   rationale: 'Primary-key duplicates resolved by keeping first occurrence' },
    { id: 5, agent: 'Consistency',  action: 'titlecase',         column: 'comune',          rows_affected: 213,  rationale: 'Apply Italian title-case to harmonize comune values' },
    { id: 6, agent: 'Schema',       action: 'flag_orphans',      column: 'categoria',       rows_affected: 8,    rationale: 'Mark for review — outside known taxonomy, do not silently drop' },
    { id: 7, agent: 'Schema',       action: 'flag_invalid',      column: 'codice_fiscale',  rows_affected: 23,   rationale: 'Checksum failures preserved with quarantine flag for human review' },
    { id: 8, agent: 'Anomaly',      action: 'ignore',            column: 'importo_netto',   rows_affected: 0,    rationale: 'Likely legitimate (bonuses) — defer to domain expert, do not auto-correct' },
    { id: 9, agent: 'Schema',       action: 'null_future_dates', column: 'data_pagamento',  rows_affected: 2,    rationale: 'Future payment dates set to null; downstream pipeline handles missing' },
    { id: 10,agent: 'Schema',       action: 'coerce_enum',       column: 'flag_attivo',     rows_affected: 5,    rationale: 'Map "1"/"0"/"true" to Y/N; preserves business semantics' },
  ],
  audit_trail: [
    'ingest:        loaded sample.csv (7,543 × 18) in 0.21s',
    'ingest:        inferred types — 9 numeric, 6 string, 3 date',
    'discover:      14 rules generated, 11 retained after sample validation',
    'audit:         9 tools executed in parallel — 29 issues raised',
    'audit:         severity distribution — 1 critical, 16 high, 6 medium, 6 low',
    'schema:        agent invoked (deepseek-chat) — 5 actions planned',
    'completeness:  agent invoked — 7 actions planned, score 0.00 → 0.74',
    'consistency:   agent invoked — 4 actions planned, score 0.78 → 0.90',
    'anomaly:       agent invoked — 3 actions planned, 12 outliers reviewed',
    'remediation:   19 actions applied — 1,847 cells modified, 47 rows dropped',
    'supervisor:    weighted score = 0.20·90 + 0.30·0 + 0.25·90 + 0.15·90 + 0.10·0',
    'supervisor:    reliability = 54.0 / 100 — verdict: medium',
  ],
};

window.PIPELINE_NODES = PIPELINE_NODES;
window.NODE_OUTCOMES = NODE_OUTCOMES;
window.SAMPLE_CSV_PREVIEW = SAMPLE_CSV_PREVIEW;
window.FIXED_PREVIEW = FIXED_PREVIEW;
window.RESULTS = RESULTS;

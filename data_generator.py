"""
data_generator.py
-----------------
Generates two realistic CSV files with intentional data-quality problems:
  - data/raw/spesa.csv              (7 543 rows × 18 columns)
  - data/raw/attivazioniCessazioni.csv  (20 102 rows × 19 columns)

Run from the project root:
    python data_generator.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

os.makedirs("data/raw", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ITALIAN_REGIONS = [
    "Lombardia", "Lazio", "Campania", "Sicilia", "Veneto",
    "Emilia-Romagna", "Piemonte", "Puglia", "Toscana", "Calabria",
    "Sardegna", "Liguria", "Marche", "Abruzzo", "Friuli-Venezia Giulia",
    "Trentino-Alto Adige", "Umbria", "Basilicata", "Molise", "Valle d'Aosta",
]

PROVINCE_CODES = [
    "RM", "MI", "NA", "TO", "PA", "GE", "BO", "FI", "BA", "CT",
    "VE", "AQ", "PT", "MO", "PD", "BS", "BG", "VA", "TS", "BZ",
    "CA", "AN", "PE", "PG", "PZ", "CB", "AO",
]


def _random_iso_timestamps(n: int) -> np.ndarray:
    """Return n ISO-like timestamp strings."""
    base = pd.Timestamp("2024-01-01")
    offsets = np.random.randint(0, 365 * 24 * 3600, size=n)
    return np.array(
        [(base + pd.Timedelta(seconds=int(s))).strftime("%Y-%m-%dT%H:%M:%S.") +
         f"{np.random.randint(0, 1000):03d}"
         for s in offsets]
    )


def _mixed_aggregation_time(n: int) -> np.ndarray:
    """
    85% ISO, 5% slash, 5% dot, 5% short-dash  (same breakdown as spesa spec).
    """
    iso = _random_iso_timestamps(n)

    years  = np.random.choice([2023, 2024], size=n)
    months = np.random.randint(1, 13, size=n)
    days   = np.random.randint(1, 29, size=n)

    slash_fmt = np.array([f"{y}/{m:02d}/{d:02d}" for y, m, d in zip(years, months, days)])
    dot_fmt   = np.array([f"{d:02d}.{m:02d}.{y}"  for y, m, d in zip(years, months, days)])
    short_fmt = np.array([f"{d:02d}-{m:02d}-{str(y)[2:]}" for y, m, d in zip(years, months, days)])

    result = iso.copy()
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_slash = int(n * 0.05)
    n_dot   = int(n * 0.05)
    n_short = int(n * 0.05)

    result[idx[:n_slash]]                    = slash_fmt[idx[:n_slash]]
    result[idx[n_slash:n_slash+n_dot]]       = dot_fmt  [idx[n_slash:n_slash+n_dot]]
    result[idx[n_slash+n_dot:n_slash+n_dot+n_short]] = short_fmt[idx[n_slash+n_dot:n_slash+n_dot+n_short]]

    return result


# ===========================================================================
# 1.  spesa.csv
# ===========================================================================
print("Generating spesa.csv ...")

N_SPESA = 7543

# ------------------------------------------------------------------
# aggregation-time
# ------------------------------------------------------------------
agg_time_spesa = _mixed_aggregation_time(N_SPESA)

# ------------------------------------------------------------------
# tipo_imposta  (10 noisy variants)
# ------------------------------------------------------------------
TIPO_IMP_VARIANTS = [
    "Erariali", "Erariali ", "erariali", "ERARIALI",
    "Previdenziali", "Netto", "Assistenziali",
    "Da definire", "Mista", "PREVIDENZIALI",
]
tipo_imposta = np.random.choice(TIPO_IMP_VARIANTS, size=N_SPESA,
                                p=[0.25, 0.05, 0.05, 0.05,
                                   0.20, 0.10, 0.10,
                                   0.07, 0.08, 0.05])

# ------------------------------------------------------------------
# Tipo Imposta  (4 clean values, 385 intentional mismatches)
# ------------------------------------------------------------------
TIPO_IMP_CLEAN = ["Erariali", "Previdenziali", "Netto", "Assistenziali"]

# Derive a "correct" mapping from tipo_imposta
def _canonical(v):
    v = v.strip().lower()
    if "erariali"      in v: return "Erariali"
    if "previdenziali" in v: return "Previdenziali"
    if "netto"         in v: return "Netto"
    if "assistenziali" in v: return "Assistenziali"
    return np.random.choice(TIPO_IMP_CLEAN)   # Da definire / Mista -> random

tipo_imposta_clean = np.array([_canonical(v) for v in tipo_imposta])

# Inject 385 mismatches
mismatch_idx = np.random.choice(N_SPESA, size=385, replace=False)
for i in mismatch_idx:
    current = tipo_imposta_clean[i]
    choices = [x for x in TIPO_IMP_CLEAN if x != current]
    tipo_imposta_clean[i] = np.random.choice(choices)

# ------------------------------------------------------------------
# spesa  (float with dirty rows)
# ------------------------------------------------------------------
# Base numeric values
spesa_values = np.random.uniform(1000, 5_000_000, size=N_SPESA)

# 35 values > 1 billion
big_idx = np.random.choice(N_SPESA, size=35, replace=False)
spesa_values[big_idx] = np.random.uniform(1_000_000_001, 9_000_000_000, size=35)

# 11 negative values
neg_idx = np.random.choice(
    [i for i in range(N_SPESA) if i not in big_idx], size=11, replace=False
)
spesa_values[neg_idx] = np.random.uniform(-999_999.5, -0.01, size=11)

# Sentinel at row 1528
spesa_values[1528] = 999_999_999.99

# Build string column
spesa_col = spesa_values.astype(str).tolist()

# 15 rows: "€{value}" format
euro_prefix_idx = np.random.choice(
    [i for i in range(N_SPESA) if i != 1528], size=15, replace=False
)
for i in euro_prefix_idx:
    spesa_col[i] = f"€{spesa_values[i]:.2f}"

# 15 rows: "{value} EUR" format
eur_suffix_idx = np.random.choice(
    [i for i in range(N_SPESA) if i != 1528 and i not in euro_prefix_idx],
    size=15, replace=False,
)
for i in eur_suffix_idx:
    spesa_col[i] = f"{spesa_values[i]:.2f} EUR"

# 197 rows: Italian comma style
it_comma_pool = [
    i for i in range(N_SPESA)
    if i != 1528 and i not in euro_prefix_idx and i not in eur_suffix_idx
]
it_comma_idx = np.random.choice(it_comma_pool, size=197, replace=False)
for i in it_comma_idx:
    formatted = f"{spesa_values[i]:.12f}"          # many decimals
    spesa_col[i] = formatted.replace(".", ",")     # Italian comma

# ------------------------------------------------------------------
# SPESA TOTALE  (same as spesa, 39 rows differ; row 1528 = 104741.55)
# ------------------------------------------------------------------
spesa_totale_values = spesa_values.copy()
spesa_totale_values[1528] = 104_741.55

diff_idx = np.random.choice(
    [i for i in range(N_SPESA) if i != 1528], size=38, replace=False
)
spesa_totale_values[diff_idx] = np.random.uniform(1000, 5_000_000, size=38)

# ------------------------------------------------------------------
# cod_imposta + 2cod_imposta + cod imposta ext
# with ~225 mismatches between cod_imposta and 2cod_imposta
# ------------------------------------------------------------------
cod_imposta = np.random.randint(1, 21, size=N_SPESA)

cod2_imposta = cod_imposta.copy()
mismatch225 = np.random.choice(N_SPESA, size=225, replace=False)
for i in mismatch225:
    current = cod2_imposta[i]
    choices = [c for c in range(1, 21) if c != current]
    cod2_imposta[i] = np.random.choice(choices)

cod_ext = cod2_imposta.copy()
mismatch_ext = np.random.choice(N_SPESA, size=180, replace=False)
for i in mismatch_ext:
    current = cod_ext[i]
    choices = [c for c in range(1, 21) if c != current]
    cod_ext[i] = np.random.choice(choices)

# ------------------------------------------------------------------
# ente + ente%code  (302 rows disagree)
# ------------------------------------------------------------------
ENTE_NAMES = [f"Ente_{i:03d}" for i in range(1, 101)]
ente_array = np.random.choice(ENTE_NAMES, size=N_SPESA)

# Consistent mapping ente -> code
ente_to_code = {name: idx + 1 for idx, name in enumerate(ENTE_NAMES)}
ente_code_array = np.array([ente_to_code[e] for e in ente_array], dtype=float)

# Inject 302 mismatches
mismatch302 = np.random.choice(N_SPESA, size=302, replace=False)
for i in mismatch302:
    current_code = ente_code_array[i]
    alt_code = float(np.random.choice([c for c in range(1, 101) if c != current_code]))
    ente_code_array[i] = alt_code

# ------------------------------------------------------------------
# cod_tipoimposta  (1-5, each code maps to 5-7 tipo_imposta values)
# ------------------------------------------------------------------
cod_tipoimposta = np.random.randint(1, 6, size=N_SPESA)

# ------------------------------------------------------------------
# rata  (period codes with format mix)
# ------------------------------------------------------------------
RATA_YYYYMM = [f"{y}{m:02d}" for y in range(2023, 2025) for m in range(1, 13)]
rata_base = np.random.choice(RATA_YYYYMM, size=N_SPESA)

rata_col = rata_base.tolist()

# 90 rows: "MM/YYYY" format
rata_slash_idx = np.random.choice(N_SPESA, size=90, replace=False)
for i in rata_slash_idx:
    ym = rata_base[i]           # "YYYYMM"
    y, m = ym[:4], ym[4:]
    rata_col[i] = f"{m}/{y}"

# 80 rows: "MON-YYYY" format
IT_MONTHS_SHORT = ["GEN","FEB","MAR","APR","MAG","GIU",
                   "LUG","AGO","SET","OTT","NOV","DIC"]
rata_mon_pool = [i for i in range(N_SPESA) if i not in rata_slash_idx]
rata_mon_idx  = np.random.choice(rata_mon_pool, size=80, replace=False)
for i in rata_mon_idx:
    ym = rata_base[i]
    y  = ym[:4]
    m  = int(ym[4:]) - 1
    rata_col[i] = f"{IT_MONTHS_SHORT[m]}-{y}"

# 6 rows: "Rata YYYY" format
rata_text_pool = [i for i in range(N_SPESA)
                  if i not in rata_slash_idx and i not in rata_mon_idx]
rata_text_idx  = np.random.choice(rata_text_pool, size=6, replace=False)
RATA_TEXT_VALS = ["Rata 2024", "Rata 2024", "Rata 2024",
                  "Rata 2023", "Rata 2023", "Rata 2023"]
for pos, i in enumerate(rata_text_idx):
    rata_col[i] = RATA_TEXT_VALS[pos]

# ------------------------------------------------------------------
# note  (98% null → 7393 nulls, 4 non-null values)
# ------------------------------------------------------------------
NOTE_VALUES = ["Da riconciliare", "Verifica manuale", "Fonte secondaria", "Dato stimato"]
note_col = np.full(N_SPESA, np.nan, dtype=object)
note_non_null_idx = np.random.choice(N_SPESA, size=150, replace=False)
for pos, i in enumerate(note_non_null_idx):
    note_col[i] = NOTE_VALUES[pos % len(NOTE_VALUES)]

# ------------------------------------------------------------------
# fonte_dato  (99% null → 7468 nulls)
# ------------------------------------------------------------------
fonte_dato_col = np.full(N_SPESA, np.nan, dtype=object)
fonte_non_null_idx = np.random.choice(N_SPESA, size=75, replace=False)
fonte_dato_col[fonte_non_null_idx] = "Sistema legacy"

# ------------------------------------------------------------------
# area_geografica  (21% null → 1582 nulls)
# ------------------------------------------------------------------
AREA_VALS = ["Nord", "Sud", "Centro", "Isole"]
area_col = np.random.choice(AREA_VALS, size=N_SPESA)
null_area_idx = np.random.choice(N_SPESA, size=1582, replace=False)
area_col = area_col.astype(object)
area_col[null_area_idx] = np.nan

# ------------------------------------------------------------------
# anno + mese  (from rata_base)
# ------------------------------------------------------------------
anno_col = np.array([int(r[:4]) for r in rata_base])
mese_col = np.array([int(r[4:]) for r in rata_base])

# ------------------------------------------------------------------
# regione
# ------------------------------------------------------------------
regione_col = np.random.choice(ITALIAN_REGIONS, size=N_SPESA)

# ------------------------------------------------------------------
# Assemble spesa DataFrame
# ------------------------------------------------------------------
df_spesa = pd.DataFrame({
    "aggregation-time": agg_time_spesa,
    "tipo_imposta":     tipo_imposta,
    "Tipo Imposta":     tipo_imposta_clean,
    "spesa":            spesa_col,
    "SPESA TOTALE":     spesa_totale_values,
    "cod_imposta":      cod_imposta,
    "2cod_imposta":     cod2_imposta,
    "cod imposta ext":  cod_ext,
    "ente":             ente_array,
    "ente%code":        ente_code_array.astype(int),
    "cod_tipoimposta":  cod_tipoimposta,
    "rata":             rata_col,
    "note":             note_col,
    "fonte_dato":       fonte_dato_col,
    "area_geografica":  area_col,
    "anno":             anno_col,
    "mese":             mese_col,
    "regione":          regione_col,
})

assert len(df_spesa) == N_SPESA, f"Expected {N_SPESA} rows, got {len(df_spesa)}"
assert len(df_spesa.columns) == 18, f"Expected 18 columns, got {len(df_spesa.columns)}"

df_spesa.to_csv("data/raw/spesa.csv", index=False)
print(f"  spesa.csv saved: {len(df_spesa)} rows × {len(df_spesa.columns)} columns")

# Sanity checks (printed, not enforced)
mismatches_tipo = (df_spesa["tipo_imposta"].str.strip().str.lower() !=
                   df_spesa["Tipo Imposta"].str.lower()).sum()
mismatches_cod  = (df_spesa["cod_imposta"] != df_spesa["2cod_imposta"]).sum()
mismatches_ente = (df_spesa["ente%code"] != df_spesa["ente"].map(ente_to_code)).sum()
print(f"  tipo_imposta vs Tipo Imposta mismatches: {mismatches_tipo}")
print(f"  cod_imposta  vs 2cod_imposta  mismatches: {mismatches_cod}")
print(f"  ente         vs ente%%code     mismatches: {mismatches_ente}")
print(f"  spesa sentinel row 1528: {df_spesa.loc[1528, 'spesa']}")
print(f"  SPESA TOTALE row 1528:   {df_spesa.loc[1528, 'SPESA TOTALE']}")


# ===========================================================================
# 2.  attivazioniCessazioni.csv
# ===========================================================================
print("\nGenerating attivazioniCessazioni.csv ...")

N_ATT = 20102

# ------------------------------------------------------------------
# RATA  (mostly YYYYMM, 72 rows in other formats)
# ------------------------------------------------------------------
RATA_POOL_ATT = [f"{y}{m:02d}" for y in range(2023, 2025) for m in range(1, 13)]
rata_att_base = np.random.choice(RATA_POOL_ATT, size=N_ATT)
rata_att_col  = rata_att_base.tolist()

# 25 rows: "MON-YYYY" (Italian abbreviation)
it_mon_idx_att = np.random.choice(N_ATT, size=25, replace=False)
for i in it_mon_idx_att:
    ym = rata_att_base[i]
    y  = ym[:4]
    m  = int(ym[4:]) - 1
    rata_att_col[i] = f"{IT_MONTHS_SHORT[m]}-{y}"

# 25 rows: "YYYY-MM" format
dash_pool_att = [i for i in range(N_ATT) if i not in it_mon_idx_att]
dash_idx_att  = np.random.choice(dash_pool_att, size=25, replace=False)
for i in dash_idx_att:
    ym = rata_att_base[i]
    rata_att_col[i] = f"{ym[:4]}-{ym[4:]}"

# 22 rows: "MM/YYYY" format
slash_pool_att = [i for i in range(N_ATT)
                  if i not in it_mon_idx_att and i not in dash_idx_att]
slash_idx_att  = np.random.choice(slash_pool_att, size=22, replace=False)
for i in slash_idx_att:
    ym = rata_att_base[i]
    rata_att_col[i] = f"{ym[4:]}/{ym[:4]}"

# ------------------------------------------------------------------
# aggregation-time
# ------------------------------------------------------------------
agg_time_att = _mixed_aggregation_time(N_ATT)

# ------------------------------------------------------------------
# provincia_sede  (noisy) + Provincia Sede  (cleaner, 367 nulls)
# ------------------------------------------------------------------
prov_upper = np.random.choice(PROVINCE_CODES, size=N_ATT)

prov_noisy = prov_upper.copy().astype(object)

# 482 lowercase
lower_idx  = np.random.choice(N_ATT, size=482, replace=False)
prov_noisy[lower_idx] = np.array([p.lower() for p in prov_upper[lower_idx]])

# 726 mixed case
mixed_pool = [i for i in range(N_ATT) if i not in lower_idx]
mixed_idx  = np.random.choice(mixed_pool, size=726, replace=False)
def _mixed_case(s):
    return s[0].upper() + s[1:].lower()
prov_noisy[mixed_idx] = np.array([_mixed_case(p) for p in prov_upper[mixed_idx]])

# 20 placeholder values
placeholder_pool = [i for i in range(N_ATT)
                    if i not in lower_idx and i not in mixed_idx]
placeholder_idx  = np.random.choice(placeholder_pool, size=20, replace=False)
PLACEHOLDERS = ["?", "//", "-", " "]
for pos, i in enumerate(placeholder_idx):
    prov_noisy[i] = PLACEHOLDERS[pos % len(PLACEHOLDERS)]

# Provincia Sede: start from uppercase, add 367 nulls
prov_clean = prov_upper.copy().astype(object)
null_prov_idx = np.random.choice(N_ATT, size=367, replace=False)
prov_clean[null_prov_idx] = np.nan

# 1671 mismatches between provincia_sede and Provincia Sede
mismatch_prov = np.random.choice(
    [i for i in range(N_ATT) if i not in null_prov_idx], size=1671, replace=False
)
for i in mismatch_prov:
    current = prov_clean[i]
    choices = [p for p in PROVINCE_CODES if p != current]
    prov_clean[i] = np.random.choice(choices)

# ------------------------------------------------------------------
# descrizione_ente + 3descrizione  (1232 mismatches)
# ------------------------------------------------------------------
DESC_ENTE_VALS = [f"Ente_{i:03d}" for i in range(1, 101)]
desc_ente_col = np.random.choice(DESC_ENTE_VALS, size=N_ATT)

desc3_col = desc_ente_col.copy().astype(object)
mismatch_desc = np.random.choice(N_ATT, size=1232, replace=False)
for i in mismatch_desc:
    current = desc3_col[i]
    choices = [d for d in DESC_ENTE_VALS if d != current]
    desc3_col[i] = np.random.choice(choices)

# ------------------------------------------------------------------
# codice_ente (210 nulls) + CODICE ENTE  (603 mismatches)
# ------------------------------------------------------------------
codice_ente_col = np.random.randint(1, 101, size=N_ATT).astype(object)
null_ce_idx = np.random.choice(N_ATT, size=210, replace=False)
codice_ente_col[null_ce_idx] = np.nan

codice_ente_clean = codice_ente_col.copy()
mismatch_ce_pool = [i for i in range(N_ATT) if i not in null_ce_idx]
mismatch_ce_idx  = np.random.choice(mismatch_ce_pool, size=603, replace=False)
for i in mismatch_ce_idx:
    current = codice_ente_clean[i]
    choices = [c for c in range(1, 101) if c != current]
    codice_ente_clean[i] = np.random.choice(choices)

# ------------------------------------------------------------------
# regione_sede (308 nulls) + regione%sede (813 mismatches)
# ------------------------------------------------------------------
reg_sede_col = np.random.choice(ITALIAN_REGIONS, size=N_ATT).astype(object)
null_reg_idx = np.random.choice(N_ATT, size=308, replace=False)
reg_sede_col[null_reg_idx] = np.nan

reg_sede2_col = reg_sede_col.copy()
mismatch_reg_pool = [i for i in range(N_ATT) if i not in null_reg_idx]
mismatch_reg_idx  = np.random.choice(mismatch_reg_pool, size=813, replace=False)
for i in mismatch_reg_idx:
    current = reg_sede2_col[i]
    choices = [r for r in ITALIAN_REGIONS if r != current]
    reg_sede2_col[i] = np.random.choice(choices)

# ------------------------------------------------------------------
# attivazioni  (int with 602 non-numeric rows)
# ------------------------------------------------------------------
att_base = np.random.randint(0, 500, size=N_ATT)
att_col  = att_base.astype(object)

# 200 rows: Italian comma "6,0"
it_comma_att = np.random.choice(N_ATT, size=200, replace=False)
for i in it_comma_att:
    att_col[i] = f"{att_base[i]},0"

# 200 rows: suffix "0 unità"
suffix_pool_att = [i for i in range(N_ATT) if i not in it_comma_att]
suffix_idx_att  = np.random.choice(suffix_pool_att, size=200, replace=False)
for i in suffix_idx_att:
    att_col[i] = f"{att_base[i]} unità"

# 202 rows: "N.D."
nd_pool_att = [i for i in range(N_ATT)
               if i not in it_comma_att and i not in suffix_idx_att]
nd_idx_att  = np.random.choice(nd_pool_att, size=202, replace=False)
for i in nd_idx_att:
    att_col[i] = "N.D."

# ------------------------------------------------------------------
# att ivazioni  (space in name, 640 mismatches with attivazioni)
# ------------------------------------------------------------------
att2_col = att_base.copy().astype(object)
mismatch_att_pool = [i for i in range(N_ATT)
                     if i not in it_comma_att
                     and i not in suffix_idx_att
                     and i not in nd_idx_att]
mismatch_att_idx  = np.random.choice(mismatch_att_pool, size=640, replace=False)
for i in mismatch_att_idx:
    att2_col[i] = np.random.randint(0, 500)

# ------------------------------------------------------------------
# cessazioni  (int with 601 non-numeric rows, same pattern)
# ------------------------------------------------------------------
ces_base = np.random.randint(0, 500, size=N_ATT)
ces_col  = ces_base.astype(object)

it_comma_ces = np.random.choice(N_ATT, size=200, replace=False)
for i in it_comma_ces:
    ces_col[i] = f"{ces_base[i]},0"

suffix_pool_ces = [i for i in range(N_ATT) if i not in it_comma_ces]
suffix_idx_ces  = np.random.choice(suffix_pool_ces, size=200, replace=False)
for i in suffix_idx_ces:
    ces_col[i] = f"{ces_base[i]} unità"

nd_pool_ces = [i for i in range(N_ATT)
               if i not in it_comma_ces and i not in suffix_idx_ces]
nd_idx_ces  = np.random.choice(nd_pool_ces, size=201, replace=False)
for i in nd_idx_ces:
    ces_col[i] = "N.D."

# ------------------------------------------------------------------
# mese  (62 unique values)
# ------------------------------------------------------------------
IT_MONTHS_FULL = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
]

# Build pool of 62 unique month representations
mese_formats = (
    [f"{m:02d}" for m in range(1, 13)]           # "01"-"12"
    + [str(m) for m in range(1, 10)]              # "1"-"9"
    + IT_MONTHS_SHORT                              # "GEN"-"DIC"
    + IT_MONTHS_FULL                               # "Gennaio"-"Dicembre"
    + [f"mese {m}" for m in range(1, 13)]          # "mese 1"-"mese 12"
    + ["-1", "0", "13", "99"]                      # out-of-range
)
# mese_formats now has 12+9+12+12+12+4 = 61 items; add one more to reach 62
mese_formats = mese_formats + ["mese 01"]
assert len(set(mese_formats)) == 62, f"Expected 62 unique, got {len(set(mese_formats))}"

mese_att_col = np.random.choice(mese_formats, size=N_ATT)

# ------------------------------------------------------------------
# anno  (10 unique values)
# ------------------------------------------------------------------
anno_formats = ["2021", "2022", "2023", "2024",
                "2023.", "2024.",
                "23", "24",
                "anno 2023", "anno 2024"]
assert len(set(anno_formats)) == 10
anno_att_col = np.random.choice(anno_formats, size=N_ATT)

# ------------------------------------------------------------------
# Now ensure 2442 rows where mese+anno is inconsistent with RATA
# We define "consistent" as: RATA encodes YYYYMM that matches anno+mese
# We'll mark 2442 rows as intentionally inconsistent.
# ------------------------------------------------------------------
# First derive the "expected" numeric mese/anno from rata_att_base
rata_mese_num = np.array([int(r[4:]) for r in rata_att_base])   # 1-12
rata_anno_num = np.array([int(r[:4]) for r in rata_att_base])   # 2023/2024

# Check consistency for all rows
def _parse_mese(s):
    """Return numeric month 1-12, or None if unparseable / out of range."""
    s = str(s).strip()
    if s.isdigit():
        v = int(s)
        return v if 1 <= v <= 12 else None
    if s.startswith("mese "):
        try:
            v = int(s[5:].strip())
            return v if 1 <= v <= 12 else None
        except ValueError:
            return None
    if s in IT_MONTHS_SHORT:
        return IT_MONTHS_SHORT.index(s) + 1
    if s in IT_MONTHS_FULL:
        return IT_MONTHS_FULL.index(s) + 1
    return None

def _parse_anno(s):
    """Return numeric year, or None."""
    s = str(s).strip().rstrip(".")
    if s.isdigit() and len(s) == 4:
        return int(s)
    if s.isdigit() and len(s) == 2:
        return 2000 + int(s)
    if s.startswith("anno "):
        try:
            return int(s[5:].strip())
        except ValueError:
            return None
    return None

parsed_mese = np.array([_parse_mese(m) for m in mese_att_col], dtype=object)
parsed_anno = np.array([_parse_anno(a) for a in anno_att_col], dtype=object)

# Rows already inconsistent (None or mismatch)
already_incon = np.array([
    (parsed_mese[i] is None or parsed_anno[i] is None or
     parsed_mese[i] != rata_mese_num[i] or
     parsed_anno[i] != rata_anno_num[i])
    for i in range(N_ATT)
])

n_already = already_incon.sum()
n_need = 2442 - n_already

if n_need > 0:
    # Force inconsistency on n_need additional rows
    consistent_rows = np.where(~already_incon)[0]
    force_incon_idx = np.random.choice(consistent_rows, size=n_need, replace=False)
    for i in force_incon_idx:
        # Swap to a different month
        cur_m = rata_mese_num[i]
        bad_months = [m for m in range(1, 13) if m != cur_m]
        bad_m = np.random.choice(bad_months)
        mese_att_col[i] = str(bad_m)          # numeric string, clearly different
elif n_need < 0:
    # Already have more than needed — force some back to consistent
    incon_rows = np.where(already_incon)[0]
    fix_idx = np.random.choice(incon_rows, size=-n_need, replace=False)
    for i in fix_idx:
        # Align mese and anno with RATA
        mese_att_col[i] = str(rata_mese_num[i])
        anno_att_col[i] = str(rata_anno_num[i])

# ------------------------------------------------------------------
# qualifica  (25.3% null → 5086/20102)
# ------------------------------------------------------------------
QUALIFICA_VALS = ["Dirigente", "Funzionario", "Impiegato", "Operatore"]
qualifica_col = np.random.choice(QUALIFICA_VALS, size=N_ATT).astype(object)
null_qual_idx = np.random.choice(N_ATT, size=5086, replace=False)
qualifica_col[null_qual_idx] = np.nan

# ------------------------------------------------------------------
# note  (98.5% null → 19802/20102)
# ------------------------------------------------------------------
note_att_col = np.full(N_ATT, np.nan, dtype=object)
note_non_null_att = np.random.choice(N_ATT, size=300, replace=False)
NOTE_ATT_VALS = ["Da riconciliare", "Verifica manuale", "Fonte secondaria", "Dato stimato"]
for pos, i in enumerate(note_non_null_att):
    note_att_col[i] = NOTE_ATT_VALS[pos % len(NOTE_ATT_VALS)]

# ------------------------------------------------------------------
# fonte_dato  (99.2% null → 19942/20102)
# ------------------------------------------------------------------
fonte_att_col = np.full(N_ATT, np.nan, dtype=object)
fonte_non_null_att = np.random.choice(N_ATT, size=160, replace=False)
fonte_att_col[fonte_non_null_att] = "Sistema legacy"

# ------------------------------------------------------------------
# tipo_contratto
# ------------------------------------------------------------------
tipo_contratto_col = np.random.choice(
    ["Indeterminato", "Determinato", "Part-time"],
    size=N_ATT,
    p=[0.60, 0.30, 0.10],
)

# ------------------------------------------------------------------
# Assemble attivazioniCessazioni DataFrame
# ------------------------------------------------------------------
df_att = pd.DataFrame({
    "RATA":              rata_att_col,
    "aggregation-time":  agg_time_att,
    "provincia_sede":    prov_noisy,
    "Provincia Sede":    prov_clean,
    "descrizione_ente":  desc_ente_col,
    "3descrizione":      desc3_col,
    "codice_ente":       codice_ente_col,
    "CODICE ENTE":       codice_ente_clean,
    "regione_sede":      reg_sede_col,
    "regione%sede":      reg_sede2_col,
    "attivazioni":       att_col,
    "att ivazioni":      att2_col,
    "cessazioni":        ces_col,
    "mese":              mese_att_col,
    "anno":              anno_att_col,
    "qualifica":         qualifica_col,
    "note":              note_att_col,
    "fonte_dato":        fonte_att_col,
    "tipo_contratto":    tipo_contratto_col,
})

assert len(df_att) == N_ATT, f"Expected {N_ATT} rows, got {len(df_att)}"
assert len(df_att.columns) == 19, f"Expected 19 columns, got {len(df_att.columns)}"

df_att.to_csv("data/raw/attivazioniCessazioni.csv", index=False)
print(f"  attivazioniCessazioni.csv saved: {len(df_att)} rows × {len(df_att.columns)} columns")

# Sanity checks
n_incon_final = 0
parsed_mese_final = np.array([_parse_mese(m) for m in df_att["mese"]], dtype=object)
parsed_anno_final = np.array([_parse_anno(a) for a in df_att["anno"]], dtype=object)
rata_mese_final   = np.array([int(str(r)[:6][4:]) if str(r)[:6].isdigit() else 0
                               for r in df_att["RATA"]])
rata_anno_final   = np.array([int(str(r)[:4]) if str(r)[:4].isdigit() else 0
                               for r in df_att["RATA"]])
for i in range(N_ATT):
    pm = parsed_mese_final[i]
    pa = parsed_anno_final[i]
    if pm is None or pa is None or pm != rata_mese_final[i] or pa != rata_anno_final[i]:
        n_incon_final += 1

print(f"  mese+anno vs RATA inconsistencies: {n_incon_final}")
print(f"  attivazioni non-numeric rows: {(~df_att['attivazioni'].apply(lambda x: str(x).replace(',','').replace('.','').replace('-','').isdigit())).sum()}")
print(f"  unique mese values: {df_att['mese'].nunique()}")
print(f"  unique anno values: {df_att['anno'].nunique()}")

print("\nDone. Files written to data/raw/")

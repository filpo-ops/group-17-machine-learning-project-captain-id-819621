"""
agents.py
=========
LangGraph multi-agent system for data quality analysis.

Architecture: Supervisor pattern with a linear pipeline and a conditional
feedback loop.  The graph visits five specialist agents in order:

    schema_agent -> completeness_agent -> consistency_agent
        -> anomaly_agent -> remediation_agent -> score_calculator

After score_calculator a routing function decides whether to loop back to
schema_agent (more iterations needed) or to stop.

Two operating modes
-------------------
- LLM mode   : OPENAI_API_KEY is set.  Each agent is a ReAct agent that can
               reason over its tools with GPT-4o-mini.
- Deterministic mode : no API key.  Tools are called directly; no LLM round-
                       trips occur.  Useful for offline testing.
"""

import os
import json
import re
import traceback
from typing import TypedDict, Optional, Any, List

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# LangGraph / LangChain imports
# ---------------------------------------------------------------------------
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Tool imports from tools.py (sibling module)
# ---------------------------------------------------------------------------
try:
    from tools import (
        check_naming_convention,
        check_data_types,
        detect_null_and_placeholders,
        calculate_completeness,
        detect_sparse_columns,
        check_format_consistency,
        check_cross_column_consistency,
        check_cross_column_logic,
        detect_duplicates,
        detect_outliers,
        detect_categorical_anomalies,
        calculate_reliability_score,
    )
    TOOLS_AVAILABLE = True
except ImportError as _tools_err:
    TOOLS_AVAILABLE = False
    print(
        f"[agents.py] WARNING: could not import from tools.py ({_tools_err}). "
        "Agents will run in stub mode."
    )

    # ------------------------------------------------------------------
    # Minimal stubs so this file is importable even without tools.py
    # ------------------------------------------------------------------
    def _stub(name):
        def _fn(*args, **kwargs):
            return {"tool": name, "issues": [], "summary": f"stub – {name}"}
        _fn.__name__ = name
        return _fn

    check_naming_convention       = _stub("check_naming_convention")
    check_data_types              = _stub("check_data_types")
    detect_null_and_placeholders  = _stub("detect_null_and_placeholders")
    calculate_completeness        = _stub("calculate_completeness")
    detect_sparse_columns         = _stub("detect_sparse_columns")
    check_format_consistency      = _stub("check_format_consistency")
    check_cross_column_consistency= _stub("check_cross_column_consistency")
    check_cross_column_logic      = _stub("check_cross_column_logic")
    detect_duplicates             = _stub("detect_duplicates")
    detect_outliers               = _stub("detect_outliers")
    detect_categorical_anomalies  = _stub("detect_categorical_anomalies")

    def calculate_reliability_score(*args, **kwargs):
        return 0.5


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODE = bool(OPENAI_API_KEY)

_llm = None  # lazy-initialised

def _get_llm():
    """Return a ChatOpenAI instance, initialising it on first call."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
    return _llm


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class DataQualityState(TypedDict):
    """Shared state flowing through every node of the LangGraph pipeline."""

    dataset: Any              # pd.DataFrame (current, possibly cleaned)
    original_dataset: Any     # pd.DataFrame (never modified)
    schema_report: dict
    completeness_report: dict
    consistency_report: dict
    anomaly_report: dict
    remediation_report: dict
    reliability_score: float
    iteration: int
    max_iterations: int
    all_reports: list         # cumulative history across iterations
    dataset_name: str


# ---------------------------------------------------------------------------
# Helper: merge tool outputs into a unified report dict
# ---------------------------------------------------------------------------

def _merge_issues(*tool_results) -> dict:
    """
    Merge one or more tool result dicts into a single report dict with an
    ``issues`` list and a ``summary`` string.
    """
    issues = []
    summaries = []
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        issues.extend(result.get("issues", []))
        s = result.get("summary", "")
        if s:
            summaries.append(s)
    return {
        "issues": issues,
        "summary": " | ".join(summaries) if summaries else "No issues found.",
    }


def _safe_call(fn, *args, **kwargs):
    """Call *fn* and return its result; on exception return an error dict."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "tool": getattr(fn, "__name__", str(fn)),
            "issues": [],
            "summary": f"ERROR in {getattr(fn, '__name__', fn)}: {exc}",
            "error": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Agent node: Schema Validation
# ---------------------------------------------------------------------------

_SCHEMA_SYSTEM_PROMPT = """
Sei lo Schema Validation Agent. Il tuo compito è analizzare la struttura e lo schema di un dataset CSV.
Hai a disposizione i seguenti tool:
- check_naming_convention: verifica la naming convention dei nomi di colonna
- check_data_types: verifica che i tipi di dato siano coerenti con il contesto di dominio

Per ogni problema trovato, DEVI produrre un report con:
1. La colonna coinvolta
2. Il tipo di problema (naming, data_type)
3. La severity (critical, warning, info)
4. Una descrizione del problema
5. L'audit trail: quale tool hai usato, quale regola hai applicato, quale ragionamento hai seguito

Il tuo output deve essere un JSON con la struttura:
{"agent": "schema_validation", "issues": [...], "summary": "..."}
""".strip()


def run_schema_agent(state: DataQualityState) -> DataQualityState:
    """
    Schema Validation Agent node.

    Checks column naming conventions and data-type coherence.
    Updates ``state["schema_report"]``.
    """
    df: pd.DataFrame = state["dataset"]

    try:
        if LLM_MODE:
            report = _run_llm_agent(
                system_prompt=_SCHEMA_SYSTEM_PROMPT,
                tools=[check_naming_convention, check_data_types],
                human_message=(
                    f"Analizza il dataset '{state['dataset_name']}' con "
                    f"colonne: {list(df.columns)}. Verifica naming e tipi."
                ),
                df=df,
            )
        else:
            r1 = _safe_call(check_naming_convention, df)
            r2 = _safe_call(check_data_types, df)
            report = _merge_issues(r1, r2)
            report["agent"] = "schema_validation"

    except Exception as exc:  # noqa: BLE001
        report = {
            "agent": "schema_validation",
            "issues": [],
            "summary": f"Schema agent error: {exc}",
            "error": traceback.format_exc(),
        }

    state["schema_report"] = report
    state["all_reports"] = state.get("all_reports", []) + [
        {"iteration": state.get("iteration", 0), "agent": "schema_validation", "report": report}
    ]
    return state


# ---------------------------------------------------------------------------
# Agent node: Completeness Analysis
# ---------------------------------------------------------------------------

_COMPLETENESS_SYSTEM_PROMPT = """
Sei il Completeness Analysis Agent. Il tuo compito è analizzare la completezza dei dati in un dataset CSV.
Hai a disposizione i seguenti tool:
- detect_null_and_placeholders: rileva valori nulli, placeholder e valori sentinella
- calculate_completeness: calcola la percentuale di completezza per ogni colonna
- detect_sparse_columns: identifica colonne con troppi valori mancanti

Per ogni problema trovato, DEVI produrre un report con:
1. La colonna coinvolta
2. Il tipo di problema (null, placeholder, sparse)
3. La severity (critical, warning, info)
4. Una descrizione del problema e la percentuale di dati mancanti
5. L'audit trail: quale tool hai usato e quale soglia hai applicato

Il tuo output deve essere un JSON con la struttura:
{"agent": "completeness_analysis", "issues": [...], "summary": "..."}
""".strip()


def run_completeness_agent(state: DataQualityState) -> DataQualityState:
    """
    Completeness Analysis Agent node.

    Detects nulls, placeholder values, and sparse columns.
    Updates ``state["completeness_report"]``.
    """
    df: pd.DataFrame = state["dataset"]

    try:
        if LLM_MODE:
            report = _run_llm_agent(
                system_prompt=_COMPLETENESS_SYSTEM_PROMPT,
                tools=[detect_null_and_placeholders, calculate_completeness, detect_sparse_columns],
                human_message=(
                    f"Analizza la completezza del dataset '{state['dataset_name']}'. "
                    f"Colonne: {list(df.columns)}. Righe: {len(df)}."
                ),
                df=df,
            )
        else:
            r1 = _safe_call(detect_null_and_placeholders, df)
            r2 = _safe_call(calculate_completeness, df)
            r3 = _safe_call(detect_sparse_columns, df)
            report = _merge_issues(r1, r2, r3)
            report["agent"] = "completeness_analysis"

    except Exception as exc:  # noqa: BLE001
        report = {
            "agent": "completeness_analysis",
            "issues": [],
            "summary": f"Completeness agent error: {exc}",
            "error": traceback.format_exc(),
        }

    state["completeness_report"] = report
    state["all_reports"] = state.get("all_reports", []) + [
        {"iteration": state.get("iteration", 0), "agent": "completeness_analysis", "report": report}
    ]
    return state


# ---------------------------------------------------------------------------
# Agent node: Consistency Validation
# ---------------------------------------------------------------------------

_CONSISTENCY_SYSTEM_PROMPT = """
Sei il Consistency Validation Agent. Il tuo compito è verificare la coerenza interna dei dati.
Hai a disposizione i seguenti tool:
- check_format_consistency: verifica che i formati siano uniformi all'interno di ogni colonna
- check_cross_column_consistency: verifica la consistenza tra colonne correlate
- check_cross_column_logic: verifica regole logiche tra colonne (es. data_inizio < data_fine)
- detect_duplicates: rileva righe duplicate o quasi-duplicate

Per ogni problema trovato, DEVI produrre un report con:
1. La colonna o coppia di colonne coinvolta
2. Il tipo di problema (format, cross_column, logic, duplicate)
3. La severity (critical, warning, info)
4. Una descrizione del problema
5. L'audit trail

Il tuo output deve essere un JSON con la struttura:
{"agent": "consistency_validation", "issues": [...], "summary": "..."}
""".strip()


def run_consistency_agent(state: DataQualityState) -> DataQualityState:
    """
    Consistency Validation Agent node.

    Checks format uniformity, cross-column relationships, logical rules,
    and duplicates.  Updates ``state["consistency_report"]``.
    """
    df: pd.DataFrame = state["dataset"]

    try:
        if LLM_MODE:
            report = _run_llm_agent(
                system_prompt=_CONSISTENCY_SYSTEM_PROMPT,
                tools=[
                    check_format_consistency,
                    check_cross_column_consistency,
                    check_cross_column_logic,
                    detect_duplicates,
                ],
                human_message=(
                    f"Verifica la consistenza del dataset '{state['dataset_name']}'. "
                    f"Colonne: {list(df.columns)}."
                ),
                df=df,
            )
        else:
            r1 = _safe_call(check_format_consistency, df)
            r2 = _safe_call(check_cross_column_consistency, df)
            r3 = _safe_call(check_cross_column_logic, df)
            r4 = _safe_call(detect_duplicates, df)
            report = _merge_issues(r1, r2, r3, r4)
            report["agent"] = "consistency_validation"

    except Exception as exc:  # noqa: BLE001
        report = {
            "agent": "consistency_validation",
            "issues": [],
            "summary": f"Consistency agent error: {exc}",
            "error": traceback.format_exc(),
        }

    state["consistency_report"] = report
    state["all_reports"] = state.get("all_reports", []) + [
        {"iteration": state.get("iteration", 0), "agent": "consistency_validation", "report": report}
    ]
    return state


# ---------------------------------------------------------------------------
# Agent node: Anomaly Detection
# ---------------------------------------------------------------------------

_ANOMALY_SYSTEM_PROMPT = """
Sei l'Anomaly Detection Agent. Il tuo compito è individuare valori anomali e outlier nel dataset.
Hai a disposizione i seguenti tool:
- detect_outliers: individua outlier numerici tramite IQR e z-score
- detect_categorical_anomalies: individua valori categorici anomali o rari

Per ogni anomalia trovata, DEVI produrre un report con:
1. La colonna coinvolta
2. Il tipo di anomalia (outlier, categorical_anomaly)
3. La severity (critical, warning, info)
4. I valori anomali trovati e la loro frequenza
5. L'audit trail: metodo usato (IQR/z-score/frequenza), soglie applicate

Il tuo output deve essere un JSON con la struttura:
{"agent": "anomaly_detection", "issues": [...], "summary": "..."}
""".strip()


def run_anomaly_agent(state: DataQualityState) -> DataQualityState:
    """
    Anomaly Detection Agent node.

    Detects numerical outliers and categorical anomalies.
    Updates ``state["anomaly_report"]``.
    """
    df: pd.DataFrame = state["dataset"]

    try:
        if LLM_MODE:
            report = _run_llm_agent(
                system_prompt=_ANOMALY_SYSTEM_PROMPT,
                tools=[detect_outliers, detect_categorical_anomalies],
                human_message=(
                    f"Cerca anomalie nel dataset '{state['dataset_name']}'. "
                    f"Colonne numeriche: "
                    f"{list(df.select_dtypes(include='number').columns)}."
                ),
                df=df,
            )
        else:
            r1 = _safe_call(detect_outliers, df)
            r2 = _safe_call(detect_categorical_anomalies, df)
            report = _merge_issues(r1, r2)
            report["agent"] = "anomaly_detection"

    except Exception as exc:  # noqa: BLE001
        report = {
            "agent": "anomaly_detection",
            "issues": [],
            "summary": f"Anomaly agent error: {exc}",
            "error": traceback.format_exc(),
        }

    state["anomaly_report"] = report
    state["all_reports"] = state.get("all_reports", []) + [
        {"iteration": state.get("iteration", 0), "agent": "anomaly_detection", "report": report}
    ]
    return state


# ---------------------------------------------------------------------------
# Agent node: Remediation  (fully deterministic – no LLM required)
# ---------------------------------------------------------------------------

# Italian month name/abbreviation -> two-digit string mapping
_IT_MONTH_MAP = {
    "gen": "01", "gennaio": "01",
    "feb": "02", "febbraio": "02",
    "mar": "03", "marzo": "03",
    "apr": "04", "aprile": "04",
    "mag": "05", "maggio": "05",
    "giu": "06", "giugno": "06",
    "lug": "07", "luglio": "07",
    "ago": "08", "agosto": "08",
    "set": "09", "settembre": "09",
    "ott": "10", "ottobre": "10",
    "nov": "11", "novembre": "11",
    "dic": "12", "dicembre": "12",
}


def _to_snake_case(name: str) -> str:
    """Convert a column name to snake_case."""
    name = re.sub(r"[\s\-]+", "_", name.strip())
    name = re.sub(r"[^\w]", "", name)
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return name.lower().strip("_")


def _standardize_rata(value) -> Any:
    """
    Normalise a ``rata``/``RATA`` value to YYYYMM format (string).

    Handles:
        - "03/2024"   -> "202403"
        - "MAR-2024"  -> "202403"
        - "LUG-2024"  -> "202407"
        - "2024-01"   -> "202401"
        - "Rata 2024" -> NaN (no month info; caller may fill from *anno*)
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()

    # Already in YYYYMM
    if re.fullmatch(r"\d{6}", s):
        return s

    # MM/YYYY
    m = re.fullmatch(r"(\d{1,2})[/\-](\d{4})", s)
    if m:
        return f"{m.group(2)}{int(m.group(1)):02d}"

    # YYYY-MM or YYYY/MM
    m = re.fullmatch(r"(\d{4})[/\-](\d{1,2})", s)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}"

    # MMM-YYYY  (Italian month abbreviation/name)
    m = re.fullmatch(r"([A-Za-z]{3,9})[.\-\s]+(\d{4})", s)
    if m:
        month_key = m.group(1).lower()
        year = m.group(2)
        month_num = _IT_MONTH_MAP.get(month_key)
        if month_num:
            return f"{year}{month_num}"

    # "Rata YYYY" or "rata YYYY" – month unknown
    m = re.fullmatch(r"[Rr]ata\s+(\d{4})", s)
    if m:
        return np.nan  # caller may fill from anno column

    return np.nan


def _standardize_mese(value) -> Any:
    """
    Normalise a ``mese`` value to an integer 1–12.

    Handles Italian month names/abbreviations, "mese N" patterns and bare
    integers.  Out-of-range values become NaN.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()

    # Bare number
    if re.fullmatch(r"\d{1,2}", s):
        n = int(s)
        return n if 1 <= n <= 12 else np.nan

    # "mese N"
    m = re.fullmatch(r"mese\s*(\d{1,2})", s)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 12 else np.nan

    # Italian name / abbreviation
    month_num = _IT_MONTH_MAP.get(s)
    if month_num:
        return int(month_num)

    return np.nan


def _standardize_anno(value) -> Any:
    """
    Normalise an ``anno`` value to a 4-digit integer.

    Handles: "2023.", "2023", "23", "anno 2023", "Anno 2023".
    Two-digit years are assumed 2000+.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()

    # Strip trailing punctuation / text prefix
    s = re.sub(r"^anno\s*", "", s)
    s = s.rstrip(".")

    if re.fullmatch(r"\d{4}", s):
        return int(s)
    if re.fullmatch(r"\d{2}", s):
        return 2000 + int(s)

    return np.nan


def _clean_numeric(value) -> Any:
    """
    Convert a messy currency/numeric string to float.

    Strips €, EUR, spaces; replaces comma decimal separator; maps
    "N.D." and similar sentinels to NaN.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if re.fullmatch(r"[Nn]\.[Dd]\.?|N/A|n/a|—|-", s):
        return np.nan
    s = re.sub(r"[€EUReur\s]", "", s)
    s = s.replace(".", "").replace(",", ".")  # EU number format
    try:
        return float(s)
    except ValueError:
        return np.nan


def _apply_remediation(df: pd.DataFrame, schema_report: dict,
                       completeness_report: dict, consistency_report: dict,
                       anomaly_report: dict) -> tuple[pd.DataFrame, list]:
    """
    Apply all remediation transformations to *df* deterministically.

    Returns
    -------
    cleaned_df : pd.DataFrame
    audit_trail : list of str
    """
    df = df.copy()
    audit: list[str] = []

    # ------------------------------------------------------------------
    # 1. Rename columns: bad naming -> snake_case
    # ------------------------------------------------------------------
    bad_naming_cols = set()
    for issue in schema_report.get("issues", []):
        if issue.get("type") in ("naming", "naming_convention"):
            col = issue.get("column")
            if col:
                bad_naming_cols.add(col)

    rename_map = {}
    for col in df.columns:
        snake = _to_snake_case(col)
        if snake != col or col in bad_naming_cols:
            rename_map[col] = snake

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        audit.append(f"Renamed columns: {rename_map}")

    # ------------------------------------------------------------------
    # 2. Replace placeholder values with NaN
    # ------------------------------------------------------------------
    placeholder_patterns = [
        r"^[Nn]\.[Dd]\.?$", r"^N/A$", r"^n/a$", r"^-$", r"^—$",
        r"^NULL$", r"^null$", r"^none$", r"^NONE$", r"^\?+$",
    ]
    placeholder_re = re.compile("|".join(placeholder_patterns))
    replaced_count = 0
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].astype(str).str.fullmatch(placeholder_re.pattern, na=False)
        replaced_count += mask.sum()
        df.loc[mask, col] = np.nan
    if replaced_count:
        audit.append(f"Replaced {replaced_count} placeholder values with NaN.")

    # ------------------------------------------------------------------
    # 3. Numeric columns: spesa / attivazioni / cessazioni
    # ------------------------------------------------------------------
    numeric_keywords = ["spesa", "attivazioni", "cessazioni", "importo", "valore",
                        "costo", "ricavo", "fatturato"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in numeric_keywords):
            if df[col].dtype == object:
                before_nulls = df[col].isna().sum()
                df[col] = df[col].apply(_clean_numeric)
                after_nulls = df[col].isna().sum()
                new_nulls = after_nulls - before_nulls
                audit.append(
                    f"Converted '{col}' to numeric "
                    f"(introduced {new_nulls} NaN for non-parseable values)."
                )

    # ------------------------------------------------------------------
    # 4. Standardize rata / RATA columns -> YYYYMM
    # ------------------------------------------------------------------
    rata_cols = [c for c in df.columns if re.search(r"\brata\b", c, re.IGNORECASE)]
    anno_cols = [c for c in df.columns if re.search(r"\banno\b", c, re.IGNORECASE)]

    for col in rata_cols:
        converted = df[col].apply(_standardize_rata)
        # Fill "Rata YYYY" blanks using anno column if available
        missing_mask = converted.isna() & df[col].notna()
        if missing_mask.any() and anno_cols:
            anno_col = anno_cols[0]
            for idx in df[missing_mask].index:
                raw_rata = str(df.at[idx, col]).strip()
                m = re.fullmatch(r"[Rr]ata\s+(\d{4})", raw_rata)
                if m and not pd.isna(df.at[idx, anno_col]):
                    year = m.group(1)
                    converted.at[idx] = f"{year}??"  # month unknown; keep partial
        df[col] = converted
        audit.append(f"Standardized '{col}' to YYYYMM format.")

    # ------------------------------------------------------------------
    # 5. Standardize mese columns -> int 1-12
    # ------------------------------------------------------------------
    mese_cols = [c for c in df.columns if re.search(r"\bmese\b", c, re.IGNORECASE)]
    for col in mese_cols:
        df[col] = df[col].apply(_standardize_mese)
        audit.append(f"Standardized '{col}' to integer 1-12.")

    # ------------------------------------------------------------------
    # 6. Standardize anno columns -> int YYYY
    # ------------------------------------------------------------------
    for col in anno_cols:
        df[col] = df[col].apply(_standardize_anno)
        audit.append(f"Standardized '{col}' to 4-digit integer.")

    # ------------------------------------------------------------------
    # 7. Duplicate columns: keep more complete, fill nulls from the other
    # ------------------------------------------------------------------
    dup_col_issues = [
        i for i in consistency_report.get("issues", [])
        if i.get("type") in ("duplicate_column", "duplicate_columns")
    ]
    for issue in dup_col_issues:
        cols_involved = issue.get("columns", [])
        if len(cols_involved) >= 2:
            c1, c2 = cols_involved[0], cols_involved[1]
            if c1 in df.columns and c2 in df.columns:
                null_c1 = df[c1].isna().sum()
                null_c2 = df[c2].isna().sum()
                keep, drop = (c1, c2) if null_c1 <= null_c2 else (c2, c1)
                df[keep] = df[keep].fillna(df[drop])
                df.drop(columns=[drop], inplace=True)
                audit.append(
                    f"Merged duplicate columns '{c1}' & '{c2}': kept '{keep}', "
                    f"filled NaNs from '{drop}', then dropped '{drop}'."
                )

    # ------------------------------------------------------------------
    # 8. Remove exact duplicate rows
    # ------------------------------------------------------------------
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_after = len(df)
    if n_before != n_after:
        audit.append(f"Removed {n_before - n_after} exact duplicate rows.")

    # ------------------------------------------------------------------
    # 9. Add _outlier_flag for critical spesa outliers
    # ------------------------------------------------------------------
    critical_outlier_cols = set()
    for issue in anomaly_report.get("issues", []):
        if issue.get("severity") == "critical" and issue.get("type") == "outlier":
            col = issue.get("column")
            if col and col in df.columns:
                critical_outlier_cols.add(col)

    # Also auto-detect spesa columns with extreme outliers
    for col in df.select_dtypes(include="number").columns:
        if "spesa" in col.lower() and col not in critical_outlier_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                if ((df[col] < lower) | (df[col] > upper)).any():
                    critical_outlier_cols.add(col)

    for col in critical_outlier_cols:
        if col not in df.columns:
            continue
        flag_col = f"{col}_outlier_flag"
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        audit.append(
            f"Added outlier flag column '{flag_col}' for '{col}' "
            f"(IQR method, bounds [{lower:.2f}, {upper:.2f}])."
        )

    # ------------------------------------------------------------------
    # 10. Standardize province to uppercase
    # ------------------------------------------------------------------
    province_cols = [
        c for c in df.columns
        if re.search(r"\bprov(incia)?\b", c, re.IGNORECASE)
    ]
    for col in province_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.upper()
            audit.append(f"Standardized '{col}' to uppercase.")

    return df, audit


def run_remediation_agent(state: DataQualityState) -> DataQualityState:
    """
    Remediation Agent node (fully deterministic).

    Applies all cleaning transformations to the DataFrame based on the
    accumulated reports from previous agents.  Updates both
    ``state["dataset"]`` and ``state["remediation_report"]``.
    """
    df: pd.DataFrame = state["dataset"]

    try:
        cleaned_df, audit_trail = _apply_remediation(
            df=df,
            schema_report=state.get("schema_report", {}),
            completeness_report=state.get("completeness_report", {}),
            consistency_report=state.get("consistency_report", {}),
            anomaly_report=state.get("anomaly_report", {}),
        )

        report = {
            "agent": "remediation",
            "actions_taken": audit_trail,
            "rows_before": len(df),
            "rows_after": len(cleaned_df),
            "columns_before": len(df.columns),
            "columns_after": len(cleaned_df.columns),
            "summary": (
                f"Applied {len(audit_trail)} remediation actions. "
                f"Rows: {len(df)} -> {len(cleaned_df)}. "
                f"Columns: {len(df.columns)} -> {len(cleaned_df.columns)}."
            ),
        }
        state["dataset"] = cleaned_df

    except Exception as exc:  # noqa: BLE001
        report = {
            "agent": "remediation",
            "actions_taken": [],
            "summary": f"Remediation agent error: {exc}",
            "error": traceback.format_exc(),
        }

    state["remediation_report"] = report
    state["all_reports"] = state.get("all_reports", []) + [
        {"iteration": state.get("iteration", 0), "agent": "remediation", "report": report}
    ]
    return state


# ---------------------------------------------------------------------------
# Score Calculator node
# ---------------------------------------------------------------------------

def calculate_score_node(state: DataQualityState) -> DataQualityState:
    """
    Score Calculator node.

    Computes the overall reliability score from all four quality reports
    and increments the iteration counter.
    """
    try:
        score = calculate_reliability_score(
            schema_report=state.get("schema_report", {}),
            completeness_report=state.get("completeness_report", {}),
            consistency_report=state.get("consistency_report", {}),
            anomaly_report=state.get("anomaly_report", {}),
            df=state["dataset"],
        )
    except Exception as exc:  # noqa: BLE001
        # Fallback: simple heuristic based on issue severities
        score = _fallback_score(state)
        print(f"[calculate_score_node] calculate_reliability_score failed ({exc}); "
              f"using fallback score {score:.3f}.")

    state["reliability_score"] = float(score)
    state["iteration"] = state.get("iteration", 0) + 1
    return state


def _fallback_score(state: DataQualityState) -> float:
    """
    Simple fallback scoring: subtract penalties for critical/warning issues
    found across all reports.
    """
    score = 1.0
    for key in ["schema_report", "completeness_report", "consistency_report", "anomaly_report"]:
        report = state.get(key, {})
        for issue in report.get("issues", []):
            sev = issue.get("severity", "info")
            if sev == "critical":
                score -= 0.05
            elif sev == "warning":
                score -= 0.01
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Routing function (used in conditional edge)
# ---------------------------------------------------------------------------

def _routing_function(state: DataQualityState) -> str:
    """
    Decide whether to loop back to ``schema_agent`` or to end the graph.

    Continues if:
    - ``reliability_score < 0.75``, OR
    - any critical issue still exists in any report

    AND the iteration limit has not been reached.
    """
    score = state.get("reliability_score", 0.0)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # Check for any remaining critical issue
    has_critical = False
    for report_key in ["schema_report", "completeness_report",
                       "consistency_report", "anomaly_report"]:
        report = state.get(report_key, {})
        issues = report.get("issues", [])
        if any(i.get("severity") == "critical" for i in issues):
            has_critical = True
            break

    if score >= 0.75 and not has_critical:
        return END
    if iteration >= max_iter:
        return END
    return "schema_agent"


# ---------------------------------------------------------------------------
# LLM agent helper (ReAct via langchain)
# ---------------------------------------------------------------------------

def _run_llm_agent(system_prompt: str, tools: list, human_message: str,
                   df: pd.DataFrame) -> dict:
    """
    Run a LangChain ReAct agent with the given *tools* and return a
    normalised report dict.

    Falls back to deterministic tool calls on error.
    """
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub

        llm = _get_llm()
        llm_with_tools = llm.bind_tools(tools)

        # Build a minimal prompt if hub pull fails
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\n{tools}\n\nTool names: {tool_names}"),
                ("human", "{input}\n\n{agent_scratchpad}"),
            ])

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False,
                                 handle_parsing_errors=True, max_iterations=5)
        result = executor.invoke({
            "input": f"{system_prompt}\n\n{human_message}",
        })

        # Try to extract JSON from the output
        output = result.get("output", "{}")
        try:
            # Find the first JSON object in the output
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {"issues": [], "summary": output}

    except Exception as exc:  # noqa: BLE001
        print(f"[_run_llm_agent] LLM agent failed ({exc}); "
              "falling back to deterministic tool calls.")
        results = [_safe_call(t, df) for t in tools]
        return _merge_issues(*results)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build and compile the LangGraph StateGraph for the data quality pipeline.

    Returns
    -------
    CompiledGraph
        The compiled graph ready to be invoked with an initial
        ``DataQualityState``.
    """
    graph = StateGraph(DataQualityState)

    # Add all nodes
    graph.add_node("schema_agent", run_schema_agent)
    graph.add_node("completeness_agent", run_completeness_agent)
    graph.add_node("consistency_agent", run_consistency_agent)
    graph.add_node("anomaly_agent", run_anomaly_agent)
    graph.add_node("remediation_agent", run_remediation_agent)
    graph.add_node("score_calculator", calculate_score_node)

    # Linear pipeline edges
    graph.add_edge("schema_agent", "completeness_agent")
    graph.add_edge("completeness_agent", "consistency_agent")
    graph.add_edge("consistency_agent", "anomaly_agent")
    graph.add_edge("anomaly_agent", "remediation_agent")
    graph.add_edge("remediation_agent", "score_calculator")

    # Conditional feedback loop after score calculation
    graph.add_conditional_edges(
        "score_calculator",
        _routing_function,
        {END: END, "schema_agent": "schema_agent"},
    )

    graph.set_entry_point("schema_agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    max_iterations: int = 3,
) -> DataQualityState:
    """
    Run the complete data quality pipeline on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to analyse.
    dataset_name : str
        A human-readable label for the dataset (used in reports).
    max_iterations : int
        Maximum number of analysis-remediation cycles.

    Returns
    -------
    DataQualityState
        The final state after the pipeline has completed, including the
        cleaned ``dataset``, all reports, and the final
        ``reliability_score``.

    Examples
    --------
    >>> import pandas as pd
    >>> from agents import run_pipeline
    >>> df = pd.read_csv("data.csv")
    >>> result = run_pipeline(df, dataset_name="my_data")
    >>> print(result["reliability_score"])
    """
    app = build_graph()

    initial_state: DataQualityState = DataQualityState(
        dataset=df,
        original_dataset=df.copy(),
        schema_report={},
        completeness_report={},
        consistency_report={},
        anomaly_report={},
        remediation_report={},
        reliability_score=0.0,
        iteration=0,
        max_iterations=max_iterations,
        all_reports=[],
        dataset_name=dataset_name,
    )

    final_state = app.invoke(initial_state)
    return final_state


# ---------------------------------------------------------------------------
# __main__ – simple smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Data Quality Pipeline – smoke test")
    print("=" * 60)

    # Build a small synthetic dataset with intentional quality issues
    test_data = {
        "CodiceCliente":   [101, 102, 103, 104, 104],          # duplicate row
        "Nome Prodotto":   ["Fibra", "ADSL", "Fibra", None, "ADSL"],  # bad name, null
        "Spesa Mensile":   ["€ 29,90", "N.D.", "35.00", "€ 120,00", "€ 120,00"],
        "mese":            ["marzo", "4", "mese 7", "13", None],
        "anno":            ["2023.", "23", "anno 2024", "2024", "2024"],
        "RATA":            ["03/2024", "MAR-2024", "2024-01", "Rata 2024", "LUG-2024"],
        "Provincia":       ["mi", "RM", "Na", None, "RM"],
    }

    df_test = pd.DataFrame(test_data)
    print("\nInput DataFrame:")
    print(df_test.to_string())
    print()

    mode = "LLM" if LLM_MODE else "Deterministic (no API key)"
    print(f"Operating mode: {mode}")
    print()

    try:
        result = run_pipeline(df_test, dataset_name="test_telecom", max_iterations=2)

        print(f"Final reliability score : {result['reliability_score']:.3f}")
        print(f"Total iterations run    : {result['iteration']}")
        print(f"Audit events logged     : {len(result.get('all_reports', []))}")

        print("\nRemediation summary:")
        rem = result.get("remediation_report", {})
        print(f"  {rem.get('summary', 'N/A')}")
        for action in rem.get("actions_taken", []):
            print(f"  - {action}")

        print("\nCleaned DataFrame:")
        print(result["dataset"].to_string())

    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()

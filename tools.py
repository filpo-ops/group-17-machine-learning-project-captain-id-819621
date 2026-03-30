"""
tools.py — Deterministic Data Quality Tools for the NoiPA project.

All 11 tools are pure Python functions decorated with @tool from langchain_core.tools
so they are directly usable in a LangGraph agent workflow.

Each tool returns list[dict] with keys:
    column   : str
    row      : int | None
    issue    : str
    severity : "critical" | "warning" | "info"
    details  : str

Additional helper: calculate_reliability_score returns a dict of dimension scores.
"""

import re
import pandas as pd
import numpy as np
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as @tool)
# ---------------------------------------------------------------------------

_NOIIPA_PLACEHOLDERS: list[str] = [
    "N.D.", "n.d.", "N/A", "n/a", "-", "?", "//",
    "unknown", "Unknown", "UNKNOWN", " ", "",
]

_NOIIPA_CRITICAL_COLS: set[str] = {
    "spesa", "attivazioni", "cessazioni", "codice_ente", "ente",
}
_NOIIPA_WARNING_COLS: set[str] = {
    "provincia_sede", "regione_sede", "tipo_imposta", "qualifica",
}
_NOIIPA_INFO_COLS: set[str] = {
    "note", "fonte_dato", "area_geografica",
}

_NOIIPA_EXPECTED_TYPES: dict[str, str] = {
    "spesa": "float",
    "rata": "str",
    "RATA": "str",
    "attivazioni": "int",
    "cessazioni": "int",
    "anno": "int",
    "mese": "int",
}

_SPESA_CROSS_PAIRS: list[tuple[str, str]] = [
    ("tipo_imposta", "Tipo Imposta"),
    ("spesa", "SPESA TOTALE"),
    ("cod_imposta", "2cod_imposta"),
    ("ente", "ente%code"),
]

_ATT_CESS_CROSS_PAIRS: list[tuple[str, str]] = [
    ("provincia_sede", "Provincia Sede"),
    ("descrizione_ente", "3descrizione"),
    ("codice_ente", "CODICE ENTE"),
    ("regione_sede", "regione%sede"),
    ("attivazioni", "att ivazioni"),
]

# Pre-configured format patterns per NoiPA column
_FORMAT_PATTERNS: dict[str, dict] = {
    "rata": {
        "primary": r"^\d{6}$",
        "alternatives": [
            r"^\d{2}/\d{4}$",
            r"^[A-Z]{3}-\d{4}$",
            r"^Rata \d{4}$",
            r"^\d{4}-\d{2}$",
        ],
    },
    "RATA": {
        "primary": r"^\d{6}$",
        "alternatives": [
            r"^\d{2}/\d{4}$",
            r"^[A-Z]{3}-\d{4}$",
            r"^Rata \d{4}$",
            r"^\d{4}-\d{2}$",
        ],
    },
    "mese": {
        "primary": r"^\d{1,2}$",
        "alternatives": [
            r"^(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)$",
            r"^(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)$",
            r"^[Mm]ese \d{1,2}$",
        ],
    },
    "anno": {
        "primary": r"^\d{4}$",
        "alternatives": [
            r"^\d{2}$",
            r"^\d{2}\.\d{2}$",
            r"^[Aa]nno \d{4}$",
        ],
    },
    "spesa": {
        "primary": r"^-?\d+(\.\d+)?$",
        "alternatives": [
            r"^-?\d+,\d+$",
            r"^€\s*-?\d+",
            r"^EUR\s*-?\d+",
        ],
    },
    "aggregation-time": {
        "primary": r"^\d{4}-\d{2}-\d{2}$",
        "alternatives": [
            r"^\d{2}/\d{2}/\d{4}$",
            r"^\d{2}\.\d{2}\.\d{4}$",
            r"^\d{4}-\d{2}$",
        ],
    },
}


def _normalize_col_name(name: str) -> str:
    """Lowercase, strip, replace non-alphanumeric with underscore."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _col_severity(col: str) -> str:
    """Return NoiPA severity for a given column."""
    if col in _NOIIPA_CRITICAL_COLS:
        return "critical"
    if col in _NOIIPA_WARNING_COLS:
        return "warning"
    return "info"


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols and replace commas for numeric conversion."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[€EUReur\s]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _is_placeholder(val, placeholders: list[str]) -> bool:
    if pd.isna(val):
        return False  # genuine NaN handled separately
    return str(val) in placeholders


# ---------------------------------------------------------------------------
# Tool 1 — check_naming_convention
# ---------------------------------------------------------------------------

@tool
def check_naming_convention(df: pd.DataFrame) -> list[dict]:
    """
    Check every column name in the DataFrame for NoiPA naming-convention violations.

    Detects: special characters (%, -, space), names starting with a digit,
    non-snake_case naming, and semantically duplicate columns (after normalization).
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    cols = list(df.columns)
    normalized: dict[str, list[str]] = {}

    for col in cols:
        col_str = str(col)

        # Special characters
        if re.search(r"[%\-\s]", col_str):
            issues.append({
                "column": col_str,
                "row": None,
                "issue": "special_char_in_column_name",
                "severity": "warning",
                "details": (
                    f"Column '{col_str}' contains special characters "
                    f"(%, -, or space). Use snake_case instead."
                ),
            })

        # Starts with digit
        if re.match(r"^\d", col_str):
            issues.append({
                "column": col_str,
                "row": None,
                "issue": "column_name_starts_with_digit",
                "severity": "warning",
                "details": (
                    f"Column '{col_str}' starts with a digit, "
                    "which is invalid in most environments."
                ),
            })

        # Not snake_case (contains uppercase after stripping leading digits)
        stripped = re.sub(r"^\d+", "", col_str)
        if not re.match(r"^[a-z0-9_]+$", stripped):
            issues.append({
                "column": col_str,
                "row": None,
                "issue": "not_snake_case",
                "severity": "warning",
                "details": (
                    f"Column '{col_str}' is not in snake_case format "
                    "(contains uppercase letters or unsupported characters)."
                ),
            })

        norm = _normalize_col_name(col_str)
        normalized.setdefault(norm, []).append(col_str)

    # Semantically duplicate columns
    for norm_key, orig_cols in normalized.items():
        if len(orig_cols) > 1:
            # Check whether they carry divergent data
            try:
                candidate_series = [df[c].dropna().astype(str) for c in orig_cols]
                all_same = all(
                    candidate_series[0].reset_index(drop=True).equals(
                        s.reset_index(drop=True)
                    )
                    for s in candidate_series[1:]
                )
                severity = "warning" if all_same else "critical"
                detail_suffix = (
                    "Values appear identical."
                    if all_same
                    else "Values DIVERGE — potential data inconsistency!"
                )
            except Exception:
                severity = "warning"
                detail_suffix = "Could not compare values."

            issues.append({
                "column": ", ".join(orig_cols),
                "row": None,
                "issue": "semantically_duplicate_columns",
                "severity": severity,
                "details": (
                    f"Columns {orig_cols} normalise to the same key '{norm_key}'. "
                    + detail_suffix
                ),
            })

    return issues


# ---------------------------------------------------------------------------
# Tool 2 — check_data_types
# ---------------------------------------------------------------------------

@tool
def check_data_types(df: pd.DataFrame, expected_types: dict = None) -> list[dict]:
    """
    Verify that each column conforms to its expected data type for NoiPA data.

    Uses default NoiPA expected types if none are provided. Reports columns
    where a significant fraction of values cannot be coerced to the expected type.
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    if expected_types is None:
        expected_types = _NOIIPA_EXPECTED_TYPES

    for col, expected in expected_types.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        total = len(series)
        failures = 0

        if expected in ("float", "int"):
            numeric = pd.to_numeric(series, errors="coerce")
            failures = numeric.isna().sum()
            if expected == "int" and failures == 0:
                # Check that values are whole numbers
                non_integer = numeric[numeric != numeric.round()].count()
                if non_integer > 0:
                    failures = int(non_integer)
        elif expected == "str":
            # Everything can be a string; check for purely whitespace entries
            failures = series.astype(str).str.strip().eq("").sum()
        else:
            continue

        failure_rate = failures / total if total > 0 else 0.0

        if failure_rate == 0:
            continue

        if failure_rate > 0.05:
            severity = "critical"
        elif failure_rate > 0.01:
            severity = "warning"
        else:
            severity = "info"

        issues.append({
            "column": col,
            "row": None,
            "issue": "type_mismatch",
            "severity": severity,
            "details": (
                f"Column '{col}' expected '{expected}' but "
                f"{failures}/{total} values ({failure_rate:.1%}) could not be coerced."
            ),
        })

    return issues


# ---------------------------------------------------------------------------
# Tool 3 — detect_null_and_placeholders
# ---------------------------------------------------------------------------

@tool
def detect_null_and_placeholders(
    df: pd.DataFrame, placeholders: list = None
) -> list[dict]:
    """
    Detect genuine NaN values and domain-specific placeholder strings in a NoiPA DataFrame.

    Uses a default placeholder list if none is supplied. Assigns severity based on
    the NoiPA column classification (critical / warning / info).
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    if placeholders is None:
        placeholders = _NOIIPA_PLACEHOLDERS

    placeholder_set = set(placeholders)

    for col in df.columns:
        col_str = str(col)
        severity = _col_severity(col_str)

        series = df[col]
        null_mask = series.isna()
        null_count = int(null_mask.sum())

        placeholder_rows: list[int] = []
        for idx, val in series.items():
            if not pd.isna(val) and str(val) in placeholder_set:
                placeholder_rows.append(int(idx))

        if null_count > 0:
            issues.append({
                "column": col_str,
                "row": None,
                "issue": "null_values",
                "severity": severity,
                "details": (
                    f"Column '{col_str}' has {null_count} null value(s) "
                    f"({null_count / len(series):.1%} of rows)."
                ),
            })

        if placeholder_rows:
            sample = placeholder_rows[:5]
            issues.append({
                "column": col_str,
                "row": sample[0] if len(sample) == 1 else None,
                "issue": "placeholder_values",
                "severity": severity,
                "details": (
                    f"Column '{col_str}' has {len(placeholder_rows)} placeholder(s). "
                    f"Sample row indices: {sample}."
                ),
            })

    return issues


# ---------------------------------------------------------------------------
# Tool 4 — calculate_completeness
# ---------------------------------------------------------------------------

@tool
def calculate_completeness(df: pd.DataFrame, placeholders: list = None) -> dict:
    """
    Calculate per-column and overall completeness rates for a NoiPA DataFrame.

    Counts both genuine NaN values and known placeholder strings as incomplete.
    Returns a dict with keys 'columns' (list of per-column stats) and
    'overall_completeness' (float 0-1).
    """
    if df is None or df.empty:
        return {"columns": [], "overall_completeness": 0.0}

    if placeholders is None:
        placeholders = _NOIIPA_PLACEHOLDERS

    placeholder_set = set(placeholders)
    total_rows = len(df)
    col_stats: list[dict] = []
    total_complete = 0

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        null_count = int(series.isna().sum())
        ph_count = int(
            series.dropna()
            .astype(str)
            .apply(lambda v: v in placeholder_set)
            .sum()
        )
        incomplete = null_count + ph_count
        complete = total_rows - incomplete
        rate = complete / total_rows if total_rows > 0 else 0.0
        total_complete += complete

        col_stats.append({
            "column": col_str,
            "completeness_rate": round(rate, 4),
            "null_count": null_count,
            "placeholder_count": ph_count,
        })

    overall = total_complete / (total_rows * len(df.columns)) if (total_rows * len(df.columns)) > 0 else 0.0

    return {
        "columns": col_stats,
        "overall_completeness": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Tool 5 — detect_sparse_columns
# ---------------------------------------------------------------------------

@tool
def detect_sparse_columns(df: pd.DataFrame, threshold: float = 0.90) -> list[dict]:
    """
    Identify columns in the DataFrame whose null/placeholder rate exceeds the threshold.

    Columns above 90% sparsity are flagged as 'investigate'; above 95% as
    'candidate_for_removal'. Severity is always 'warning'.
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    total_rows = len(df)
    if total_rows == 0:
        return issues

    placeholder_set = set(_NOIIPA_PLACEHOLDERS)

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        null_count = series.isna().sum()
        ph_count = (
            series.dropna().astype(str).apply(lambda v: v in placeholder_set).sum()
        )
        empty_count = null_count + ph_count
        sparsity = empty_count / total_rows

        if sparsity > 0.95:
            tag = "candidate_for_removal"
        elif sparsity > threshold:
            tag = "investigate"
        else:
            continue

        issues.append({
            "column": col_str,
            "row": None,
            "issue": f"sparse_column_{tag}",
            "severity": "warning",
            "details": (
                f"Column '{col_str}' is {sparsity:.1%} empty/null "
                f"({int(empty_count)}/{total_rows} rows). "
                f"Recommendation: {tag.replace('_', ' ')}."
            ),
        })

    return issues


# ---------------------------------------------------------------------------
# Tool 6 — check_format_consistency
# ---------------------------------------------------------------------------

@tool
def check_format_consistency(
    df: pd.DataFrame, column: str, expected_patterns: list = None
) -> list[dict]:
    """
    Analyse format distribution of a column and flag values that do not match any
    known pattern for that NoiPA column.

    Uses pre-configured primary and alternative patterns for NoiPA-specific columns
    (rata, RATA, mese, anno, spesa, aggregation-time).
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty or column not in df.columns:
        return issues

    series = df[column].dropna().astype(str).str.strip()
    if series.empty:
        return issues

    # Build pattern list
    cfg = _FORMAT_PATTERNS.get(column, {})
    primary_pattern = cfg.get("primary") if cfg else None
    alt_patterns: list[str] = cfg.get("alternatives", []) if cfg else []

    if expected_patterns:
        # Caller override: use as primary + any cfg alternatives
        all_patterns = expected_patterns + alt_patterns
        primary_pattern = expected_patterns[0]
    else:
        all_patterns = ([primary_pattern] if primary_pattern else []) + alt_patterns

    if not all_patterns:
        # No patterns: just report unique value counts as info
        top_vals = series.value_counts().head(10).to_dict()
        issues.append({
            "column": column,
            "row": None,
            "issue": "no_pattern_configured",
            "severity": "info",
            "details": f"No format pattern configured for '{column}'. Top values: {top_vals}.",
        })
        return issues

    # Classify each value
    distribution: dict[str, int] = {}
    non_matching: list[int] = []

    for idx, val in zip(df[column].dropna().index, series):
        matched = False
        for pat in all_patterns:
            if re.match(pat, val, re.IGNORECASE):
                distribution[pat] = distribution.get(pat, 0) + 1
                matched = True
                break
        if not matched:
            distribution["__other__"] = distribution.get("__other__", 0) + 1
            non_matching.append(int(idx))

    total_non_null = len(series)
    other_count = distribution.get("__other__", 0)
    other_rate = other_count / total_non_null if total_non_null else 0.0

    # Distribution summary as info
    issues.append({
        "column": column,
        "row": None,
        "issue": "format_distribution",
        "severity": "info",
        "details": (
            f"Column '{column}' format distribution over {total_non_null} non-null rows: "
            + str({k: v for k, v in distribution.items() if k != "__other__"})
            + f". Unmatched: {other_count} ({other_rate:.1%})."
        ),
    })

    if non_matching:
        severity = "critical" if other_rate > 0.05 else ("warning" if other_rate > 0.01 else "info")
        sample = non_matching[:5]
        issues.append({
            "column": column,
            "row": sample[0] if len(sample) == 1 else None,
            "issue": "format_mismatch",
            "severity": severity,
            "details": (
                f"Column '{column}': {other_count}/{total_non_null} values "
                f"({other_rate:.1%}) do not match any known pattern. "
                f"Sample row indices: {sample}."
            ),
        })

    # Special range check for mese
    if column == "mese" and primary_pattern:
        numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
        out_of_range = numeric_vals[(numeric_vals < 1) | (numeric_vals > 12)]
        if not out_of_range.empty:
            bad_idx = [int(i) for i in out_of_range.index[:5]]
            issues.append({
                "column": column,
                "row": bad_idx[0] if len(bad_idx) == 1 else None,
                "issue": "mese_out_of_range",
                "severity": "critical",
                "details": (
                    f"Column 'mese' has {len(out_of_range)} value(s) outside 1-12. "
                    f"Sample indices: {bad_idx}."
                ),
            })

    return issues


# ---------------------------------------------------------------------------
# Tool 7 — check_cross_column_consistency
# ---------------------------------------------------------------------------

@tool
def check_cross_column_consistency(
    df: pd.DataFrame, column_pairs: list = None
) -> list[dict]:
    """
    Detect pairs of columns that appear to encode the same information but contain
    inconsistent or divergent values in a NoiPA DataFrame.

    Auto-detects which dataset (spesa.csv vs attivazioniCessazioni.csv) by inspecting
    column names, then checks concordance between each pair.
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    if column_pairs is None:
        # Auto-detect dataset type from columns present
        cols_set = set(df.columns)
        spesa_score = sum(1 for c, _ in _SPESA_CROSS_PAIRS if c in cols_set or _ in cols_set)
        att_score = sum(1 for c, _ in _ATT_CESS_CROSS_PAIRS if c in cols_set or _ in cols_set)
        column_pairs = _SPESA_CROSS_PAIRS if spesa_score >= att_score else _ATT_CESS_CROSS_PAIRS

    for col_a, col_b in column_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue

        a = df[col_a].astype(str).str.strip().str.lower()
        b = df[col_b].astype(str).str.strip().str.lower()

        concordant = (a == b).sum()
        discordant = (a != b).sum()
        total = len(df)
        discord_rate = discordant / total if total > 0 else 0.0

        # Recommend more reliable column
        null_a = df[col_a].isna().sum()
        null_b = df[col_b].isna().sum()
        recommended = col_a if null_a <= null_b else col_b

        if discordant > 0:
            sample_idx = df[a != b].index[:5].tolist()
            issues.append({
                "column": f"{col_a} vs {col_b}",
                "row": None,
                "issue": "cross_column_discordance",
                "severity": "critical",
                "details": (
                    f"Columns '{col_a}' and '{col_b}' are semantically equivalent but "
                    f"diverge in {discordant}/{total} rows ({discord_rate:.1%}). "
                    f"Concordant: {concordant}. "
                    f"Recommended column: '{recommended}' ({min(null_a, null_b)} nulls). "
                    f"Sample discordant indices: {sample_idx}."
                ),
            })

    return issues


# ---------------------------------------------------------------------------
# Tool 8 — check_cross_column_logic
# ---------------------------------------------------------------------------

@tool
def check_cross_column_logic(
    df: pd.DataFrame, rules: list = None
) -> list[dict]:
    """
    Validate logical relationships between columns in a NoiPA DataFrame.

    Auto-detects applicable rules from column names:
    - RATA consistent with mese + anno (YYYYMM format)
    - cod_tipoimposta / tipo_imposta near-1:1 mapping
    - spesa / attivazioni / cessazioni non-negative
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    cols = set(df.columns)

    # Rule 1: RATA == YYYYMM consistent with mese + anno
    if {"RATA", "mese", "anno"}.issubset(cols) or {"rata", "mese", "anno"}.issubset(cols):
        rata_col = "RATA" if "RATA" in cols else "rata"
        sub = df[[rata_col, "mese", "anno"]].dropna()
        if not sub.empty:
            expected_rata = (
                sub["anno"].astype(str).str.zfill(4)
                + sub["mese"].astype(str).str.zfill(2)
            )
            actual_rata = sub[rata_col].astype(str).str.strip()
            mismatch = actual_rata != expected_rata
            mismatch_count = mismatch.sum()
            if mismatch_count > 0:
                sample_idx = sub[mismatch].index[:5].tolist()
                issues.append({
                    "column": rata_col,
                    "row": None,
                    "issue": "rata_mese_anno_inconsistency",
                    "severity": "critical",
                    "details": (
                        f"Column '{rata_col}' value does not match YYYYMM(mese+anno) "
                        f"in {mismatch_count} rows. "
                        f"Sample indices: {sample_idx}."
                    ),
                })

    # Rule 2: cod_tipoimposta / tipo_imposta near-1:1 mapping
    for pair in [("cod_tipoimposta", "tipo_imposta"), ("cod_imposta", "tipo_imposta")]:
        if all(c in cols for c in pair):
            sub = df[list(pair)].dropna()
            if not sub.empty:
                mapping = sub.groupby(pair[0])[pair[1]].nunique()
                multi = mapping[mapping > 1]
                if not multi.empty:
                    issues.append({
                        "column": f"{pair[0]} -> {pair[1]}",
                        "row": None,
                        "issue": "code_description_not_1to1",
                        "severity": "critical",
                        "details": (
                            f"{len(multi)} value(s) of '{pair[0]}' map to multiple "
                            f"'{pair[1]}' descriptions: {multi.head(5).to_dict()}."
                        ),
                    })

    # Rule 3: numeric columns must be >= 0
    for num_col in ["spesa", "attivazioni", "cessazioni"]:
        if num_col not in cols:
            continue
        numeric = _clean_numeric(df[num_col])
        negative = numeric[numeric < 0]
        if not negative.empty:
            sample_idx = negative.index[:5].tolist()
            issues.append({
                "column": num_col,
                "row": None,
                "issue": "negative_value",
                "severity": "critical",
                "details": (
                    f"Column '{num_col}' has {len(negative)} negative value(s). "
                    f"Sample indices: {sample_idx}."
                ),
            })

    return issues


# ---------------------------------------------------------------------------
# Tool 9 — detect_duplicates
# ---------------------------------------------------------------------------

@tool
def detect_duplicates(
    df: pd.DataFrame, key_columns: list = None, fuzzy: bool = False
) -> list[dict]:
    """
    Detect exact (and optionally fuzzy) duplicate rows in a NoiPA DataFrame.

    If key_columns is None all columns are used. Fuzzy duplicates are detected by
    comparing normalised string representations. Severity is 'warning' for exact
    duplicates and 'info' for fuzzy ones.
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty:
        return issues

    subset = key_columns if key_columns else None

    # Exact duplicates
    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_count = dup_mask.sum()

    if dup_count > 0:
        dup_indices = df[dup_mask].index[:10].tolist()
        issues.append({
            "column": ", ".join(map(str, key_columns)) if key_columns else "all_columns",
            "row": None,
            "issue": "exact_duplicates",
            "severity": "warning",
            "details": (
                f"{dup_count} rows are exact duplicates "
                f"({'on key columns: ' + str(key_columns) if key_columns else 'across all columns'}). "
                f"Sample indices: {dup_indices}."
            ),
        })

    # Fuzzy duplicates (normalised string comparison)
    if fuzzy:
        try:
            norm_df = df.astype(str).apply(
                lambda col: col.str.strip().str.lower()
            )
            fuzzy_dup_mask = norm_df.duplicated(subset=subset, keep=False)
            # Subtract exact duplicates
            fuzzy_only = fuzzy_dup_mask & ~dup_mask
            fuzzy_count = fuzzy_only.sum()
            if fuzzy_count > 0:
                fuzzy_indices = df[fuzzy_only].index[:10].tolist()
                issues.append({
                    "column": ", ".join(map(str, key_columns)) if key_columns else "all_columns",
                    "row": None,
                    "issue": "fuzzy_duplicates",
                    "severity": "info",
                    "details": (
                        f"{fuzzy_count} rows are fuzzy duplicates "
                        "(identical after lowercasing and stripping whitespace). "
                        f"Sample indices: {fuzzy_indices}."
                    ),
                })
        except Exception as e:
            issues.append({
                "column": "all_columns",
                "row": None,
                "issue": "fuzzy_duplicate_check_error",
                "severity": "info",
                "details": f"Fuzzy duplicate check failed: {e}",
            })

    return issues


# ---------------------------------------------------------------------------
# Tool 10 — detect_outliers
# ---------------------------------------------------------------------------

@tool
def detect_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr"
) -> list[dict]:
    """
    Detect statistical outliers and sentinel values in a numeric NoiPA column.

    Supports 'iqr' (Tukey fence) and 'zscore' methods. Sentinel values
    (999999999.99, -999999.5) are flagged as critical; other outliers as warning.
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty or column not in df.columns:
        return issues

    numeric = _clean_numeric(df[column])
    valid = numeric.dropna()

    if valid.empty:
        issues.append({
            "column": column,
            "row": None,
            "issue": "non_numeric_column",
            "severity": "info",
            "details": f"Column '{column}' could not be converted to numeric.",
        })
        return issues

    # Sentinel detection
    sentinels = {999999999.99: "max_sentinel", -999999.5: "min_sentinel"}
    for sentinel_val, sentinel_name in sentinels.items():
        sentinel_mask = np.isclose(valid, sentinel_val, atol=0.01)
        count = sentinel_mask.sum()
        if count > 0:
            sample_idx = valid[sentinel_mask].index[:5].tolist()
            issues.append({
                "column": column,
                "row": sample_idx[0] if count == 1 else None,
                "issue": f"sentinel_value_{sentinel_name}",
                "severity": "critical",
                "details": (
                    f"Column '{column}' contains {count} sentinel value(s) "
                    f"({sentinel_val}). Sample indices: {sample_idx}."
                ),
            })

    # Statistical outlier detection
    if method == "iqr":
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (valid < lower) | (valid > upper)
        method_detail = f"IQR method: [{lower:.4f}, {upper:.4f}]"
    elif method == "zscore":
        mean = valid.mean()
        std = valid.std()
        if std == 0:
            return issues
        z = (valid - mean) / std
        outlier_mask = z.abs() > 3
        method_detail = f"Z-score method: |z| > 3 (mean={mean:.4f}, std={std:.4f})"
    else:
        issues.append({
            "column": column,
            "row": None,
            "issue": "unknown_outlier_method",
            "severity": "info",
            "details": f"Unknown method '{method}'. Use 'iqr' or 'zscore'.",
        })
        return issues

    outlier_count = int(outlier_mask.sum())
    if outlier_count > 0:
        sample_idx = valid[outlier_mask].index[:5].tolist()
        sample_vals = valid[outlier_mask].head(5).tolist()
        issues.append({
            "column": column,
            "row": sample_idx[0] if outlier_count == 1 else None,
            "issue": "statistical_outlier",
            "severity": "warning",
            "details": (
                f"Column '{column}': {outlier_count} outlier(s) detected "
                f"using {method_detail}. "
                f"Sample indices: {sample_idx}, values: {[round(v, 2) for v in sample_vals]}."
            ),
        })

    return issues


# ---------------------------------------------------------------------------
# Tool 11 — detect_categorical_anomalies
# ---------------------------------------------------------------------------

@tool
def detect_categorical_anomalies(
    df: pd.DataFrame, column: str, min_frequency: float = 0.01
) -> list[dict]:
    """
    Identify rare categorical values and whitespace/case variants in a NoiPA column.

    Values with frequency below min_frequency (default 1%) of total rows are flagged.
    Trailing-space and case-variant duplicates are reported as a separate issue.
    Severity is 'info'.
    Returns a list of issue dicts with keys: column, row, issue, severity, details.
    """
    issues: list[dict] = []

    if df is None or df.empty or column not in df.columns:
        return issues

    series = df[column].dropna()
    if series.empty:
        return issues

    total = len(df)
    value_counts = series.value_counts()

    # Rare values
    rare = value_counts[value_counts / total < min_frequency]
    if not rare.empty:
        rare_dict = rare.head(10).to_dict()
        issues.append({
            "column": column,
            "row": None,
            "issue": "rare_categorical_values",
            "severity": "info",
            "details": (
                f"Column '{column}' has {len(rare)} rare value(s) "
                f"(frequency < {min_frequency:.1%}). "
                f"Sample (value: count): {rare_dict}."
            ),
        })

    # Trailing spaces and case variants
    # Normalise: strip + lower; group original values by normalised key
    norm_groups: dict[str, list[str]] = {}
    for val in value_counts.index:
        norm = str(val).strip().lower()
        norm_groups.setdefault(norm, []).append(str(val))

    case_variants = {k: v for k, v in norm_groups.items() if len(v) > 1}
    if case_variants:
        issues.append({
            "column": column,
            "row": None,
            "issue": "case_or_whitespace_variants",
            "severity": "info",
            "details": (
                f"Column '{column}' has {len(case_variants)} group(s) of values "
                "that differ only by case or trailing/leading whitespace. "
                f"Sample: { {k: v for k, v in list(case_variants.items())[:5]} }."
            ),
        })

    return issues


# ---------------------------------------------------------------------------
# Helper — calculate_reliability_score
# ---------------------------------------------------------------------------

def calculate_reliability_score(
    all_issues: list[dict], df: pd.DataFrame
) -> dict:
    """
    Compute a multi-dimensional reliability score from a collected list of issues.

    Dimensions and weights:
        schema       : 0.15  (naming + type issues)
        completeness : 0.30  (null/placeholder rates, severity-weighted)
        consistency  : 0.35  (format + cross-column issues)
        anomaly      : 0.20  (outlier + categorical issues)

    Returns a dict with keys: schema, completeness, consistency, anomaly, overall.
    All scores are in [0.0, 1.0].
    """
    if df is None or df.empty:
        return {
            "schema": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "anomaly": 0.0,
            "overall": 0.0,
        }

    total_cols = max(len(df.columns), 1)
    total_rows = max(len(df), 1)

    def _count_issues(issue_types: set[str], severities: set[str] | None = None) -> int:
        return sum(
            1 for iss in all_issues
            if iss.get("issue", "").split("_")[0] in issue_types
            or any(t in iss.get("issue", "") for t in issue_types)
            and (severities is None or iss.get("severity") in severities)
        )

    # ---- Schema score ----
    naming_issues = sum(
        1 for iss in all_issues
        if iss.get("issue") in {
            "special_char_in_column_name",
            "column_name_starts_with_digit",
            "not_snake_case",
            "semantically_duplicate_columns",
        }
    )
    type_issues = sum(
        1 for iss in all_issues if iss.get("issue") == "type_mismatch"
    )
    s_schema = max(
        0.0,
        1.0 - (naming_issues / total_cols) - 0.1 * (type_issues / total_cols),
    )

    # ---- Completeness score ----
    severity_weight = {"critical": 3, "warning": 2, "info": 1}
    completeness_issues = [
        iss for iss in all_issues
        if iss.get("issue") in {"null_values", "placeholder_values"}
    ]
    if completeness_issues:
        weighted_deductions = sum(
            severity_weight.get(iss.get("severity", "info"), 1)
            for iss in completeness_issues
        )
        max_weight = total_cols * 3  # worst case all critical
        s_completeness = max(0.0, 1.0 - weighted_deductions / max_weight)
    else:
        s_completeness = 1.0

    # ---- Consistency score ----
    consistency_issues = sum(
        1 for iss in all_issues
        if iss.get("issue") in {
            "format_mismatch",
            "cross_column_discordance",
            "rata_mese_anno_inconsistency",
            "code_description_not_1to1",
            "mese_out_of_range",
        }
    )
    tested_cols = max(
        len({iss.get("column", "") for iss in all_issues if "format" in iss.get("issue", "")}),
        1,
    )
    s_consistency = max(
        0.0,
        1.0 - consistency_issues / (total_rows * tested_cols),
    )

    # ---- Anomaly score ----
    critical_outliers = sum(
        1 for iss in all_issues
        if "sentinel" in iss.get("issue", "") or (
            "outlier" in iss.get("issue", "") and iss.get("severity") == "critical"
        )
    )
    categorical_anomalies = sum(
        1 for iss in all_issues
        if iss.get("issue") in {"rare_categorical_values", "case_or_whitespace_variants"}
    )
    s_anomaly = max(
        0.0,
        1.0 - (critical_outliers + categorical_anomalies) / total_rows,
    )

    # ---- Overall weighted score ----
    overall = (
        0.15 * s_schema
        + 0.30 * s_completeness
        + 0.35 * s_consistency
        + 0.20 * s_anomaly
    )

    return {
        "schema": round(s_schema, 4),
        "completeness": round(s_completeness, 4),
        "consistency": round(s_consistency, 4),
        "anomaly": round(s_anomaly, 4),
        "overall": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== tools.py smoke test ===\n")

    # Build a small synthetic DataFrame with known issues
    data = {
        "codice_ente": ["ENT001", "ENT002", "ENT003", "ENT001", None],
        "ente": ["Ministero A", "Ministero B", "Ministero C", "Ministero A", "N.D."],
        "anno": [2023, 2023, 2023, 2023, 2023],
        "mese": [1, 2, 3, 13, 5],         # row 3: mese=13 out of range
        "RATA": ["202301", "202302", "202303", "202313", "202305"],  # row 3: mese mismatch
        "spesa": [1000.0, 2500.50, -50.0, 999999999.99, 3200.0],    # negative + sentinel
        "attivazioni": [10, 20, 30, 40, 50],
        "cessazioni": [5, 10, 15, 20, 25],
        "tipo_imposta": ["IRPEF", "IRAP", "IRPEF", "IRAP", "IRPEF"],
        "Tipo Imposta": ["IRPEF", "IRAP", "irpef", "IRAP", "IRPEF"],  # case divergence
        "2cod_imposta": ["C1", "C2", "C1", "C2", "C1"],
        "provincia_sede": ["Roma", "Milano", "Napoli", None, "Torino"],
        "note": ["ok", None, None, None, None],
    }
    df_test = pd.DataFrame(data)

    all_issues: list[dict] = []

    print("--- check_naming_convention ---")
    r = check_naming_convention.invoke({"df": df_test})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- check_data_types ---")
    r = check_data_types.invoke({"df": df_test})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- detect_null_and_placeholders ---")
    r = detect_null_and_placeholders.invoke({"df": df_test})
    for item in r:
        print(f"  [{item['severity']}] {item['column']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- calculate_completeness ---")
    comp = calculate_completeness.invoke({"df": df_test})
    print(f"  Overall completeness: {comp['overall_completeness']:.2%}")
    for col_stat in comp["columns"]:
        if col_stat["completeness_rate"] < 1.0:
            print(f"    {col_stat['column']}: {col_stat['completeness_rate']:.2%} complete")

    print("\n--- detect_sparse_columns ---")
    r = detect_sparse_columns.invoke({"df": df_test})
    for item in r:
        print(f"  [{item['severity']}] {item['column']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- check_format_consistency (RATA) ---")
    r = check_format_consistency.invoke({"df": df_test, "column": "RATA"})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- check_format_consistency (mese) ---")
    r = check_format_consistency.invoke({"df": df_test, "column": "mese"})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- check_cross_column_consistency ---")
    r = check_cross_column_consistency.invoke({"df": df_test})
    for item in r:
        print(f"  [{item['severity']}] {item['column']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- check_cross_column_logic ---")
    r = check_cross_column_logic.invoke({"df": df_test})
    for item in r:
        print(f"  [{item['severity']}] {item['column']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- detect_duplicates ---")
    r = detect_duplicates.invoke({"df": df_test, "fuzzy": True})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- detect_outliers (spesa, iqr) ---")
    r = detect_outliers.invoke({"df": df_test, "column": "spesa", "method": "iqr"})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- detect_categorical_anomalies (tipo_imposta) ---")
    r = detect_categorical_anomalies.invoke({"df": df_test, "column": "tipo_imposta"})
    for item in r:
        print(f"  [{item['severity']}] {item['issue']}: {item['details'][:80]}")
    all_issues.extend(r)

    print("\n--- calculate_reliability_score ---")
    score = calculate_reliability_score(all_issues, df_test)
    for k, v in score.items():
        print(f"  {k:>15}: {v:.4f}")

    print("\n=== Smoke test complete ===")

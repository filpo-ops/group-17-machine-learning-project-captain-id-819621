"""Multi-agent data quality pipeline — runtime module.

Public API consumed by the webapp (webapp/server.py) and as a CLI smoke test:
    - run_quality_pipeline(df, name, with_narrative=False) -> (state, html)
    - stream_quality_pipeline(df, name) -> Iterator[(node_name, state_snapshot)]
    - render_quality_report(state) -> str
    - quality_graph: compiled LangGraph
    - RELIABILITY_WEIGHTS: dict
    - verify_llm() -> str   # optional API smoke test (1 small call)

Extracted from agents/Main.ipynb. The notebook is the single source of truth
for design, narrative, and pedagogical structure; this module is the runtime
API used by the webapp (FastAPI backend) without booting the notebook.
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import json
import re
import hashlib
import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated, Dict, Any, Iterator, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from jinja2 import Template


# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    raise RuntimeError(
        f"DEEPSEEK_API_KEY not found in {ENV_PATH}. "
        "Add `DEEPSEEK_API_KEY=your_key` to the .env file at the project root."
    )

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# LLM client is initialized at import time but NO API call is made here.
# This keeps import fast and avoids burning a token on every webapp boot.
# Call verify_llm() explicitly when you want to smoke-test the connection.
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com/v1",
    timeout=60,
)

_provider = f"{DEEPSEEK_MODEL} (DeepSeek)"

# In-memory registry of datasets keyed by dataset_name. The notebook populates
# this at cell 3 with the 4 NoiPA fixtures; in module mode it starts empty and
# `run_quality_pipeline` / `stream_quality_pipeline` insert entries as they
# receive dataframes. node_ingest falls back to this registry when `raw_df` is
# absent from state.
datasets: Dict[str, pd.DataFrame] = {}


def verify_llm() -> str:
    """Smoke-test the LLM connection with a 1-token request. Returns provider name on success.

    Raises RuntimeError if the response doesn't contain "ok" (case-insensitive).
    Costs ~10 tokens. Skip this on every-import paths (see module docstring).
    """
    response = llm.invoke("Reply with exactly one word: OK")
    if "ok" not in response.content.lower():
        raise RuntimeError(f"LLM verification failed: unexpected response {response.content!r}")
    return _provider


# ─── Extracted from notebook cell 13 ────────────────────────────────────
from collections import defaultdict

DISGUISED_NULLS = {
    "n.d.", "nd", "n/a", "null", "none", "?", "-", "//",
    "unknown", "na", "missing", "", " "
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}

def normalize_text(x):
    """Strips surrounding whitespace and returns None for empty strings."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x != "" else None

def normalize_token(x):
    """Lower-cases a value after normalizing it; used when comparing tokens case-insensitively."""
    x = normalize_text(x)
    return x.lower() if x is not None else None

def is_disguised_null(x):
    """Returns True when a value is semantically empty (placeholder string) even if pandas sees it as non-null."""
    token = normalize_token(x)
    return token in DISGUISED_NULLS if token is not None else False

def safe_samples(series, n=5):
    """Pulls up to n distinct non-null string samples from a series for use as evidence in issue reports."""
    s = series.dropna().astype(str).str.strip()
    s = s[s.ne("")]
    return s.drop_duplicates().head(n).tolist()

def coerce_numeric_loose(series):
    """Attempts a best-effort numeric parse by stripping currency symbols, spaces,
    thousand-dot separators, and comma decimals before calling pd.to_numeric.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
    s = s.str.replace("€", "", regex=False)
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.replace({"": np.nan, "-": np.nan, ".": np.nan, "-.": np.nan})
    return pd.to_numeric(s, errors="coerce")

def infer_semantic_type(series, threshold=0.90):
    """Guesses whether a column is "numeric", "date", or "string" based on
    what fraction of present values can be parsed as each type.
    """
    s = series.copy()
    mask_present = s.notna() & s.astype(str).str.strip().ne("")
    s = s[mask_present]
    if len(s) == 0:
        return "unknown"
    numeric_rate = coerce_numeric_loose(s).notna().mean()
    date_rate = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", dayfirst=True).notna().mean()
    if numeric_rate >= threshold:
        return "numeric"
    if date_rate >= threshold:
        return "date"
    return "string"

def make_issue(dataset, tool, issue_type, severity, message,
               columns=None, row_count=None, evidence=None, suggested_fix=None):
    """Builds a single issue dict with a consistent schema so all tools speak the same language
    and LangGraph agents can parse results without any special-casing.
    """
    return {
        "dataset": dataset,
        "tool": tool,
        "issue_type": issue_type,
        "severity": severity,
        "columns": columns if columns is not None else [],
        "row_count": int(row_count) if row_count is not None else None,
        "message": message,
        "evidence": evidence if evidence is not None else {},
        "suggested_fix": suggested_fix
    }

def build_result(dataset, tool, issues, meta=None):
    """Wraps a list of issues from one tool into a standard result envelope that includes
    an issue count, a per-severity breakdown, and optional metadata.
    """
    severity_breakdown = defaultdict(int)
    for issue in issues:
        severity_breakdown[issue["severity"]] += 1
    return {
        "dataset": dataset,
        "tool": tool,
        "issue_count": len(issues),
        "severity_breakdown": dict(severity_breakdown),
        "issues": issues,
        "meta": meta or {}
    }

# ─── Extracted from notebook cell 16 ────────────────────────────────────
def infer_temporal_kind(series, min_share=0.85):
    """Infer the temporal kind (year/month/period/date/datetime) of a Series; require `min_share` of values to fit the kind."""
    # Try to understand which temporal family best describes a column:
    # year, month, period, date, or datetime.
    s = series.dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if len(s) == 0:
        return None

    # Regex-based match rates for the temporal formats already used later
    # by the format validation layer.
    checks = {
        "year":   s.str.fullmatch(r"(19|20)\d{2}", na=False).mean(),
        "month":  s.str.fullmatch(r"(0?[1-9]|1[0-2])", na=False).mean(),
        "period": s.str.fullmatch(r"\d{4}-\d{2}|\d{2}/\d{4}|\d{4}/\d{2}", na=False).mean(),
        "date":   s.str.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", na=False).mean(),
    }

    # Datetime detection: parseable dates + visible time tokens.
    parsed_dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    datetime_like = parsed_dt.notna().mean()
    time_tokens = s.str.contains(r":|T", regex=True, na=False).mean()

    if datetime_like >= min_share and time_tokens >= 0.20:
        return "datetime"

    # Pick the strongest regex-based temporal type if it is reliable enough.
    best_kind, best_score = max(checks.items(), key=lambda kv: kv[1])
    if best_score >= min_share:
        return best_kind

    # Fallback: if many values are parseable as dates, classify as generic date.
    if datetime_like >= min_share:
        return "date"

    return None


def profile_column(series):
    """Build a compact profile of a column: semantic type, completeness, uniqueness, numeric/temporal parseability."""
    # Build a compact profile for one column so discovery can infer
    # better mandatory/format/numeric/key rules.
    total = len(series)
    raw = series.astype(str).str.strip()

    # Separate real pandas nulls from disguised null tokens like "N.D." or "?".
    real_nulls = int(series.isna().sum())
    disguised_nulls = int(raw.str.lower().isin(DISGUISED_NULLS).sum())
    effective_missing = real_nulls + disguised_nulls
    completeness = 1 - (effective_missing / total if total else 0)

    # Keep only semantically present values for profiling.
    present_mask = series.notna() & raw.ne("") & ~raw.str.lower().isin(DISGUISED_NULLS)
    present = series[present_mask]

    # Broad semantic type from the existing helper.
    semantic_type = infer_semantic_type(series)

    # If a column looks temporal, refine it into year/month/period/date/datetime.
    temporal_kind = infer_temporal_kind(series) if semantic_type == "date" else None

    # Numeric parseability profile.
    numeric = coerce_numeric_loose(series)
    numeric_present = numeric[present_mask]
    numeric_nonnull = numeric_present.dropna()

    numeric_parse_rate = float(numeric_present.notna().mean()) if len(numeric_present) else 0.0
    unique_ratio = float(present.nunique(dropna=True) / len(present)) if len(present) else 0.0
    sample_values = safe_samples(series, n=5)

    # Detect suspicious symbols/text that suggest a numeric column is dirty.
    forbid_tokens = []
    token_patterns = {
        "€": r"€",
        "$": r"\$",
        "%": r"%",
        "unit_text": r"[A-Za-z]{2,}"
    }
    for token_name, pattern in token_patterns.items():
        if raw.str.contains(pattern, regex=True, na=False).any():
            forbid_tokens.append(token_name)

    # Heuristic minimum rule:
    # if almost all observed numeric values are non-negative, use min=0.
    min_value = None
    if semantic_type == "numeric" and len(numeric_nonnull):
        if (numeric_nonnull >= 0).mean() >= 0.98:
            min_value = 0

    return {
        "semantic_type": semantic_type,
        "temporal_kind": temporal_kind,
        "completeness": round(float(completeness), 4),
        "real_nulls": real_nulls,
        "disguised_nulls": disguised_nulls,
        "effective_missing": int(effective_missing),
        "unique_ratio": round(unique_ratio, 4),
        "nunique_present": int(present.nunique(dropna=True)) if len(present) else 0,
        "present_count": int(len(present)),
        "numeric_parse_rate": round(numeric_parse_rate, 4),
        "samples": sample_values,
        "forbid_tokens": forbid_tokens,
        "min_value": min_value,
    }

# ─── Extracted from notebook cell 17 ────────────────────────────────────
# ─── Generic schema discovery (no hardcoded per-dataset rules) ──────────────
# We DO NOT hardcode schemas for specific datasets. Rules are discovered from
# the dataframe sample at runtime, then cached per dataset_name.

EXPECTED_SCHEMAS = {}        # populated by discover_dataset_rules
SCHEMA_NAMING_RULES = {"snake_case_preferred": True}

def discover_dataset_rules(df, name, sample_size=10000, random_state=42):
    """
    Heuristically discover schema, completeness, format, numeric,
    and duplicate-key rules from the dataframe itself.

    This is still inference, not a persisted schema contract.
    The goal is to bootstrap the validation layer for arbitrary CSVs
    without hardcoding dataset-specific rules.
    """
    if df is None or df.empty:
        raise ValueError("discover_dataset_rules() received an empty dataframe")

    # Work on a sample for speed on large datasets, but keep the full dataframe
    # available for final metadata like total row/column counts.
    sample = df.sample(min(sample_size, len(df)), random_state=random_state) \
        if len(df) > sample_size else df.copy()

    n_rows, n_cols = df.shape

    # Build a compact profile for each column.
    # This gives us richer signals than only semantic type:
    # completeness, uniqueness, numeric parseability, temporal subtype, etc.
    column_profiles = {col: profile_column(sample[col]) for col in df.columns}

    # Expected semantic type per column: numeric, date, string, or unknown.
    expected_types = {col: p["semantic_type"] for col, p in column_profiles.items()}

    # Split columns into required vs optional using simple heuristics.
    # A column is considered required when it is mostly populated
    # and carries at least some informational value.
    required_columns = []
    optional_columns = []

    for col, p in column_profiles.items():
        informative = p["present_count"] > 0 and p["nunique_present"] > 1
        if p["completeness"] >= 0.80 and informative:
            required_columns.append(col)
        else:
            optional_columns.append(col)

    # Discover temporal validation rules.
    # Unlike the old version, this can distinguish year/month/period/date/datetime.
    format_rules = {}
    for col, p in column_profiles.items():
        if p["semantic_type"] == "date" and p["temporal_kind"] is not None:
            format_rules[col] = p["temporal_kind"]

    # Discover numeric validation rules.
    # We keep them conservative:
    # - add min=0 only when the column looks non-negative almost everywhere
    # - add forbidden tokens when symbols like €, $, % appear in the raw text
    numeric_rules = {}
    for col, p in column_profiles.items():
        if p["semantic_type"] != "numeric":
            continue

        rule = {}

        if p["min_value"] is not None:
            rule["min"] = p["min_value"]

        forbid = []
        if "€" in p["forbid_tokens"]:
            forbid.append("€")
        if "$" in p["forbid_tokens"]:
            forbid.append("$")
        if "%" in p["forbid_tokens"]:
            forbid.append("%")

        if forbid:
            rule["forbid_tokens"] = forbid

        numeric_rules[col] = rule

    # Heuristically guess candidate business keys:
    # columns that are almost always present and almost always unique.
    # We keep only a few candidates because this is just a bootstrap guess.
    candidate_keys = []
    for col, p in column_profiles.items():
        if p["present_count"] == 0:
            continue
        if p["unique_ratio"] >= 0.98 and p["completeness"] >= 0.95:
            candidate_keys.append(col)

    # Fingerprint the discovered schema so it is easy to compare snapshots later
    # or debug why discovery changed between runs.
    schema_fingerprint = hashlib.md5(
        json.dumps(
            {
                "columns": list(df.columns),
                "expected_types": expected_types,
                "format_rules": format_rules,
                "numeric_rules": numeric_rules,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    # Store a richer schema object than before.
    # This keeps compatibility with the current pipeline while adding metadata
    # that can be useful later for drift checks or reporting.
    discovered = {
        "required_columns": required_columns,
        "optional_columns": optional_columns,
        "all_columns": list(df.columns),
        "expected_types": expected_types,
        "column_profiles": column_profiles,
        "schema_meta": {
            "dataset_name": name,
            "n_rows_profiled": int(len(sample)),
            "n_rows_total": int(n_rows),
            "n_columns": int(n_cols),
            "schema_fingerprint": schema_fingerprint,
            "discovery_mode": "heuristic_inference",
        },
    }

    # Populate the global registries used by the rest of Phase 3.
    EXPECTED_SCHEMAS[name] = discovered
    MANDATORY_COLUMNS[name] = required_columns
    FORMAT_RULES[name] = format_rules
    NUMERIC_RULES[name] = numeric_rules
    DUPLICATE_KEY_RULES[name] = candidate_keys[:3]

    # Return the discovered rule packs in the same spirit as the old function,
    # plus the candidate key suggestions.
    return {
        "schema": discovered,
        "mandatory": required_columns,
        "format_rules": format_rules,
        "numeric_rules": numeric_rules,
        "duplicate_key_rules": candidate_keys[:3],
    }

def check_schema(df, dataset_name, expected_schemas=None):
    """Validates a DataFrame's structure against the discovered/expected schema."""
    expected_schemas = expected_schemas if expected_schemas is not None else EXPECTED_SCHEMAS
    tool = "check_schema"
    issues = []
    expected = expected_schemas.get(dataset_name, {})

    required_columns = expected.get("required_columns", [])
    expected_types = expected.get("expected_types", {})
    actual_columns = list(df.columns)

    missing_required = [c for c in required_columns if c not in actual_columns]
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()

    if missing_required:
        issues.append(make_issue(
            dataset_name, tool, "missing_required_columns", "critical",
            f"Missing required columns: {missing_required}",
            columns=missing_required, row_count=len(df),
            evidence={"missing_required": missing_required},
            suggested_fix="Restore missing columns before downstream validation."
        ))

    if duplicate_columns:
        issues.append(make_issue(
            dataset_name, tool, "duplicate_column_names", "high",
            f"Duplicate column names detected: {duplicate_columns}",
            columns=duplicate_columns, row_count=len(df),
            evidence={"duplicate_columns": duplicate_columns},
            suggested_fix="Rename duplicate headers."
        ))

    actual_types = {col: infer_semantic_type(df[col]) for col in df.columns}
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = actual_types[col]
            if expected_type != "unknown" and actual_type != "unknown" and actual_type != expected_type:
                issues.append(make_issue(
                    dataset_name, tool, "semantic_type_mismatch", "high",
                    f"Column `{col}` looks like {actual_type}, expected {expected_type}.",
                    columns=[col], row_count=int(df[col].notna().sum()),
                    evidence={"expected_type": expected_type, "actual_type": actual_type,
                              "samples": safe_samples(df[col])},
                    suggested_fix="Standardize the column representation."
                ))

    for col in df.columns:
        flags = []
        if SCHEMA_NAMING_RULES["snake_case_preferred"]:
            if re.search(r"[\s%]", col):
                flags.append("special_char_or_space")
        if col[:1].isdigit():
            flags.append("starts_with_digit")
        if flags:
            issues.append(make_issue(
                dataset_name, tool, "naming_convention_violation", "low",
                f"Column `{col}` violates preferred naming conventions.",
                columns=[col], row_count=len(df),
                evidence={"violations": flags},
                suggested_fix="Standardize column naming."
            ))

    return build_result(dataset_name, tool, issues, meta={"column_count": len(df.columns)})

# ─── Extracted from notebook cell 19 ────────────────────────────────────
MANDATORY_COLUMNS = {}  # populated by discover_dataset_rules at runtime

def check_nulls(df, dataset_name, mandatory_columns=MANDATORY_COLUMNS):
    """Combines real nulls and disguised null tokens to compute effective missingness per column.
    Severity escalates to critical when mandatory columns exceed 20% effective missingness.
    """
    tool = "check_nulls"
    issues = []
    mandatory = mandatory_columns.get(dataset_name, [])

    for col in df.columns:
        total = len(df)
        real_nulls = int(df[col].isna().sum())
        disguised_nulls = int(df[col].astype(str).str.strip().str.lower().isin(DISGUISED_NULLS).sum())
        effective_missing = real_nulls + disguised_nulls
        missing_ratio = effective_missing / total if total else 0

        if effective_missing == 0:
            continue

        if col in mandatory:
            severity = "critical" if missing_ratio >= 0.20 else "high"
            issue_type = "missing_mandatory_values"
        else:
            severity = "medium" if missing_ratio >= 0.20 else "low"
            issue_type = "missing_optional_values"

        issues.append(make_issue(
            dataset_name, tool, issue_type, severity,
            f"Column `{col}` has {effective_missing} effectively missing values ({missing_ratio:.1%}).",
            columns=[col], row_count=effective_missing,
            evidence={"real_nulls": real_nulls, "disguised_nulls": disguised_nulls,
                      "missing_ratio": round(missing_ratio, 4), "samples": safe_samples(df[col])},
            suggested_fix="Impute, standardize, or drop depending on business criticality."
        ))

    return build_result(dataset_name, tool, issues)

def check_sparse_columns(df, dataset_name, threshold=0.85):
    """Flags any column whose completeness (share of non-missing values) falls below a
    configurable threshold; useful for quickly surfacing near-empty columns that
    the per-column null check might bury.
    """
    tool = "check_sparse_columns"
    issues = []

    for col in df.columns:
        total = len(df)
        real_nulls = int(df[col].isna().sum())
        disguised_nulls = int(df[col].astype(str).str.strip().str.lower().isin(DISGUISED_NULLS).sum())
        effective_missing = real_nulls + disguised_nulls
        completeness = 1 - (effective_missing / total if total else 0)

        if completeness < threshold:
            severity = "high" if completeness < 0.60 else "medium"
            issues.append(make_issue(
                dataset_name, tool, "sparse_column", severity,
                f"Column `{col}` completeness is only {completeness:.1%}.",
                columns=[col], row_count=effective_missing,
                evidence={"completeness": round(completeness, 4),
                          "real_nulls": real_nulls, "disguised_nulls": disguised_nulls},
                suggested_fix="Assess whether the column should be imputed, deprecated, or excluded."
            ))

    return build_result(dataset_name, tool, issues, meta={"threshold": threshold})

# ─── Extracted from notebook cell 21 ────────────────────────────────────
FORMAT_RULES = {}  # populated by discover_dataset_rules at runtime

def classify_format(value, kind):
    """Classifies a single cell value against the expected temporal kind
    (year, month, period, date, or datetime) and returns a label like
    "yyyy-mm-dd", "invalid", or "missing" to build a format distribution.
    """
    if pd.isna(value) or normalize_text(value) is None or is_disguised_null(value):
        return "missing"

    text = str(value).strip()

    if kind == "year":
        return "yyyy" if re.fullmatch(r"(19|20)\d{2}", text) else "invalid"

    if kind == "month":
        return "m_or_mm" if re.fullmatch(r"(0?[1-9]|1[0-2])", text) else "invalid"

    if kind == "period":
        if re.fullmatch(r"\d{4}-\d{2}", text): return "yyyy-mm"
        if re.fullmatch(r"\d{2}/\d{4}", text): return "mm/yyyy"
        if re.fullmatch(r"\d{4}/\d{2}", text): return "yyyy/mm"
        return "invalid"

    if kind == "date":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text): return "yyyy-mm-dd"
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", text): return "dd/mm/yyyy"
        if pd.to_datetime(pd.Series([text]), errors="coerce", dayfirst=True).notna().iloc[0]:
            return "other_parseable_date"
        return "invalid"

    if kind == "datetime":
        if pd.to_datetime(pd.Series([text]), errors="coerce", dayfirst=True).notna().iloc[0]:
            return "datetime_like" if ("T" in text or ":" in text) else "date_like"
        return "invalid"

    return "unknown"

def check_formats(df, dataset_name, format_rules=FORMAT_RULES):
    """Scans columns listed in FORMAT_RULES and raises an issue when invalid values
    exist or when multiple valid format families are mixed in the same column
    (e.g., some dates as "dd/mm/yyyy" and others as "yyyy-mm-dd").
    """
    tool = "check_formats"
    issues = []
    rules = format_rules.get(dataset_name, {})

    for col, kind in rules.items():
        if col not in df.columns:
            continue

        classified = df[col].apply(lambda x: classify_format(x, kind))
        non_missing = classified[classified != "missing"]

        if len(non_missing) == 0:
            continue

        invalid_count = int((non_missing == "invalid").sum())
        valid_share = 1 - (invalid_count / len(non_missing))
        format_mix = non_missing[non_missing != "invalid"].value_counts().to_dict()

        if invalid_count > 0:
            severity = "high" if valid_share < 0.80 else "medium"
            issues.append(make_issue(
                dataset_name, tool, "invalid_format_values", severity,
                f"Column `{col}` contains {invalid_count} invalid {kind} values.",
                columns=[col], row_count=invalid_count,
                evidence={"kind": kind, "valid_share": round(valid_share, 4),
                          "format_distribution": format_mix, "samples": safe_samples(df[col])},
                suggested_fix="Normalize this field to one accepted format before validation or joins."
            ))

        if len(format_mix) > 1:
            issues.append(make_issue(
                dataset_name, tool, "mixed_format_family", "medium",
                f"Column `{col}` mixes multiple valid format families: {list(format_mix.keys())}.",
                columns=[col], row_count=len(non_missing),
                evidence={"kind": kind, "format_distribution": format_mix},
                suggested_fix="Convert all values to a single canonical format."
            ))

    return build_result(dataset_name, tool, issues)

def check_categorical_case_variants(df, dataset_name, max_unique=40):
    """Looks for casing or spelling variants that map to the same lowercase token
    (e.g., "Roma", "ROMA", "roma") in low-cardinality string columns, which would
    inflate group counts in any downstream aggregation.
    """
    tool = "check_categorical_case_variants"
    issues = []
    object_cols = df.select_dtypes(include="object").columns.tolist()

    for col in object_cols:
        s = df[col].dropna().astype(str).str.strip()
        s = s[s.ne("")]
        if len(s) == 0 or s.nunique() > max_unique:
            continue

        grouped = defaultdict(set)
        for val in s.unique():
            grouped[val.lower()].add(val)

        suspicious = {k: sorted(list(v)) for k, v in grouped.items() if len(v) > 1}
        if suspicious:
            issues.append(make_issue(
                dataset_name, tool, "categorical_variant_inconsistency", "low",
                f"Column `{col}` contains casing or spelling variants that may represent the same category.",
                columns=[col],
                row_count=sum(len(v) for v in suspicious.values()),
                evidence={"variant_groups": suspicious},
                suggested_fix="Standardize category labels before aggregation or modeling."
            ))

    return build_result(dataset_name, tool, issues)

# ─── Extracted from notebook cell 23 ────────────────────────────────────
NUMERIC_RULES = {}  # populated by discover_dataset_rules at runtime

DUPLICATE_KEY_RULES = {}  # populated by discover_dataset_rules at runtime

def check_numeric_validity(df, dataset_name, numeric_rules=NUMERIC_RULES):
    """Validates numeric columns by detecting values that cannot be parsed to a number
    even after loose cleaning, forbidden tokens like "€", and values that violate
    a known minimum threshold (e.g., negative counts or costs).
    """
    tool = "check_numeric_validity"
    issues = []
    rules = numeric_rules.get(dataset_name, {})

    for col, rule in rules.items():
        if col not in df.columns:
            continue

        raw = df[col]
        raw_text = raw.astype(str).str.strip()
        parsed = coerce_numeric_loose(raw)
        present_mask = raw.notna() & raw_text.ne("") & ~raw_text.str.lower().isin(DISGUISED_NULLS)
        invalid_mask = present_mask & parsed.isna()

        if invalid_mask.any():
            issues.append(make_issue(
                dataset_name, tool, "non_numeric_values_in_numeric_field", "high",
                f"Column `{col}` contains non-numeric values in a numeric field.",
                columns=[col], row_count=int(invalid_mask.sum()),
                evidence={"samples": raw[invalid_mask].astype(str).drop_duplicates().head(10).tolist()},
                suggested_fix="Strip symbols/text and coerce to numeric using a canonical parser."
            ))

        for token in rule.get("forbid_tokens", []):
            token_mask = raw.astype(str).str.contains(re.escape(token), na=False)
            if token_mask.any():
                issues.append(make_issue(
                    dataset_name, tool, "forbidden_token_in_numeric_field", "medium",
                    f"Column `{col}` contains forbidden token `{token}`.",
                    columns=[col], row_count=int(token_mask.sum()),
                    evidence={"token": token, "samples": raw[token_mask].astype(str).drop_duplicates().head(10).tolist()},
                    suggested_fix="Store numeric values without currency/unit symbols."
                ))

        min_value = rule.get("min")
        if min_value is not None:
            negative_mask = parsed.notna() & (parsed < min_value)
            if negative_mask.any():
                issues.append(make_issue(
                    dataset_name, tool, "value_below_minimum", "high",
                    f"Column `{col}` contains values below the allowed minimum ({min_value}).",
                    columns=[col], row_count=int(negative_mask.sum()),
                    evidence={"min_allowed": min_value,
                              "samples": raw[negative_mask].astype(str).drop_duplicates().head(10).tolist()},
                    suggested_fix="Review source data or clamp/remove impossible values."
                ))

    return build_result(dataset_name, tool, issues)

def check_outliers_iqr(df, dataset_name, numeric_rules=NUMERIC_RULES, whisker=1.5):
    """Uses the Interquartile Range (IQR) method to flag statistical outliers in numeric
    columns. The whisker multiplier (default 1.5) controls sensitivity; only columns
    with enough data variety are tested to avoid false positives on near-constant fields.
    """
    tool = "check_outliers_iqr"
    issues = []
    candidate_cols = [c for c in numeric_rules.get(dataset_name, {}) if c in df.columns]

    for col in candidate_cols:
        vals = coerce_numeric_loose(df[col]).dropna()
        if len(vals) < 8 or vals.nunique() < 4:
            continue

        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower, upper = q1 - whisker * iqr, q3 + whisker * iqr
        outlier_mask = coerce_numeric_loose(df[col]).notna() & ~coerce_numeric_loose(df[col]).between(lower, upper)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            severity = "medium" if outlier_count / len(df) < 0.05 else "high"
            issues.append(make_issue(
                dataset_name, tool, "iqr_outliers", severity,
                f"Column `{col}` contains {outlier_count} IQR-based outliers.",
                columns=[col], row_count=outlier_count,
                evidence={"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                          "lower_bound": float(lower), "upper_bound": float(upper),
                          "samples": df.loc[outlier_mask, col].astype(str).drop_duplicates().head(10).tolist()},
                suggested_fix="Review whether these are valid spikes or ingestion errors."
            ))

    return build_result(dataset_name, tool, issues, meta={"whisker": whisker})

def check_duplicates(df, dataset_name, duplicate_key_rules=DUPLICATE_KEY_RULES):
    """Detects fully identical duplicate rows (exact copies) and, when business-key
    columns are configured, also finds key duplicates that carry divergent data —
    a more dangerous form of duplication that silently inflates aggregations.
    """
    tool = "check_duplicates"
    issues = []

    exact_dup_mask = df.duplicated(keep=False)
    if exact_dup_mask.any():
        issues.append(make_issue(
            dataset_name, tool, "exact_duplicate_rows", "medium",
            "Exact duplicate rows detected.",
            columns=list(df.columns), row_count=int(exact_dup_mask.sum()),
            evidence={"sample_rows": df[exact_dup_mask].head(5).to_dict(orient="records")},
            suggested_fix="Deduplicate exact copies before aggregation or scoring."
        ))

    key_cols = [c for c in duplicate_key_rules.get(dataset_name, []) if c in df.columns]
    if key_cols:
        key_dup_mask = df.duplicated(subset=key_cols, keep=False)
        if key_dup_mask.any():
            issues.append(make_issue(
                dataset_name, tool, "duplicate_business_keys", "high",
                f"Duplicate business keys detected on {key_cols}.",
                columns=key_cols, row_count=int(key_dup_mask.sum()),
                evidence={"sample_rows": df[key_dup_mask].head(5).to_dict(orient="records")},
                suggested_fix="Investigate whether repeated keys represent conflicts or valid snapshots."
            ))

    return build_result(dataset_name, tool, issues)

# ─── Extracted from notebook cell 25 ────────────────────────────────────
def parse_year(text):
    """Parses a string that should contain a 4-digit year; returns np.nan on failure."""
    if pd.isna(text) or normalize_text(text) is None or is_disguised_null(text):
        return np.nan
    text = str(text).strip()
    return int(text) if re.fullmatch(r"(19|20)\d{2}", text) else np.nan

def parse_year_month(text):
    """Parses a period string (yyyy-mm, mm/yyyy, yyyy/mm) into a (year, month) tuple;
    returns (nan, nan) when the format is unrecognized.
    """
    if pd.isna(text) or normalize_text(text) is None or is_disguised_null(text):
        return (np.nan, np.nan)
    text = str(text).strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        y, m = text.split("-"); return int(y), int(m)
    if re.fullmatch(r"\d{4}/\d{2}", text):
        y, m = text.split("/"); return int(y), int(m)
    if re.fullmatch(r"\d{2}/\d{4}", text):
        m, y = text.split("/"); return int(y), int(m)
    return (np.nan, np.nan)

def parse_date_year(text):
    """Attempts to parse a date string and extracts only the year component;
    returns np.nan when the string cannot be parsed as a date.
    """
    if pd.isna(text) or normalize_text(text) is None or is_disguised_null(text):
        return np.nan
    dt = pd.to_datetime(pd.Series([str(text).strip()]), errors="coerce", dayfirst=True).iloc[0]
    return dt.year if pd.notna(dt) else np.nan

def check_cross_column(df, dataset_name):
    """Enforces dataset-specific multi-column business rules:
    - tipologia:   INVESTIGATI must not exceed ENTRATI (logical count ceiling)
    - tipologia/allarmi: year extracted from DATA_PARTENZA must match ANNO_PARTENZA
    - attivazioni: mese and anno must be consistent with the RATA period string
    """
    tool = "check_cross_column"
    issues = []

    if dataset_name == "tipologia":
        if {"ENTRATI", "INVESTIGATI"}.issubset(df.columns):
            entrati = coerce_numeric_loose(df["ENTRATI"])
            investigati = coerce_numeric_loose(df["INVESTIGATI"])
            bad = investigati.notna() & entrati.notna() & (investigati > entrati)
            if bad.any():
                issues.append(make_issue(
                    dataset_name, tool, "logical_count_violation", "critical",
                    "`INVESTIGATI` cannot be greater than `ENTRATI`.",
                    columns=["INVESTIGATI", "ENTRATI"], row_count=int(bad.sum()),
                    evidence={"sample_rows": df.loc[bad, ["INVESTIGATI", "ENTRATI"]].head(10).to_dict(orient="records")},
                    suggested_fix="Review row-level counts and restore logical consistency."
                ))

    if dataset_name in {"tipologia", "allarmi"}:
        if {"DATA_PARTENZA", "ANNO_PARTENZA"}.issubset(df.columns):
            data_year = df["DATA_PARTENZA"].apply(parse_date_year)
            anno = df["ANNO_PARTENZA"].apply(parse_year)
            bad = data_year.notna() & anno.notna() & (data_year != anno)
            if bad.any():
                issues.append(make_issue(
                    dataset_name, tool, "year_date_mismatch", "high",
                    "`ANNO_PARTENZA` does not match the year extracted from `DATA_PARTENZA`.",
                    columns=["DATA_PARTENZA", "ANNO_PARTENZA"], row_count=int(bad.sum()),
                    evidence={"sample_rows": df.loc[bad, ["DATA_PARTENZA", "ANNO_PARTENZA"]].head(10).to_dict(orient="records")},
                    suggested_fix="Derive one field from the other or standardize both from source."
                ))

    if dataset_name == "attivazioni":
        if {"mese", "anno", "RATA"}.issubset(df.columns):
            ym_from_rata = df["RATA"].apply(parse_year_month)
            rata_year  = ym_from_rata.apply(lambda x: x[0])
            rata_month = ym_from_rata.apply(lambda x: x[1])
            anno = df["anno"].apply(parse_year)
            mese = pd.to_numeric(df["mese"], errors="coerce")
            bad = anno.notna() & mese.notna() & rata_year.notna() & rata_month.notna() & (
                (anno != rata_year) | (mese != rata_month)
            )
            if bad.any():
                issues.append(make_issue(
                    dataset_name, tool, "month_year_period_mismatch", "high",
                    "`mese` and `anno` are inconsistent with `RATA`.",
                    columns=["mese", "anno", "RATA"], row_count=int(bad.sum()),
                    evidence={"sample_rows": df.loc[bad, ["mese", "anno", "RATA"]].head(10).to_dict(orient="records")},
                    suggested_fix="Rebuild the period fields from one canonical temporal source."
                ))

    return build_result(dataset_name, tool, issues)

# ─── Extracted from notebook cell 27 ────────────────────────────────────
PHASE3_TOOLS = [
    check_schema,
    check_nulls,
    check_sparse_columns,
    check_formats,
    check_categorical_case_variants,
    check_numeric_validity,
    check_outliers_iqr,
    check_duplicates,
    check_cross_column
]

# Keyed by function name so LangGraph agents can call tools by string reference.
agent_tools = {tool.__name__: tool for tool in PHASE3_TOOLS}

def audit_dataset(df, dataset_name, auto_discover=True):
    """Runs every tool in PHASE3_TOOLS against a single DataFrame and collects their results."""
    if auto_discover and dataset_name not in EXPECTED_SCHEMAS:
        discover_dataset_rules(df, dataset_name)

    return {
        "dataset": dataset_name,
        "results": [tool(df, dataset_name) for tool in PHASE3_TOOLS]
    }

def flatten_issues(audit_output):
    """Explodes all nested issue dicts across all datasets into a single flat DataFrame
    for easy sorting, filtering, and display.
    """
    rows = []
    for dataset_name, payload in audit_output.items():
        for result in payload["results"]:
            rows.extend(result["issues"])
    return pd.DataFrame(rows)

def flatten_summary(audit_output):
    """Produces a one-row-per-tool summary table showing issue counts and
    severity breakdowns — useful as a quick health check before agent reasoning.
    """
    rows = []
    for dataset_name, payload in audit_output.items():
        for result in payload["results"]:
            rows.append({
                "dataset": dataset_name,
                "tool": result["tool"],
                "issue_count": result["issue_count"],
                "severity_breakdown": json.dumps(result["severity_breakdown"]),
            })
    return pd.DataFrame(rows)

print(f"Phase 3 ready. Tools: {[t.__name__ for t in PHASE3_TOOLS]}")

# ─── Extracted from notebook cell 43 ────────────────────────────────────
# ─── Atomic remediation tools — pure functions, returning (df, log_entry) ──
# Each fix is short and self-contained. The log_entry uses a uniform shape
# {action, column(s), rows_affected, applied, ...} so the report renders easily.

def _log(action, applied=True, **extra):
    out = {"action": action, "applied": applied}
    out.update(extra)
    return out

def _missing(action, col):
    return None, _log(action, applied=False, column=col, reason="column missing")


def fix_strip_currency(df, col, **_):
    """Strip currency symbols (€, $, £) from a column and coerce the result to numeric.

    Robust against pandas' StringDtype (which rejects integer assignment) and
    against the removal of `errors='ignore'` in modern pd.to_numeric: we always
    use `errors='coerce'` and let downstream tools handle the resulting NaNs.
    """
    if col not in df.columns:
        return df, _log("strip_currency", False, column=col, reason="column missing")
    df = df.copy()
    raw_str = df[col].astype(str)
    before = int(raw_str.str.contains(r"[€$£]", na=False).sum())
    cleaned = raw_str.str.replace(r"[€$£]", "", regex=True).str.strip().str.replace(",", ".", regex=False)
    coerced = pd.to_numeric(cleaned, errors="coerce")
    # Replace the column entirely — switches dtype from object/string to float64,
    # which is what downstream numeric tools (clip_iqr, clip_to_min) expect.
    df[col] = coerced
    return df, _log("strip_currency", column=col, rows_affected=before)


def fix_cast_numeric(df, col, **_):
    """Coerce a column to numeric using loose parsing (commas, units, currency)."""
    if col not in df.columns: return df, _log("cast_numeric", False, column=col, reason="column missing")
    df = df.copy()
    new = coerce_numeric_loose(df[col])
    n = int((df[col].astype(str) != new.astype(str)).sum())
    df[col] = new
    return df, _log("cast_numeric", column=col, rows_affected=n)


def fix_impute_median(df, col, **_):
    """Replace nulls and disguised-null tokens with the column's median value.

    Replaces the whole column with the imputed Series to force float dtype —
    string-dtype columns reject numeric assignment in modern pandas.
    """
    if col not in df.columns:
        return df, _log("impute_median", False, column=col, reason="column missing")
    df = df.copy()
    s = coerce_numeric_loose(df[col])
    median = s.median()
    if pd.isna(median):
        return df, _log("impute_median", False, column=col, reason="no numeric data")
    mask = s.isna() | df[col].astype(str).str.strip().str.lower().isin(DISGUISED_NULLS)
    s_imputed = s.where(~mask, float(median))
    df[col] = s_imputed
    return df, _log("impute_median", column=col, rows_affected=int(mask.sum()), value=float(median))


def fix_impute_mode(df, col, **_):
    """Replace nulls and disguised-null tokens with the column's most frequent value (mode).

    Type-aware: if the column is already numeric (e.g. after cast_numeric in pass 1),
    we compute a *numeric* mode and stay in numeric dtype. Otherwise we go through
    string normalization (handling DISGUISED_NULLS) and impute the string mode.
    Replacing the column in full avoids pandas' dtype-mismatch errors on string
    or float columns.
    """
    if col not in df.columns:
        return df, _log("impute_mode", False, column=col, reason="column missing")
    df = df.copy()

    if pd.api.types.is_numeric_dtype(df[col]):
        # Numeric column — compute mode in numeric domain, no string dance needed
        s = df[col]
        clean = s.dropna()
        if clean.empty:
            return df, _log("impute_mode", False, column=col, reason="all values missing")
        mode = clean.mode().iloc[0]
        mask = s.isna()
        s_imputed = s.where(~mask, mode)
        df[col] = s_imputed
        return df, _log("impute_mode", column=col, rows_affected=int(mask.sum()), value=float(mode))

    # Non-numeric column — handle string + disguised nulls
    raw_str = df[col].astype(str).str.strip()
    clean = raw_str[~raw_str.str.lower().isin(DISGUISED_NULLS) & df[col].notna()]
    if clean.empty:
        return df, _log("impute_mode", False, column=col, reason="all values missing")
    mode = clean.mode().iloc[0]
    mask = df[col].isna() | raw_str.str.lower().isin(DISGUISED_NULLS)
    # Cast to object so pandas accepts the (potentially) heterogeneous string assignment
    base = df[col].astype(object)
    s_imputed = base.where(~mask, mode)
    df[col] = s_imputed
    return df, _log("impute_mode", column=col, rows_affected=int(mask.sum()), value=str(mode))


def fix_clip_iqr(df, col, **_):
    """Clip outliers to the IQR fence (q1 − 1.5·IQR, q3 + 1.5·IQR).

    Replaces the column entirely with the coerced numeric Series (clipped),
    so the resulting dtype is float64 — works on string-dtype columns that
    pandas would otherwise reject for assigning numeric values.
    """
    if col not in df.columns:
        return df, _log("clip_iqr", False, column=col, reason="column missing")
    df = df.copy()
    s = coerce_numeric_loose(df[col])
    if s.dropna().empty:
        return df, _log("clip_iqr", False, column=col, reason="no numeric data")
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    mask = s.notna() & ~s.between(lo, hi)
    # Build the clipped Series and replace the entire column (forces float64 dtype).
    s_clipped = s.where(~mask, s.clip(lo, hi))
    df[col] = s_clipped
    return df, _log("clip_iqr", column=col, rows_affected=int(mask.sum()),
                    bounds={"lower": float(lo), "upper": float(hi)})


def fix_drop_duplicates(df, **_):
    """Drop exact duplicate rows from the DataFrame."""
    df = df.copy()
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, _log("drop_duplicates", rows_affected=n_before - len(df))


def fix_normalize_dates(df, col, target="%Y-%m-%d", **_):
    """Parse date strings in `col` and reformat them to the `target` ISO format (default %Y-%m-%d)."""
    if col not in df.columns: return df, _log("normalize_dates", False, column=col, reason="column missing")
    df = df.copy()
    parsed = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce", dayfirst=True)
    df[col] = parsed.dt.strftime(target).where(parsed.notna(), df[col])
    return df, _log("normalize_dates", column=col, rows_affected=int(parsed.notna().sum()))


def fix_drop_unexpected_columns(df, columns=None, **_):
    """Drop the listed `columns` from the DataFrame (used for unexpected_columns issues)."""
    cols = [c for c in (columns or []) if c in df.columns]
    df = df.copy().drop(columns=cols)
    return df, _log("drop_unexpected_columns", columns=cols, rows_affected=len(df))


def fix_normalize_categorical(df, col, **_):
    """Trim and Title-Case all values in a categorical column for casing harmonization."""
    if col not in df.columns: return df, _log("normalize_categorical", False, column=col, reason="column missing")
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.lower().str.title()
    return df, _log("normalize_categorical", column=col, rows_affected=len(df))


def fix_clip_to_min(df, col, dataset_name=None, **_):
    """Clip values below the configured minimum to the minimum itself.

    Targeted fix for `value_below_minimum` issues. Reads the `min` constraint
    from `NUMERIC_RULES[dataset_name][col]` (populated by `discover_dataset_rules`)
    and replaces every value below it with the minimum.

    Why this matters: `clip_iqr` clips at the statistical IQR fence, which can
    well *include* values that violate a hard domain constraint (e.g. negatives
    when min=0 if the IQR is wide). `clip_to_min` enforces the hard constraint,
    so post-audit no longer flags `value_below_minimum`.
    """
    if col not in df.columns:
        return df, _log("clip_to_min", False, column=col, reason="column missing")
    rule = NUMERIC_RULES.get(dataset_name or "", {}).get(col, {})
    min_value = rule.get("min")
    if min_value is None:
        return df, _log("clip_to_min", False, column=col, reason="no min rule for column")
    df = df.copy()
    parsed = coerce_numeric_loose(df[col])
    mask = parsed.notna() & (parsed < min_value)
    # Replace the whole column to force float dtype — string-dtype columns reject
    # numeric assignment in modern pandas (TypeError: Invalid value '0' for dtype 'str').
    s_clipped = parsed.where(~mask, float(min_value))
    df[col] = s_clipped
    return df, _log("clip_to_min", column=col, rows_affected=int(mask.sum()),
                    bounds={"min": float(min_value)})


def fix_ignore(df, col=None, **_):
    return df, _log("ignore", column=col, reason="deferred or low-priority")


REMEDIATION_TOOLS = {
    "strip_currency":          fix_strip_currency,
    "cast_numeric":            fix_cast_numeric,
    "impute_median":           fix_impute_median,
    "impute_mode":             fix_impute_mode,
    "clip_iqr":                fix_clip_iqr,
    "clip_to_min":             fix_clip_to_min,
    "drop_duplicates":         fix_drop_duplicates,
    "normalize_dates":         fix_normalize_dates,
    "drop_unexpected_columns": fix_drop_unexpected_columns,
    "normalize_categorical":   fix_normalize_categorical,
    "ignore":                  fix_ignore,
}
print(f"Loaded {len(REMEDIATION_TOOLS)} remediation tools.")

# ─── Extracted from notebook cell 45 ────────────────────────────────────
def _merge_dicts(a, b): return {**a, **b}

class AgentState(TypedDict, total=False):
    """─── AgentState (LangGraph TypedDict) ─────────────────────────────────────
    Aggregators for parallel-friendly accumulation across agents:
    """
    dataset_name: str
    raw_df: Any
    issues: List[Dict[str, Any]]
    severity_breakdown: Dict[str, int]
    plan: Annotated[List[Dict[str, Any]], operator.add]   # accumulates from each analysis agent
    sub_scores: Annotated[Dict[str, float], _merge_dicts] # 5 dimensions: validity/completeness/consistency/uniqueness/accuracy
    fixed_df: Any
    correction_log: Annotated[List[Dict[str, Any]], operator.add]  # accumulates from remediation + second_pass
    reliability_score: float
    remediation_score: float
    remediation_score_weighted: float
    # Post-remediation re-audit: deterministic measurement of the fixed_df.
    # Lets the UI show a meaningful before/after delta (otherwise reliability_score
    # always reflects pre-fix issues, which is misleading after a successful run).
    post_issues: List[Dict[str, Any]]
    post_severity_breakdown: Dict[str, int]
    post_sub_scores: Annotated[Dict[str, float], _merge_dicts]
    post_reliability_score: float
    audit_trail: Annotated[List[str], operator.add]       # narrative trace from each agent
    errors: Annotated[List[str], operator.add]


def node_ingest(state: AgentState) -> Dict[str, Any]:
    """Load the dataframe (from datasets registry or already-passed df)."""
    name = state["dataset_name"]
    df = state.get("raw_df")
    if df is None:
        df = datasets[name]
    return {"raw_df": df.copy(), "audit_trail": [f"ingest: loaded `{name}` ({df.shape[0]:,}x{df.shape[1]})"], "errors": []}


def node_discover(state: AgentState) -> Dict[str, Any]:
    """Discover schema/mandatory/format/numeric rules from the dataframe sample.
    Heuristic-only — no LLM call here, to keep the discovery deterministic and free."""
    name = state["dataset_name"]
    df = state["raw_df"]
    rules = discover_dataset_rules(df, name)
    msg = (f"discover: schema_cols={len(rules['schema']['required_columns'])} "
           f"mandatory={len(rules['mandatory'])} dates={len(rules['format_rules'])} "
           f"numeric={len(rules['numeric_rules'])}")
    return {"audit_trail": [msg]}


def node_audit(state: AgentState) -> Dict[str, Any]:
    """Run all 9 deterministic tools and collect issues."""
    name = state["dataset_name"]
    df = state["raw_df"]
    audit = audit_dataset(df, name, auto_discover=False)  # rules already populated
    flat = []
    for r in audit["results"]:
        flat.extend(r["issues"])
    sev = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in flat:
        sev[issue["severity"]] = sev.get(issue["severity"], 0) + 1
    msg = f"audit: {len(flat)} issues — {sev}"
    return {"issues": flat, "severity_breakdown": sev, "audit_trail": [msg]}

# ─── Extracted from notebook cell 47 ────────────────────────────────────
# ─── 4 LLM analysis agents — focused, contextual, structured ──────────────
# Each agent receives:
#   • its slice of issues (filtered by issue_type)
#   • column-level context (samples + stats) for the columns involved
# and returns a plan slice + a 0-1 sub-score for its dimension(s).

SCHEMA_ISSUE_TYPES       = {"missing_required_columns", "duplicate_column_names",
                            "semantic_type_mismatch", "naming_convention_violation"}
COMPLETENESS_ISSUE_TYPES = {"missing_mandatory_values", "missing_optional_values", "sparse_column"}
CONSISTENCY_ISSUE_TYPES  = {"invalid_format_values", "mixed_format_family",
                            "categorical_variant_inconsistency",
                            "exact_duplicate_rows", "duplicate_business_keys",
                            "year_date_mismatch", "month_year_period_mismatch",
                            "logical_count_violation"}
ANOMALY_ISSUE_TYPES      = {"iqr_outliers", "non_numeric_values_in_numeric_field",
                            "forbidden_token_in_numeric_field", "value_below_minimum"}

SEVERITY_PENALTY = {"critical": 0.40, "high": 0.20, "medium": 0.08, "low": 0.02}


# Sparsity threshold above which a column is treated as "structurally dead":
# imputing >95%-missing data introduces noise, so the only sensible action is `ignore`.
# We don't penalise the sub-score for these columns — they're an artefact of the source data,
# not something the pipeline can or should fix.
_DEAD_COLUMN_THRESHOLD = 0.05  # completeness ratio below which we skip the penalty


# Issue types whose only safe action is `ignore` (cosmetic / structural). They show up
# in audits but the correct response is always to leave them alone — renaming a column
# would break downstream references, semantic mismatches require human judgement, etc.
# Excluding them from the sub-score prevents the system from being penalised for
# correctly identifying problems it cannot or should not fix.
_COSMETIC_ISSUE_TYPES = {
    "naming_convention_violation",
}


def _compute_subscore(issues):
    """Compute a 0–1 sub-score from severity-weighted issue penalties, clipped to [0, 1].

    Two exemption rules apply, both designed so the score reflects *fixable* anomalies
    rather than artefacts of source-data structure:

    - **Sparsity-aware**: issues on columns >95% missing are excluded — imputation would
      add noise, `ignore` is the correct action, and the column is structurally untreatable.
    - **Cosmetic-aware**: issues whose only safe action is `ignore` (e.g. naming convention
      violations on column headers — renaming would break downstream references) are excluded
      via the `_COSMETIC_ISSUE_TYPES` set.
    """
    score = 1.0
    for iss in issues:
        # Cosmetic / structural: always exempt
        if iss["issue_type"] in _COSMETIC_ISSUE_TYPES:
            continue
        ev = iss.get("evidence", {}) or {}
        # sparse_column carries `completeness`; missing_*_values carry `missing_ratio`
        if iss["issue_type"] == "sparse_column":
            if ev.get("completeness", 1.0) < _DEAD_COLUMN_THRESHOLD:
                continue
        if iss["issue_type"] in ("missing_optional_values", "missing_mandatory_values"):
            if ev.get("missing_ratio", 0.0) > (1.0 - _DEAD_COLUMN_THRESHOLD):
                continue
        score -= SEVERITY_PENALTY.get(iss["severity"], 0.05)
    return max(0.0, round(score, 3))


def _column_context(df, columns, max_samples=5):
    """Compact dict of {col: {dtype, n_unique, n_nulls, samples}} — only for columns referenced by the issues.
    This is the 'fact pack' the LLM uses to reason concretely instead of hallucinating."""
    ctx = {}
    for col in set(columns):
        if col not in df.columns:
            continue
        s = df[col]
        samples = s.dropna().astype(str).str.strip()
        samples = samples[samples.ne("")].drop_duplicates().head(max_samples).tolist()
        ctx[col] = {
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_nulls": int(s.isna().sum()),
            "samples": samples,
        }
    return ctx


def _safe_parse_json_list(text):
    """Robustly extract a JSON array from LLM output, tolerating code fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"): text = text[4:].strip()
    s, e = text.find("["), text.rfind("]")
    if s >= 0 and e > s:
        try: return json.loads(text[s:e+1])
        except Exception: pass
    return None


# Keywords that justify an `ignore` decision from the LLM. Substring match in the
# `why` field is enough — if NONE of these appear AND the issue isn't on a structurally
# dead column, we treat the `ignore` as "safety-driven, not evidence-driven" and
# override with the issue type's fallback action.
#
# Kept loose intentionally: false negatives (= override when LLM was actually right)
# are cheaper than false positives (= leave fixable issues untouched). The list covers
# the three valid `ignore` reasons explicitly enumerated in the prompt: high missingness,
# cosmetic / structural, and column-type mismatch with the proposed action.
_VALID_IGNORE_KEYWORDS = (
    # (a) high-missingness
    "missing", "sparse", "mostly null", "mostly empty",
    "98%", "99%", ">95%", "98 %", "99 %",
    # (b) cosmetic / structural
    "naming", "cosmetic", "header", "rename", "downstream",
    "would break", "break references", "structural", "untreatable", "irreparable",
    # (c) type-mismatch (most common LLM rationale)
    "string column", "string identifier", "string type", "free text",
    "categorical", "category", "for codes", "for code", "code column",
    "not meaningful", "meaningless", "not applicable", "doesn't apply", "does not apply",
    "not numeric", "non-numeric", "not a date", "not a number", "id column", "identifier",
    # (d) data-corruption
    "corrupt", "destroy", "lossy", "unsafe", "would harm",
    "would corrupt", "introduces noise", "introduce noise",
    # (e) IQR-on-discrete (common edge case)
    "few unique", "low cardinality", "constant",
)


def _ignore_is_valid(rationale: str, issue: Dict[str, Any]) -> bool:
    """Decide whether an LLM `ignore` decision is evidence-based or safety-driven.

    Returns True if any of:
      - the issue is on a structurally dead column (>95% missing, naming convention)
      - the rationale text matches one of `_VALID_IGNORE_KEYWORDS`
    Returns False otherwise — meaning the `ignore` is over-cautious and should be
    replaced by the issue type's deterministic fallback.
    """
    # Structural exemptions — independent of rationale
    ev = issue.get("evidence") or {}
    if issue.get("issue_type") == "sparse_column" and ev.get("completeness", 1.0) < 0.05:
        return True
    if issue.get("issue_type") in ("missing_optional_values", "missing_mandatory_values") \
            and ev.get("missing_ratio", 0.0) > 0.95:
        return True
    if issue.get("issue_type") == "naming_convention_violation":
        return True
    # Rationale-based check
    if not rationale:
        return False
    low = rationale.lower()
    return any(kw in low for kw in _VALID_IGNORE_KEYWORDS)


_FALLBACK = {
    "duplicate_column_names": "drop_unexpected_columns",
    "semantic_type_mismatch": "cast_numeric",
    "missing_mandatory_values": "impute_mode",
    "invalid_format_values": "normalize_dates",
    "mixed_format_family": "normalize_dates",
    "categorical_variant_inconsistency": "normalize_categorical",
    "exact_duplicate_rows": "drop_duplicates",
    "duplicate_business_keys": "drop_duplicates",
    "iqr_outliers": "clip_iqr",
    "value_below_minimum": "clip_to_min",
    "non_numeric_values_in_numeric_field": "cast_numeric",
    "forbidden_token_in_numeric_field": "strip_currency",
}


def _make_agent(name, issue_types, allowed, dims):
    """Build an analysis-agent node with a focused, fix-biased prompt.

    The prompt v3 combines four techniques:
      1. **Senior-engineer framing** — primes the model toward decisive action.
      2. **Default-action mapping** — exposes the deterministic fallback for each
         issue type, so the model has a concrete prior to deviate from only with
         evidence.
      3. **Tight `ignore` policy** — restricts `ignore` to three named cases
         (>95% missing, cosmetic, would-corrupt). Anything else is over-cautious.
      4. **Few-shot examples** — three concrete decisions on NoiPA-style data,
         showing both the right call and a tempting wrong one.

    Without this priming, LLMs default to `ignore` whenever they're uncertain,
    leaving 30–40% of fixable issues untouched. The deterministic fallback is
    aggressive by construction; this prompt aligns the LLM with that aggression
    while still giving it room to override when the column context says so.
    """
    actions_str = ", ".join(allowed)
    # Per-issue-type default action, filtered to what this agent is allowed to do
    defaults_for_agent = {
        it: _FALLBACK[it] for it in issue_types
        if it in _FALLBACK and _FALLBACK[it] in allowed
    }
    defaults_block = "\n".join(f"  - {it:40s} → {act}" for it, act in defaults_for_agent.items()) \
        or "  (no defaults applicable; choose from allowed actions)"

    # Few-shot examples picked to cover the three failure modes we observed:
    # over-cautious ignore, type-mismatch on string-as-numeric, structural sparsity.
    few_shots = (
        "EXAMPLES (calibrate your decisions on these):\n"
        '  Issue: missing_mandatory_values on "descrizione" (6% nulls, 113 unique values)\n'
        "    ✓ impute_mode  — high-cardinality categorical, mode is plausible\n"
        "    ✗ ignore       — \"uncertain\" is NOT a valid reason\n"
        "\n"
        '  Issue: missing_optional_values on "note_operatore" (98% nulls)\n'
        "    ✓ ignore       — column is 98% missing, imputation = noise\n"
        "    ✗ impute_mode  — would replace 98% with the mode, distorting data\n"
        "\n"
        '  Issue: iqr_outliers on "ente%code" (integer, 8 unique values)\n'
        "    ✓ ignore       — integer categorical / code, IQR meaningless for codes\n"
        "    ✗ clip_iqr     — would corrupt valid category codes\n"
        "\n"
        '  Issue: value_below_minimum on "spesa" (numeric, min=0, 11 negative rows)\n'
        "    ✓ clip_to_min  — clips -5, -3 etc to 0; respects domain constraint\n"
        "    ✗ clip_iqr     — IQR fence may include negatives, leaving them unfixed\n"
    )

    sys_prompt = (
        f"You are a senior data engineer responsible for the {name} dimension of a "
        "data-quality pipeline on Italian public-sector CSVs (NoiPA). You've seen "
        "thousands of these files: disguised nulls, currency in numeric fields, "
        "mixed date formats, value-below-minimum violations. You know that *fixing* "
        "the anomaly is almost always preferable to skipping it, because downstream "
        "queries fail on bad data. Be decisive, not cautious.\n\n"

        f"Scope: issues of type {sorted(issue_types)}.\n"
        f"Allowed actions: {actions_str}.\n\n"

        "DECISION POLICY:\n"
        "Pick the action that ACTUALLY FIXES the issue. The defaults below are "
        "calibrated for the average case — diverge from them only if the column "
        "context (samples, null rate, cardinality, dtype) makes the default harmful.\n\n"

        f"DEFAULT ACTION per issue type:\n{defaults_block}\n\n"

        "USE `ignore` ONLY WHEN one of these three conditions holds (and your `why` "
        "must cite which one):\n"
        "  (a) the column is >95% missing → imputation = noise\n"
        "  (b) the issue is cosmetic (e.g. naming convention, header style) and the "
        "fix would break downstream code/SQL/serialization\n"
        "  (c) the column type makes the default action invalid (e.g. clip_iqr on a "
        "string identifier, normalize_dates on a non-date column)\n"
        "Choosing `ignore` outside these three reasons is a SCORE PENALTY for the "
        "system. Do NOT use `ignore` because you are uncertain — the default is "
        "calibrated for uncertainty.\n\n"

        f"{few_shots}\n"

        "REASONING PROTOCOL — for every issue, mentally walk through:\n"
        "  1) What is this column really? (numeric / categorical / string-id / date / free-text)\n"
        "  2) Does the default action match that type, or would it corrupt the data?\n"
        "  3) If the default fits → use it. If not → pick another allowed action that fits, "
        "or `ignore` only if condition (a)/(b)/(c) above applies.\n\n"

        "Output STRICT JSON: an array of objects, one per input issue in the same order:\n"
        '  [{"i": <issue_index>, "action": "<allowed_name>", "why": "<≤25-word reason citing column type or one of (a)/(b)/(c)>"}]\n'
        "No markdown fencing, no prose outside the JSON."
    )

    def node(state: AgentState) -> Dict[str, Any]:
        issues = state.get("issues", [])
        relevant = [(i, iss) for i, iss in enumerate(issues) if iss["issue_type"] in issue_types]

        sub_score = _compute_subscore([iss for _, iss in relevant])
        sub_scores = {k: sub_score for k in dims}

        if not relevant:
            return {"plan": [], "sub_scores": sub_scores,
                    "audit_trail": [f"{name}: no issues in scope (score={sub_score})"]}

        df = state["raw_df"]
        cols_involved = [c for _, iss in relevant for c in iss.get("columns", [])]
        col_ctx = _column_context(df, cols_involved)

        compact = [{"i": i, "type": iss["issue_type"], "sev": iss["severity"],
                    "cols": iss.get("columns", []),
                    "msg": (iss.get("message") or "")[:120]}
                   for i, iss in relevant]

        user_msg = (
            f"Dataset: {state['dataset_name']}\n"
            f"Issues to plan ({len(compact)}):\n{json.dumps(compact, ensure_ascii=False)}\n\n"
            f"Column context (referenced by these issues):\n{json.dumps(col_ctx, ensure_ascii=False, default=str)}"
        )

        try:
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
            parsed = _safe_parse_json_list(resp.content)
        except Exception as e:
            parsed = None
            err = f"{name}_llm_failed: {type(e).__name__}: {e}"
        else:
            err = None

        if parsed is None or len(parsed) != len(relevant):
            plan = [{"issue_index": i, "action": _FALLBACK.get(iss["issue_type"], "ignore"),
                     "rationale": f"deterministic fallback for {iss['issue_type']}", "agent": name}
                    for i, iss in relevant]
            tag = " (fallback)"
        else:
            plan = []
            n_overrides = 0
            for entry, (i, iss) in zip(parsed, relevant):
                action = entry.get("action") if entry.get("action") in allowed else "ignore"
                rationale = (entry.get("why") or "")[:200]

                # Safety net: if the LLM picked `ignore` and the issue type has a
                # sensible deterministic fallback, AND the rationale doesn't cite
                # one of the three valid reasons (>95% missing / cosmetic / type-
                # mismatch), override with the fallback. This counteracts the
                # safety-driven `ignore` bias that LLMs exhibit on ambiguous cases.
                if action == "ignore":
                    fallback_action = _FALLBACK.get(iss["issue_type"])
                    if fallback_action and fallback_action in allowed:
                        if not _ignore_is_valid(rationale, iss):
                            action = fallback_action
                            rationale = (f"override: LLM chose ignore without an evidence-based reason; "
                                         f"applying default '{fallback_action}'. "
                                         f"Original LLM rationale: {rationale[:80]}")
                            n_overrides += 1

                plan.append({"issue_index": i, "action": action,
                             "rationale": rationale,
                             "agent": name})
            tag = f" (LLM, {n_overrides} ignore-overrides)" if n_overrides else " (LLM)"

        out = {"plan": plan, "sub_scores": sub_scores,
               "audit_trail": [f"{name}: {len(plan)} actions planned, score={sub_score}{tag}"]}
        if err: out["errors"] = [err]
        return out

    return node


# Allowed actions per agent — restricted enums keep LLM output sharp and predictable
SCHEMA_ACTIONS       = ["drop_unexpected_columns", "cast_numeric", "normalize_dates", "ignore"]
COMPLETENESS_ACTIONS = ["impute_median", "impute_mode", "ignore"]
CONSISTENCY_ACTIONS  = ["normalize_dates", "drop_duplicates", "normalize_categorical", "ignore"]
ANOMALY_ACTIONS      = ["clip_iqr", "clip_to_min", "cast_numeric", "strip_currency", "ignore"]

node_schema_agent       = _make_agent("Schema",       SCHEMA_ISSUE_TYPES,       SCHEMA_ACTIONS,       ["validity"])
node_completeness_agent = _make_agent("Completeness", COMPLETENESS_ISSUE_TYPES, COMPLETENESS_ACTIONS, ["completeness"])
node_consistency_agent  = _make_agent("Consistency",  CONSISTENCY_ISSUE_TYPES,  CONSISTENCY_ACTIONS,  ["consistency", "uniqueness"])
node_anomaly_agent      = _make_agent("Anomaly",      ANOMALY_ISSUE_TYPES,      ANOMALY_ACTIONS,      ["accuracy"])

# ─── Extracted from notebook cell 49 ────────────────────────────────────
# Actions that operate on a specific column — they require col to be set.
# Actions outside this set work on the whole df (drop_duplicates) or use `columns` (drop_unexpected_columns).
_COLUMN_BOUND_ACTIONS = {
    "strip_currency", "cast_numeric", "impute_median", "impute_mode",
    "clip_iqr", "clip_to_min", "normalize_dates", "normalize_categorical",
}

def node_remediation(state: AgentState) -> Dict[str, Any]:
    """Apply the merged plan to the dataframe with atomic deterministic tools.

    Defensive: column-bound actions (impute_*, clip_iqr, normalize_*, etc.) require
    a real column name. If the planner produced an entry without a usable column —
    or with a column that isn't in the dataframe — we log it explicitly with the
    reason instead of crashing inside the tool with a KeyError.
    """
    df = state["raw_df"].copy()
    issues = state["issues"]
    plan = state.get("plan", [])
    log = []

    for entry in plan:
        idx = entry.get("issue_index", -1)
        if not (0 <= idx < len(issues)):
            continue
        action = entry["action"]
        tool_fn = REMEDIATION_TOOLS.get(action, fix_ignore)
        issue = issues[idx]
        cols = [c for c in (issue.get("columns") or []) if c is not None and c != ""]
        col = cols[0] if cols else None

        # Pre-flight guard: column-bound actions need a real column present in df.
        if action in _COLUMN_BOUND_ACTIONS and (col is None or col not in df.columns):
            reason = ("missing column reference in audit issue"
                      if col is None else f"column `{col}` not found in dataframe")
            log.append({
                "action": action, "applied": False,
                "column": col, "columns": cols,
                "rows_affected": 0,
                "reason": reason,
                "issue_index": idx, "issue_type": issue["issue_type"],
                "rationale": entry.get("rationale", ""),
                "agent": entry.get("agent", ""),
            })
            continue

        try:
            # `dataset_name` is needed by `clip_to_min` to look up `min` in NUMERIC_RULES.
            # Other fix tools accept **_ so the extra kwarg is harmless.
            df, log_entry = tool_fn(df, col=col, columns=cols, dataset_name=state.get("dataset_name"))
            log_entry.update({
                "issue_index": idx,
                "issue_type": issue["issue_type"],
                "rationale": entry.get("rationale", ""),
                "agent": entry.get("agent", ""),
            })
            # Ensure column field is always populated for the UI
            log_entry.setdefault("column", col)
            log.append(log_entry)
        except Exception as e:
            log.append({
                "action": action, "applied": False,
                "column": col, "columns": cols,
                "rows_affected": 0,
                "error": f"{type(e).__name__}: {e}",
                "reason": f"{type(e).__name__}: {e}",
                "issue_index": idx, "issue_type": issue["issue_type"],
                "rationale": entry.get("rationale", ""),
                "agent": entry.get("agent", ""),
            })

    n_applied = sum(1 for e in log if e.get("applied"))
    return {"fixed_df": df, "correction_log": log,
            "audit_trail": [f"remediation: applied {n_applied}/{len(log)} fixes"]}


# ─── 5-dimension reliability scoring (ISO-8000-style) ──────────────────────
RELIABILITY_WEIGHTS = {
    "completeness": 0.30,
    "consistency":  0.25,
    "validity":     0.20,
    "uniqueness":   0.15,
    "accuracy":     0.10,
}
assert abs(sum(RELIABILITY_WEIGHTS.values()) - 1.0) < 1e-9


# Mapping from reliability dimension → set of issue types that drive that dimension's sub-score.
# Mirrors the `dims` parameter passed to `_make_agent` for each LLM agent.
_DIMENSION_ISSUE_TYPES = {
    "validity":     SCHEMA_ISSUE_TYPES,
    "completeness": COMPLETENESS_ISSUE_TYPES,
    "consistency":  CONSISTENCY_ISSUE_TYPES,
    "uniqueness":   {"exact_duplicate_rows", "duplicate_business_keys"},
    "accuracy":     ANOMALY_ISSUE_TYPES,
}


def node_re_audit(state: AgentState) -> Dict[str, Any]:
    """Re-run the deterministic audit on `fixed_df` to measure post-remediation reliability.

    Zero LLM calls. This node closes the loop: without it, the reliability score
    always reflects pre-remediation issues (which is misleading once fixes are applied).
    The supervisor uses these post-scores to compute a `post_reliability_score`
    that the UI can show as a before/after delta.
    """
    name = state["dataset_name"]
    df = state.get("fixed_df")
    if df is None:
        return {"audit_trail": ["re_audit: skipped (no fixed_df)"]}

    audit = audit_dataset(df, name, auto_discover=False)  # rules already populated upstream
    flat: List[Dict[str, Any]] = []
    for r in audit["results"]:
        flat.extend(r["issues"])

    sev = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for iss in flat:
        sev[iss["severity"]] = sev.get(iss["severity"], 0) + 1

    post_sub: Dict[str, float] = {}
    for dim, types in _DIMENSION_ISSUE_TYPES.items():
        relevant = [iss for iss in flat if iss["issue_type"] in types]
        post_sub[dim] = _compute_subscore(relevant)

    msg = f"re_audit: {len(flat)} residual issues post-fix — {sev}"
    return {
        "post_issues": flat,
        "post_severity_breakdown": sev,
        "post_sub_scores": post_sub,
        "audit_trail": [msg],
    }


def node_second_pass_remediation(state: AgentState) -> Dict[str, Any]:
    """Apply deterministic fallback remediation to issues that survived the first pass.

    Zero LLM calls. For every residual issue produced by `re_audit`, look up the
    deterministic fallback action (`_FALLBACK[issue_type]`) and apply it directly,
    bypassing the LLM. This closes the gap between the LLM's safety-driven `ignore`
    decisions and the deterministic floor that the rule-based fallback would have
    produced.

    Why two passes are better than one:
      - Pass 1 (`remediation`): the LLM picks actions, optionally over-ridden when
        the `ignore` rationale is not evidence-based. Some genuinely cautious
        ignores (rationale cites column-type mismatch, etc.) survive.
      - Pass 2 (this node): for the surviving residual issues, if `_FALLBACK`
        provides a non-`ignore` default that's compatible with the column, we
        apply it. The deterministic fallback is calibrated to the average case
        and is the right floor when the LLM has been over-cautious.

    The fixed_df is updated in place; the new fix entries are concatenated to
    the existing `correction_log` so the audit trail stays complete.
    """
    df = state.get("fixed_df")
    if df is None:
        return {"audit_trail": ["second_pass_remediation: skipped (no fixed_df)"]}

    df = df.copy()
    post_issues = state.get("post_issues") or []
    existing_log = list(state.get("correction_log") or [])
    new_log: List[Dict[str, Any]] = []

    for iss in post_issues:
        action = _FALLBACK.get(iss.get("issue_type"))
        if not action or action == "ignore":
            continue
        cols = [c for c in (iss.get("columns") or []) if c and c != ""]
        col = cols[0] if cols else None

        # Skip if action is column-bound but we have no usable column
        if action in _COLUMN_BOUND_ACTIONS and (col is None or col not in df.columns):
            continue

        tool_fn = REMEDIATION_TOOLS.get(action)
        if tool_fn is None:
            continue

        try:
            df, log_entry = tool_fn(df, col=col, columns=cols, dataset_name=state.get("dataset_name"))
            log_entry.update({
                "issue_type": iss.get("issue_type"),
                "agent": "SecondPass",
                "rationale": f"second-pass deterministic fallback ({action}) for {iss.get('issue_type')}",
            })
            log_entry.setdefault("column", col)
            new_log.append(log_entry)
        except Exception as e:
            new_log.append({
                "action": action, "applied": False,
                "column": col, "columns": cols,
                "issue_type": iss.get("issue_type"),
                "agent": "SecondPass",
                "error": f"{type(e).__name__}: {e}",
                "rationale": f"second-pass exception while applying {action}",
            })

    n_applied = sum(1 for e in new_log if e.get("applied"))
    msg = f"second_pass_remediation: applied {n_applied}/{len(new_log)} fallback fixes on {len(post_issues)} residual issues"
    return {
        "fixed_df": df,
        # `correction_log` is Annotated[..., operator.add] in AgentState — return only
        # NEW entries; LangGraph's reducer concatenates them with the existing log.
        # (Previously we returned `existing_log + new_log`, which double-appended
        # under the webapp's stream_quality_pipeline custom merge.)
        "correction_log": new_log,
        "audit_trail": [msg],
    }


def node_final_audit(state: AgentState) -> Dict[str, Any]:
    """Re-audit `fixed_df` after the second-pass deterministic remediation.

    Replaces the `post_issues`, `post_severity_breakdown`, and `post_sub_scores`
    produced by `node_re_audit` with the *final* values measured after both
    LLM remediation and deterministic-fallback second pass. This is what the
    supervisor uses to compute the headline post_reliability_score.
    """
    name = state["dataset_name"]
    df = state.get("fixed_df")
    if df is None:
        return {"audit_trail": ["final_audit: skipped (no fixed_df)"]}

    audit = audit_dataset(df, name, auto_discover=False)
    flat: List[Dict[str, Any]] = []
    for r in audit["results"]:
        flat.extend(r["issues"])

    sev = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for iss in flat:
        sev[iss["severity"]] = sev.get(iss["severity"], 0) + 1

    post_sub: Dict[str, float] = {}
    for dim, types in _DIMENSION_ISSUE_TYPES.items():
        relevant = [iss for iss in flat if iss["issue_type"] in types]
        post_sub[dim] = _compute_subscore(relevant)

    msg = f"final_audit: {len(flat)} residual issues after both passes — {sev}"
    return {
        "post_issues": flat,
        "post_severity_breakdown": sev,
        "post_sub_scores": post_sub,
        "audit_trail": [msg],
    }


# Severity weights for the resolution-rate metric — closing one critical issue counts
# more than closing five lows. Independent of SEVERITY_PENALTY (which feeds reliability).
_SEVERITY_RESOLUTION_WEIGHT = {"critical": 4.0, "high": 2.0, "medium": 1.0, "low": 0.5}


def node_supervisor(state: AgentState) -> Dict[str, Any]:
    """Deterministic supervisor — aggregates sub-scores into 0-100 quality metrics.

    Produces three complementary numbers:
    - **reliability_score** (pre-fix): "how dirty is the input?" (penalty-based)
    - **post_reliability_score**: "how clean is the output?" (penalty-based on
      issues that survive both LLM and deterministic-fallback passes)
    - **remediation_score** + **remediation_score_weighted**: "how much of the
      detected work did the pipeline actually close?". A separate axis from
      reliability — useful for honest reporting and pitch slides.

    No LLM call: the math is exact and reproducible.
    """
    sub = state.get("sub_scores", {})
    pre_final = round(sum(RELIABILITY_WEIGHTS[k] * sub.get(k, 1.0) for k in RELIABILITY_WEIGHTS) * 100, 1)

    post_sub = state.get("post_sub_scores", {})
    if post_sub:
        post_final = round(sum(RELIABILITY_WEIGHTS[k] * post_sub.get(k, 1.0) for k in RELIABILITY_WEIGHTS) * 100, 1)
        delta_str = f" → post-fix={post_final}/100 (Δ={round(post_final - pre_final, 1):+})"
    else:
        post_final = None
        delta_str = ""

    # Resolution-rate metrics — meaningful only if final_audit produced post_issues
    issues_pre = state.get("issues") or []
    issues_post = state.get("post_issues") or []
    n_pre, n_post = len(issues_pre), len(issues_post)
    if n_pre > 0:
        rem_plain = round(100.0 * max(0, n_pre - n_post) / n_pre, 1)
        w_pre = sum(_SEVERITY_RESOLUTION_WEIGHT.get(i.get("severity", "medium"), 1.0) for i in issues_pre)
        w_post = sum(_SEVERITY_RESOLUTION_WEIGHT.get(i.get("severity", "medium"), 1.0) for i in issues_post)
        rem_weighted = round(100.0 * max(0.0, w_pre - w_post) / w_pre, 1) if w_pre > 0 else 100.0
    else:
        rem_plain = 100.0
        rem_weighted = 100.0
    rem_str = f" · remediation={rem_plain}% (weighted {rem_weighted}%)" if issues_post else ""

    msg = f"supervisor: reliability={pre_final}/100{delta_str}{rem_str}"

    out: Dict[str, Any] = {
        "reliability_score": pre_final,
        "sub_scores": {k: round(sub.get(k, 1.0) * 100, 1) for k in RELIABILITY_WEIGHTS},
        "audit_trail": [msg],
    }
    if post_final is not None:
        out["post_reliability_score"] = post_final
        out["post_sub_scores"] = {k: round(post_sub.get(k, 1.0) * 100, 1) for k in RELIABILITY_WEIGHTS}
        out["remediation_score"] = rem_plain
        out["remediation_score_weighted"] = rem_weighted
    return out

# ─── Extracted from notebook cell 51 ────────────────────────────────────
# ─── Compile the StateGraph (6 nodes, linear, single iteration) ───────────
graph_builder = StateGraph(AgentState)
graph_builder.add_node("ingest",        node_ingest)
graph_builder.add_node("discover",      node_discover)
graph_builder.add_node("audit",         node_audit)
graph_builder.add_node("schema",        node_schema_agent)
graph_builder.add_node("completeness",  node_completeness_agent)
graph_builder.add_node("consistency",   node_consistency_agent)
graph_builder.add_node("anomaly",       node_anomaly_agent)
graph_builder.add_node("remediation",       node_remediation)
graph_builder.add_node("re_audit",          node_re_audit)
graph_builder.add_node("second_pass",       node_second_pass_remediation)
graph_builder.add_node("final_audit",       node_final_audit)
graph_builder.add_node("supervisor",        node_supervisor)

graph_builder.add_edge(START, "ingest")
graph_builder.add_edge("ingest", "discover")
graph_builder.add_edge("discover", "audit")
graph_builder.add_edge("audit", "schema")
graph_builder.add_edge("schema", "completeness")
graph_builder.add_edge("completeness", "consistency")
graph_builder.add_edge("consistency", "anomaly")
graph_builder.add_edge("anomaly",      "remediation")
graph_builder.add_edge("remediation",  "re_audit")
graph_builder.add_edge("re_audit",     "second_pass")
graph_builder.add_edge("second_pass",  "final_audit")
graph_builder.add_edge("final_audit",  "supervisor")
graph_builder.add_edge("supervisor",   END)

quality_graph = graph_builder.compile()
print(f"StateGraph compiled: 12 nodes, 4 LLM agents, two-pass remediation (LLM + deterministic).")
print(f"Reliability dimensions: {list(RELIABILITY_WEIGHTS.keys())}")

# ─── Extracted from notebook cell 54 ────────────────────────────────────
from jinja2 import Template

REPORT_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Data Quality Report — {{ dataset_name }}</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:16px;color:#222}
h1{color:#0a6e2c;border-bottom:3px solid #0a6e2c;padding-bottom:8px}
h2{color:#444;margin-top:32px;border-left:4px solid #0a6e2c;padding-left:10px}
.score{display:flex;gap:24px;align-items:center;background:#f4faf6;border:1px solid #cde9d6;padding:18px;border-radius:8px;margin:16px 0}
.gauge{font-size:56px;font-weight:700;color:{{ score_color }}}
.delta{font-size:13px;color:#0a6e2c;margin-top:6px;font-family:monospace}
.delta.neg{color:#c0392b}
.subs{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0}
.sub{background:#fafafa;border:1px solid #e0e0e0;padding:10px;border-radius:6px;text-align:center}
.sub .l{font-size:11px;color:#666;text-transform:uppercase}
.sub .v{font-size:22px;font-weight:600;color:#0a6e2c}
.sub .pre{font-size:11px;color:#888;text-decoration:line-through;font-family:monospace}
.sub .arr{font-size:11px;color:#0a6e2c;font-family:monospace;margin:2px 0}
.sub .w{font-size:10px;color:#999}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f0f0f0}
.sev-critical{background:#ffd6d6}.sev-high{background:#ffe9d6}.sev-medium{background:#fff5d6}.sev-low{background:#f0f0f0}
.tag{background:#0a6e2c;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-family:monospace}
.summary{background:#fffdf3;border-left:4px solid #d4ad28;padding:12px 16px;margin:16px 0}
.trail{background:#f4f4f4;border-left:4px solid #888;padding:8px 12px;font-family:monospace;font-size:12px;white-space:pre-wrap}
.muted{color:#888;font-size:12px}
</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head><body>

<h1>Data Quality Report — {{ dataset_name }}</h1>
<p class="muted">{{ generated_at }} · {{ provider }} · multi-agent pipeline</p>

<div class="score">
  <div class="gauge">{{ display_score }}/100</div>
  <div>
    <strong>Reliability {% if post_reliability_score is not none %}— post-fix{% else %}score{% endif %}</strong> (ISO-8000 weighted)<br>
    <span class="muted">{{ issues|length }} issues detected · {{ applied_count }} corrections applied{% if post_issues_count is not none %} · {{ post_issues_count }} residual after re-audit{% endif %}</span>
    {% if post_reliability_score is not none %}
    <div class="delta {% if score_delta < 0 %}neg{% endif %}">
      before {{ reliability_score }} → after {{ post_reliability_score }} ({{ '+' if score_delta >= 0 else '' }}{{ score_delta }})
    </div>
    {% endif %}
  </div>
</div>

<div class="subs">
{% for cat, val in display_sub_scores.items() %}
  <div class="sub">
    <div class="l">{{ cat }}</div>
    {% if pre_sub_scores and pre_sub_scores[cat] != val %}
      <div class="arr">{{ pre_sub_scores[cat] }} →</div>
    {% endif %}
    <div class="v">{{ val }}</div>
    <div class="w">w={{ weights[cat] }}</div>
  </div>
{% endfor %}
</div>

<div id="radar" style="max-width:480px;margin:16px auto"></div>
<script>
Plotly.newPlot("radar", [{type:"scatterpolar",r:{{ radar_values|tojson }},theta:{{ radar_labels|tojson }},fill:"toself",marker:{color:"#0a6e2c"}}],
{polar:{radialaxis:{visible:true,range:[0,100]}},showlegend:false,margin:{t:20,b:20,l:20,r:20},height:340},{displayModeBar:false});
</script>

<h2>Executive Summary</h2>
<div class="summary">{{ exec_summary | safe }}</div>

<h2>Audit trail</h2>
<div class="trail">{% for line in audit_trail %}{{ line }}\n{% endfor %}</div>

<h2>Top issues</h2>
<table><tr><th>#</th><th>Tool</th><th>Type</th><th>Sev</th><th>Cols</th><th>Rows</th><th>Message</th></tr>
{% for issue in top_issues %}<tr class="sev-{{ issue.severity }}">
<td>{{ loop.index }}</td><td><span class="tag">{{ issue.tool }}</span></td>
<td>{{ issue.issue_type }}</td><td>{{ issue.severity }}</td>
<td>{{ issue.columns | join(", ") }}</td><td>{{ issue.row_count or "-" }}</td>
<td>{{ issue.message }}</td></tr>{% endfor %}</table>

<h2>Correction log ({{ applied_count }} applied)</h2>
<table><tr><th>#</th><th>Agent</th><th>Action</th><th>Issue</th><th>Col(s)</th><th>Rows</th><th>Why</th></tr>
{% for log in correction_log %}<tr>
<td>{{ loop.index }}</td><td>{{ log.agent or "-" }}</td>
<td><span class="tag">{{ log.action }}</span></td><td>{{ log.issue_type or "-" }}</td>
<td>{{ log.columns or log.column or "-" }}</td><td>{{ log.rows_affected or "-" }}</td>
<td>{{ log.rationale or log.reason or "" }}</td></tr>{% endfor %}</table>

<h2>Dataset overview</h2>
<p>Shape: <strong>{{ raw_shape }}</strong> → <strong>{{ fixed_shape }}</strong> after remediation.</p>
{{ sample_html | safe }}

</body></html>
""")


def render_quality_report(state, provider_name="DeepSeek-Chat (V3)"):
    """Render the full quality report HTML page from the final pipeline state (with embedded plotly radar).

    When `state` includes `post_reliability_score` and `post_sub_scores` (i.e. the
    `re_audit` node ran), the report shows the *post-fix* score as the primary
    gauge and renders the pre→post delta on each sub-score box.
    """
    from datetime import datetime
    issues = state.get("issues", [])
    log = state.get("correction_log", [])
    applied = sum(1 for e in log if e.get("applied"))

    pre_score  = state.get("reliability_score", 0)
    post_score = state.get("post_reliability_score")  # None if re_audit didn't run

    # The gauge shows the post-fix score (the *delivered* quality) when available;
    # otherwise it falls back to pre-fix. score_color tracks the displayed value.
    display_score = post_score if post_score is not None else pre_score
    score_color = "#0a6e2c" if display_score >= 70 else ("#d4ad28" if display_score >= 40 else "#c0392b")
    score_delta = round(post_score - pre_score, 1) if post_score is not None else None

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    top_issues = sorted(issues, key=lambda x: (sev_order.get(x["severity"], 4), -(x.get("row_count") or 0)))[:25]

    pre_sub_scores  = state.get("sub_scores", {k: 100.0 for k in RELIABILITY_WEIGHTS})
    post_sub_scores = state.get("post_sub_scores")
    display_sub_scores = post_sub_scores if post_sub_scores else pre_sub_scores

    radar_labels = list(RELIABILITY_WEIGHTS.keys())
    radar_values = [display_sub_scores.get(k, 100.0) for k in radar_labels]

    raw_df, fixed_df = state["raw_df"], state.get("fixed_df", state["raw_df"])
    post_issues = state.get("post_issues", [])

    return REPORT_TEMPLATE.render(
        dataset_name=state["dataset_name"],
        # Gauge shows post-fix when available; pre/post passed for the delta line
        display_score=display_score, score_color=score_color,
        reliability_score=pre_score,
        post_reliability_score=post_score,
        score_delta=score_delta,
        post_issues_count=(len(post_issues) if post_issues else None),
        # Sub-score grid: display = post if available, with pre alongside
        display_sub_scores=display_sub_scores,
        pre_sub_scores=(pre_sub_scores if post_sub_scores else None),
        weights={k: f"{int(v*100)}%" for k, v in RELIABILITY_WEIGHTS.items()},
        radar_labels=radar_labels, radar_values=radar_values,
        issues=issues, top_issues=top_issues,
        correction_log=log, applied_count=applied,
        audit_trail=state.get("audit_trail", []),
        raw_shape=f"{raw_df.shape[0]:,} × {raw_df.shape[1]}",
        fixed_shape=f"{fixed_df.shape[0]:,} × {fixed_df.shape[1]}",
        sample_html=fixed_df.head(5).to_html(border=0, index=False),
        provider=provider_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        exec_summary=state.get("exec_summary") or _default_exec_summary(state),
    )


def _default_exec_summary(state):
    """Build a deterministic executive-summary string from severity counts and reliability score.

    Mentions the post-fix score and delta when the re_audit ran; falls back to
    pre-fix-only narrative for backward compatibility.
    """
    pre  = state.get("reliability_score", 0)
    post = state.get("post_reliability_score")
    sev  = state.get("severity_breakdown", {})
    post_sev = state.get("post_severity_breakdown") or {}
    n_applied = sum(1 for e in state.get("correction_log", []) if e.get("applied"))

    display = post if post is not None else pre
    verdict = "high" if display >= 70 else ("medium" if display >= 40 else "low")

    base = (f"Dataset <strong>{state['dataset_name']}</strong> reached reliability "
            f"<strong>{display}/100</strong> ({verdict} level). Initial issues: "
            f"{sev.get('critical',0)} critical / {sev.get('high',0)} high / "
            f"{sev.get('medium',0)} medium / {sev.get('low',0)} low. "
            f"The pipeline applied {n_applied} corrections")

    if post is not None:
        residual = sum(post_sev.values()) if post_sev else 0
        delta = round(post - pre, 1)
        return (base + f", lifting the score from <strong>{pre}/100</strong> to "
                f"<strong>{post}/100</strong> (Δ {'+' if delta >= 0 else ''}{delta}). "
                f"{residual} issue(s) remain after the deterministic re-audit.")
    return base + "."

# ─── Extracted from notebook cell 56 ────────────────────────────────────
# Optional LLM-narrative for the executive summary (single short call per run)
NARRATIVE_PROMPT = """You are a data quality analyst. In ONE concise paragraph (40-80 words) in English,
summarize the data quality of dataset '{name}': mention its reliability score ({score}/100),
the top severity issues, and the most impactful remediation actions taken. Use professional, factual tone.
Output plain text only (no markdown, no headers)."""


def add_llm_narrative(state):
    """Optionally call LLM to produce a narrative executive summary.
    Falls back gracefully if the LLM is unavailable."""
    try:
        prompt = NARRATIVE_PROMPT.format(name=state["dataset_name"], score=state["reliability_score"])
        sev = state.get("severity_breakdown", {})
        sample_issues = "\n".join([f"- {i['issue_type']} ({i['severity']}) on {i.get('columns')}"
                                     for i in state.get("issues", [])[:5]])
        sample_actions = "\n".join([f"- {l.get('action')} on {l.get('column') or l.get('columns')} ({l.get('agent')})"
                                      for l in state.get("correction_log", []) if l.get("applied")][:5])
        ctx = f"Severity breakdown: {sev}\nTop issues:\n{sample_issues}\nApplied actions:\n{sample_actions}"
        resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=ctx)])
        state["exec_summary"] = resp.content.strip()
    except Exception:
        state["exec_summary"] = None
    return state


def run_quality_pipeline(df, name="user_dataset", with_narrative=True):
    """Run the full pipeline on a single dataframe and return both the final state and the rendered HTML.
    This is the canonical entry point — used by the notebook smoke test, the Streamlit app,
    and any external caller."""
    # Ensure datasets registry has this entry (so node_ingest can pull it back if df not in state)
    datasets[name] = df
    final_state = quality_graph.invoke({"dataset_name": name, "raw_df": df.copy()})
    if with_narrative:
        final_state = add_llm_narrative(final_state)
    html = render_quality_report(final_state)
    return final_state, html


print("Phase 6 ready. Use `run_quality_pipeline(df, name='your_dataset')` to invoke the pipeline.")


# ─── stream_quality_pipeline (NEW — used by webapp for live SSE timelines) ───
def stream_quality_pipeline(df: pd.DataFrame, name: str = "user") -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Run the pipeline and yield (node_name, accumulated_state_snapshot) after each node.

    Used by webapp/server.py to drive live timeline updates via Server-Sent Events.
    The state snapshot is the merged accumulator (mirrors LangGraph reducers) so the
    caller can inspect partial results node-by-node without re-implementing the merge.
    """
    state: Dict[str, Any] = {"dataset_name": name, "raw_df": df.copy()}
    # Ensure the in-memory `datasets` registry has this entry, so node_ingest can
    # look it up if needed (the notebook pattern).
    if "datasets" in globals():
        datasets[name] = df  # type: ignore[name-defined]

    for event in quality_graph.stream({"dataset_name": name, "raw_df": df.copy()}):
        for node_name, update in event.items():
            for k, v in update.items():
                if isinstance(state.get(k), list) and isinstance(v, list):
                    state[k] = state[k] + v
                elif isinstance(state.get(k), dict) and isinstance(v, dict):
                    state[k] = {**state[k], **v}
                else:
                    state[k] = v
            # yield a shallow copy so the caller can hold onto the snapshot
            yield node_name, dict(state)


# ─── CLI smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print(f"Verifying LLM... ", end="", flush=True)
    print(verify_llm())

    spesa_path = PROJECT_ROOT / "agents" / "data" / "project_data_quality" / "spesa.csv"
    if not spesa_path.exists():
        print(f"ERROR: demo dataset not found at {spesa_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading {spesa_path.name}...")
    df = pd.read_csv(spesa_path)
    print(f"  shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("\nRunning pipeline (live LLM calls)...")
    final_state, html = run_quality_pipeline(df, name="spesa", with_narrative=False)

    print(f"\n=== Result ===")
    print(f"  Reliability: {final_state['reliability_score']}/100")
    print(f"  Sub-scores:  {final_state['sub_scores']}")
    print(f"  Issues:      {len(final_state['issues'])}")
    applied = sum(1 for e in final_state['correction_log'] if e.get('applied'))
    print(f"  Applied:     {applied}/{len(final_state['correction_log'])}")
    print(f"  HTML report: {len(html):,} bytes")
    print(f"\nAudit trail:")
    for line in final_state.get("audit_trail", []):
        print(f"  {line}")

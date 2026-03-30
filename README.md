# Agents for Data Quality — NoiPA

## Introduction

This project implements a multi-agent system for automated data quality analysis applied to NoiPA, the MEF's (Ministero dell'Economia e delle Finanze) digital platform responsible for managing payroll and fiscal obligations for the Italian Public Administration. The goal is to systematically assess and improve the quality of administrative datasets by leveraging a coordinated pipeline of AI-powered agents, each specializing in a distinct dimension of data quality.

The system operates on two real datasets provided by NoiPA. The first, `spesa.csv`, contains 7,543 rows and 18 columns and captures government spending records, including budget lines, expenditure categories, and financial transactions across public sector entities. The second, `attivazioniCessazioni.csv`, contains 20,102 rows and 19 columns and tracks employee activation and cessation events within the PA workforce, covering hire dates, contract types, role assignments, and termination records.

The architecture follows a supervisor-pattern multi-agent design implemented with LangGraph. A central supervisor agent orchestrates the execution flow, delegating tasks to specialized sub-agents and managing a feedback cycle that allows agents to revisit and refine their analyses based on intermediate findings. This design enables modular, traceable, and extensible data quality workflows.

By combining deterministic Python tools with LLM-driven reasoning, the system is able to handle both rule-based checks (e.g., format validation, null detection) and context-sensitive judgments (e.g., semantic inconsistencies, domain-specific anomalies), producing a final reliability score and a remediated dataset ready for downstream analytical use.

## Methods

The pipeline consists of five specialized agents, each responsible for a distinct data quality dimension:

1. **Schema Validation Agent** — Verifies that column names comply with agreed naming conventions (e.g., snake_case, no special characters) and that each column contains values consistent with its declared or inferred data type. Flags mismatches between expected and observed schemas.

2. **Completeness Analysis Agent** — Identifies missing values across all columns, including explicit nulls, empty strings, and common placeholders such as `"N/A"`, `"ND"`, or `"-"`. Also detects sparse columns where the proportion of populated values falls below a configurable threshold.

3. **Consistency Validation Agent** — Validates the internal coherence of the dataset by checking format compliance (e.g., date formats, fiscal codes), enforcing cross-column business rules (e.g., cessation date must follow activation date), and identifying duplicate records based on configurable key combinations.

4. **Anomaly Detection Agent** — Applies statistical methods to detect outliers in numerical columns (e.g., Z-score, IQR-based bounds) and flags unexpected values in categorical columns (e.g., rare or unseen categories, values outside a known domain). Distinguishes between univariate and multivariate anomalies where applicable.

5. **Remediation Agent** — Synthesizes the findings from all upstream agents and applies targeted corrections to the dataset. Corrections include imputation of missing values, normalization of inconsistent formats, removal or flagging of duplicates, and capping or annotation of outliers. The agent produces a cleaned dataset alongside a structured remediation report.

Supporting the agents are 11 deterministic Python tools built to perform precise, reproducible operations: null counters, type checkers, format validators, duplicate detectors, outlier calculators, cross-column rule evaluators, placeholder scanners, schema comparators, imputation routines, deduplication functions, and report serializers. These tools ensure that agent actions are grounded in well-defined logic and remain auditable.

The overall orchestration is managed by a LangGraph supervisor pattern. The supervisor maintains a shared state graph, routes tasks to the appropriate agent at each step, and implements a feedback cycle whereby downstream agents can signal upstream agents to re-examine specific columns or row subsets before the pipeline concludes.

## Experimental Design

The system was evaluated across three distinct dataset runs:

- **spesa.csv**: The real government spending dataset (7,543 rows × 18 columns), used to assess the pipeline's behavior on financial administrative data with real-world noise and inconsistencies.
- **attivazioniCessazioni.csv**: The real employee lifecycle dataset (20,102 rows × 19 columns), used to evaluate the pipeline on HR and workforce data with temporal and relational constraints.
- **Synthetic dataset**: A controlled dataset of approximately 2,000 rows generated using the `Faker` library and custom injection scripts. Ten distinct categories of problems were injected with known ground truth labels, including: missing values, wrong data types, format violations, out-of-range numerical values, duplicate rows, cross-column inconsistencies, placeholder strings, rare categorical values, structural schema deviations, and combined multi-issue rows.

The synthetic dataset enables quantitative evaluation of each agent's detection capabilities. For each agent and each problem category, the following metrics are computed:

- **Precision**: Proportion of flagged items that are true positives.
- **Recall**: Proportion of injected problems that were correctly detected.
- **F1-score**: Harmonic mean of Precision and Recall, used as the primary agent-level performance indicator.

A composite **Reliability Score** is computed for each dataset before and after remediation, defined as a weighted average across four quality dimensions:

```
Reliability Score = 0.15 * Schema + 0.30 * Completeness + 0.35 * Consistency + 0.20 * Anomaly
```

Each dimension score ranges from 0 to 1, where 1 indicates full compliance. The weights reflect the relative criticality of each dimension for the NoiPA use case, with Consistency weighted most heavily due to the legal and administrative implications of incoherent records.

## Results

*(Placeholder — to be filled after running the notebook)*

- Reliability score before and after remediation for each of the three datasets
- F1-score per agent on the synthetic dataset, broken down by injected problem category
- Key issues identified in the real datasets (e.g., most frequent null patterns, dominant inconsistency types, outlier distributions)
- Quantitative comparison of dataset quality before and after the remediation agent's interventions
- Observations on agent agreement and conflict resolution during the feedback cycle

## Conclusions

This project demonstrates that a supervisor-pattern multi-agent architecture, grounded in deterministic tooling and guided by LLM reasoning, offers a scalable and interpretable approach to automated data quality management in public administration contexts. The modular design allows each quality dimension to be assessed independently and improved incrementally, while the LangGraph feedback cycle enables dynamic re-evaluation without requiring a full pipeline restart.

Key limitations include the inherent non-determinism of LLM-based agents, which may produce varying outputs across runs for the same input — a concern in high-stakes administrative settings where reproducibility is essential. Additionally, domain-specific tuning is required for each new dataset type: business rules, naming conventions, and acceptable value ranges must be configured by subject matter experts before deployment.

Future work directions include: extending the system to support real-time monitoring of data streams rather than batch processing; integrating additional NoiPA datasets to broaden coverage; developing a self-calibrating weighting scheme for the reliability score based on historical remediation outcomes; and exploring human-in-the-loop validation steps for ambiguous or high-impact corrections.

---

*Team: LUISS — Whitehall Reply — NoiPA Project*
*AI tools used: Claude (Anthropic) — referenced in code comments*

# Generative AI Usage

This file documents the prompts the team submitted to generative AI tools
during the project; the two prompts below cover (1) the synthetic neutral dataset used as a generalisation test and (2) the visual design of the webapp demo.

---

## 1. Synthetic test dataset (`agents/data/synthetic/orders.csv`)

Used to obtain a CSV for the multi-agent pipeline that does **not** belong
to the four NoiPA fixtures the system was tuned on, so we could measure
generalisation. The prompt below produces 1100 rows of e-commerce order
data with a moderate, well-distributed level of injected noise (~13 % of
rows touched). It is written to be deterministic enough that a re-run
produces a CSV with the same shape and anomaly profile — only the
specific values change.

````
You are a data engineer producing a synthetic test fixture for a data
quality pipeline. Generate a single CSV file `orders.csv` with 1 100 rows
and 13 columns representing e-commerce orders. Output only the CSV content
(header + data rows), nothing else. Use a fixed pseudo-random seed of 42
for reproducibility.

CLEAN SCHEMA
- order_id (string, format "ORD-2025-NNNNNN", unique identifier)
- customer_id (integer, range 1000–9999)
- product_name (string drawn from a fixed pool of about 30 plausible
  retail items spread across the six categories below)
- category (one of: Electronics, Clothing, Home, Books, Sports, Beauty)
- order_date (ISO date YYYY-MM-DD, between 2025-01-01 and 2025-12-31)
- ship_date (ISO date YYYY-MM-DD, typically 0–7 days after order_date)
- quantity (integer, 1–10)
- unit_price (float with two decimals, 5.00–500.00)
- total_price (float, must equal quantity × unit_price within rounding)
- payment_method (one of: Credit Card, PayPal, Bank Transfer,
  Cash on Delivery, Apple Pay)
- status (one of: pending, shipped, delivered, cancelled, returned)
- shipping_country (ISO-2 code from: IT, FR, DE, ES, UK, US, NL, BE, AT, PT)
- customer_age (integer, 18–85)

INJECTED ANOMALIES (intentional noise, ~13 % of rows touched)
- 44 disguised null tokens scattered across product_name, payment_method,
  shipping_country and customer_age. Tokens to use: "N.D.", "?", "-",
  "null", "na", "--".
- 11 IQR outliers in numeric fields: 6 rows with unit_price = 9999.99,
  2 rows with customer_age = 200, 3 rows with quantity = 999.
- 33 mixed-format dates: 12 rows with order_date in EU "DD/MM/YYYY",
  11 rows with ship_date in short EU "DD-MM-YY", 10 rows with order_date
  in Italian textual month form "DD MMM YYYY" using the Italian
  abbreviations GEN, FEB, MAR, APR, MAG, GIU, LUG, AGO, SET, OTT, NOV, DIC.
- 22 wrong-but-parseable values inside numeric columns: 10 rows with
  unit_price like "52.87 EUR" (currency suffix), 7 rows with quantity
  like "3 pcs" (unit suffix), 5 rows with a negative unit_price.
- 7 full-row exact duplicates (each row is a verbatim copy of a different
  random row, so all 13 columns match).
- 20 cross-field inconsistencies: 12 rows with ship_date set 1–5 days
  before order_date, 8 rows with total_price multiplied by a factor of
  2.5–4.0 so it no longer matches quantity × unit_price.
- 3 categorical singleton rare values, each appearing exactly once in the
  whole dataset so the frequency 1/1100 ≈ 0.091 % is below a 0.1 %
  detection threshold: payment_method = "Bitcoin", payment_method =
  "Crypto", category = "Pet".
- 1 column-naming-convention violation: rename the header `customer_age`
  to `Customer Age` (with a literal space) in the CSV header row only.

SHOWCASE WINDOW (first 8 rows)
The first 8 rows must each carry exactly one distinct anomaly so that a
preview of the corrected dataset shows visible variety without scrolling.
Use exactly these placements:

- Row 0: product_name = "N.D." (disguised null).
- Row 1: order_date = "17/02/2025" (EU date format).
- Row 2: unit_price = 9999.99 (IQR outlier).
- Row 3: unit_price = "52.87 EUR" (currency suffix in numeric column).
- Row 4: payment_method = "?" (disguised null with a different token).
- Row 5: ship_date set to 3 days BEFORE order_date (cross-field).
- Row 6: payment_method = "Bitcoin" (rare singleton).
- Row 7: full-row exact duplicate of row 0.

The remaining 133 anomalies must be scattered across rows 8–1099 according
to the totals listed above. Do not place any extra anomaly in rows 0–7.

OUTPUT
A valid RFC 4180 CSV with a single header line followed by 1 100 data
lines. Quote fields that contain commas or whitespace. No comments, no
trailing summary inside the file.
````

---

## 2. Webapp visual design (Claude Design)

Used to produce the React + FastAPI demo's visual layout (the screens at
`webapp/static/screens-intro.jsx`, `screen-pipeline.jsx`,
`screen-results.jsx`, plus the styles in `styles.css`). The prompt was
submitted via the Claude Design preview interface and the resulting code
was then adapted to consume the live SSE stream from the pipeline.

````
# Brief: Interactive Frontend for "Agents for Data Quality"

## Project context

We're building a multi-agent data quality system as a university capstone project (LUISS Machine Learning A.A. 2025/26, partnered with Reply Whitehall). The system takes a CSV file from a user, runs it through 9 sequential pipeline nodes — 5 deterministic Python steps and 4 LLM-powered analysis agents — and returns: a numeric reliability score 0–100, five sub-scores along ISO-8000 dimensions, a corrected CSV, a list of detected issues, and a log of the corrections applied with rationale.

We need an interactive HTML frontend to demo this pipeline live during a 5-minute pitch presentation. The audience is academic (a Machine Learning professor) plus enterprise (Reply consultants). Italian primary language; technical terms in English are fine.

## What the interface must support — five functional moments

**1. Welcome / explainer.** When the user opens the app with no dataset loaded, the interface must communicate (a) what the system does — multi-agent CSV quality assessment + remediation, (b) why it's different from a one-shot LLM call — deterministic-first with surgical LLM reasoning, (c) entry points: upload a CSV, or try with a built-in demo dataset (NoiPA tax data from the Italian Ministry of Economy).

**2. Dataset preview.** Once a CSV is loaded, the interface shows: filename, row count, column count, distribution of column types (numeric vs string vs date), and a small preview of the first rows. From here the user triggers the pipeline.

**3. Live pipeline execution.** While the pipeline runs, the user must see real-time progress through the 9 nodes:
- 5 deterministic nodes: ingest, discover (rule discovery), audit (9 quality tools), remediation (apply fixes), supervisor (compute final score)
- 4 LLM agents — Schema, Completeness, Consistency, Anomaly — each makes one LLM call to plan fixes for issues in its dimension
The visual must distinguish "deterministic Python step" from "LLM agent reasoning step" so the audience grasps that the system is hybrid, not pure-LLM. Each node updates with: status (pending/running/done), elapsed time, and a one-line outcome message (e.g. "29 issues found across 4 categories" or "Schema agent: 5 actions planned, validity score 0.9").

**4. Results dashboard.** When done, the interface must display:
- **Reliability score** as the headline number (0–100) with an immediate quality verdict (high ≥70, medium 40–69, low <40)
- **Five sub-scores**: validity (weight 20%), completeness (30%), consistency (25%), uniqueness (15%), accuracy (10%) — each 0–100
- **Severity breakdown** of detected issues (count of critical / high / medium / low)
- **Issues list**: each row has tool name, issue type, severity, affected columns, row count, message
- **Correction log**: each row has the agent that proposed it (Schema/Completeness/Consistency/Anomaly), the action applied (impute_median, drop_duplicates, normalize_dates, ignore, etc.), the affected column(s), the rows changed, and the rationale (one short sentence)
- **Fixed dataset preview**: first rows of the corrected CSV
- **Audit trail**: the full sequence of node-level messages from the run

**5. Export.** The user can download: the fixed CSV, an HTML report (a separate self-contained page, not part of this UI), the correction log as JSON.

## Sample data shapes (so you understand what's being shown)

```json
{
  "reliability_score": 54.0,
  "sub_scores": {"validity": 90, "completeness": 0, "consistency": 90, "uniqueness": 90, "accuracy": 0},
  "severity_breakdown": {"critical": 1, "high": 16, "medium": 6, "low": 6},
  "issues": [
    {"tool": "check_nulls", "issue_type": "missing_mandatory_values", "severity": "critical",
     "columns": ["spesa"], "row_count": 1582,
     "message": "Column `spesa` has 1582 effectively missing values (21.0%)"}
  ],
  "correction_log": [
    {"agent": "Completeness", "action": "impute_median", "column": "spesa",
     "rows_affected": 1582, "rationale": "21% missing on critical column; median preserves distribution"}
  ],
  "audit_trail": [
    "ingest: loaded `spesa` (7,543×18)",
    "Schema: 5 actions planned, score=0.9",
    "supervisor: reliability=54.0/100"
  ]
}
```

## Tech constraints

- Single HTML page that runs in a modern browser, no build step
- Vanilla JS or minimal libraries via CDN (Plotly, Tailwind, etc. are fine)
- Mock data is acceptable — the UI doesn't need to actually call an API for the prototype, but should be structured so a real backend could later serve the JSON shapes shown above
- Should look professional enough to present to consultants and a professor — but minimal and confident, not flashy or game-like
- The interface must work on a laptop screen during a live pitch (1280×800 minimum)

## Positioning / tone

- Hybrid AI system, not "yet another LLM tool". The visual narrative should make the deterministic layer visible — this is part of the value proposition.
- Trustworthy, technical, calm. No emoji-heavy gamification. No vague "AI magic" language.
- The reliability score is the moment of truth. It must land with weight when revealed.

## Out of scope

- User authentication, accounts, history of past runs
- Multi-file batch processing
- A/B testing or feature flags
- Mobile layout (laptop only)
- Marketing landing page features (testimonials, pricing, FAQ)
- Settings/admin panels

## What I want from you

A working interactive HTML prototype that supports the five moments above, with realistic mock data so I can walk through the full flow during a pitch demo. Show me how you'd structure the experience — the layout, components, interactions, transitions are your design choice. I'll tell you what to refine after seeing the first version.
````

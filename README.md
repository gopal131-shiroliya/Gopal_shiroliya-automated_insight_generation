# Assignment 4 — Automated Insight Generation

A general-purpose Python Auto-Analytics Engine for district-level healthcare performance data.

## What it does

- Loads and validates CSV data with Pandas
- Reports missing values and basic data information
- Provides live filters for district, month and indicator
- Detects month-to-month trends using configurable percentage-change thresholds
- Detects outliers using configurable IQR or Z-score methods
- Computes Pearson correlation matrix and flags strong pairs
- Generates structured, human-readable insights dynamically
- Assigns Low / Medium / High severity from observed magnitude relative to configurable thresholds
- Displays insight tables, severity counts, correlation heatmap and district trend lines
- Allows CSV downloads from the UI

## Technical constraints followed

- Backend: Python only
- Frontend: Streamlit
- No separate backend API
- Pandas / NumPy / Plotly
- Thresholds are configurable through UI sliders

## Project structure

```text
automated-insight-generation/
├── app.py
├── analytics.py
├── data/
│   └── healthcare_data.csv
├── outputs/
├── requirements.txt
├── README.md
└── sample_insights.json
```

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The browser should open the Streamlit dashboard.

## Expected sample behavior

With the default dataset and a 10% trend threshold:

- Ahmedabad ANC coverage changes from 85 to 69, approximately -18.8%, so a trend insight is generated.
- Mehsana ANC coverage of 42 is an IQR outlier under the default configuration.
- Mehsana high-risk cases change from 11 to 28, generating a strong trend insight.

The exact set of insights can change when filters or thresholds are changed.

## Insight schema

Each generated insight contains:

- `insight_id`
- `type`
- `indicator`
- `entity`
- `period`
- `value`
- `prev_value`
- `change_pct`
- `severity`
- `explanation`

Supported types:

- `trend`
- `outlier`
- `correlation`
- `threshold_breach`

## Severity design

Severity is not based on a fixed raw number such as:

```python
if change > 50:
    severity = "High"
```

Instead, the observed magnitude is normalized against a configurable threshold.

For trends:

```text
ratio = abs(change_pct) / trend_threshold
```

For correlations:

```text
ratio = abs(r) / correlation_threshold
```

For IQR outliers, the distance outside the IQR boundary is normalized by the IQR.

The UI exposes the Medium and High multiplier bands.

## Correlation limitation

The supplied sample contains only 2 months × 6 districts = 12 rows. Pearson correlations calculated from this small sample are fragile and should be treated as indicative rather than stable evidence. The dashboard shows the matrix honestly and does not imply causation.

For more stable correlation analysis, use more months and more districts.

## General-purpose / no hardcoded narrative

No district-specific sentence is hardcoded. Narrative templates use the actual values, district, indicator and period produced by the analytics engine.

For example, the engine can generate:

> ANC Coverage in Ahmedabad decreased by 18.8% compared to 2026-07. The observed change exceeds the configured 10.0% threshold.

The same code works for another district or indicator without changing the source code.

## Output files

The Streamlit UI provides:

- `insights.csv`
- `correlation_matrix.csv`
- `filtered_healthcare_data.csv`

## Important note about threshold_breach

The assignment requires support for an insight type named `threshold_breach`, but it does not provide domain-specific medical threshold values. This implementation therefore does not invent medical limits. The type is supported in the normalized insight schema and can be enabled later by supplying explicit domain thresholds through configuration.

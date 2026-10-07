import pandas as pd
import streamlit as st
import plotly.express as px

from analytics import (
    REQUIRED_COLUMNS,
    INDICATORS,
    load_and_validate,
    generate_insights,
    build_correlation_matrix,
)

st.set_page_config(
    page_title="Automated Insight Generation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Automated Insight Generation Engine")
st.caption(
    "General-purpose district-level healthcare analytics: trends, outliers, "
    "correlations and data-driven insights."
)

@st.cache_data
def load_default_data():
    return load_and_validate("data/healthcare_data.csv")

# -----------------------------
# Sidebar: input + thresholds
# -----------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload healthcare CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df, validation = load_and_validate(uploaded_file)
    except Exception as exc:
        st.error(f"Could not load the uploaded CSV: {exc}")
        st.stop()
else:
    df, validation = load_default_data()

if validation["missing_columns"]:
    st.error(
        "Missing required columns: "
        + ", ".join(validation["missing_columns"])
    )
    st.stop()

if validation["duplicate_rows"] > 0:
    st.warning(
        f"Found {validation['duplicate_rows']} duplicate rows. "
        "Duplicate rows are removed for analytics."
    )

st.sidebar.subheader("Filters")

districts = sorted(df["district"].dropna().unique().tolist())
selected_districts = st.sidebar.multiselect(
    "District",
    options=districts,
    default=districts,
)

month_values = sorted(df["month"].dropna().dt.strftime("%Y-%m").unique().tolist())
selected_months = st.sidebar.multiselect(
    "Month",
    options=month_values,
    default=month_values,
)

selected_indicators = st.sidebar.multiselect(
    "Indicator",
    options=INDICATORS,
    default=INDICATORS,
)

st.sidebar.subheader("Detection thresholds")

trend_threshold = st.sidebar.slider(
    "Trend threshold (%)",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=0.5,
)

correlation_threshold = st.sidebar.slider(
    "Correlation threshold |r|",
    min_value=0.10,
    max_value=1.00,
    value=0.70,
    step=0.05,
)

outlier_method = st.sidebar.radio(
    "Outlier method",
    ["IQR", "Z-score"],
)

iqr_multiplier = st.sidebar.slider(
    "IQR multiplier",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
)

z_threshold = st.sidebar.slider(
    "Z-score threshold",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.1,
)

st.sidebar.subheader("Severity configuration")

medium_multiplier = st.sidebar.slider(
    "Medium starts at × trend threshold",
    min_value=1.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

high_multiplier = st.sidebar.slider(
    "High starts at × trend threshold",
    min_value=1.1,
    max_value=5.0,
    value=1.5,
    step=0.1,
)

if high_multiplier <= medium_multiplier:
    st.sidebar.error("High multiplier must be greater than Medium multiplier.")
    st.stop()

# -----------------------------
# Filter data
# -----------------------------
filtered = df[
    df["district"].isin(selected_districts)
    & df["month"].dt.strftime("%Y-%m").isin(selected_months)
].copy()

# -----------------------------
# Validation / data overview
# -----------------------------
with st.expander("🔎 Data validation & preview", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", int(df.isna().sum().sum()))

    st.write("Missing-value count per column")
    st.dataframe(
        validation["missing_counts"].rename("missing_count"),
        use_container_width=True,
    )

    st.write("First 5 rows")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Data info")
    info_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
        }
    )
    st.dataframe(info_df, use_container_width=True)

# -----------------------------
# Generate analytics
# -----------------------------
insights = generate_insights(
    filtered,
    selected_indicators=selected_indicators,
    trend_threshold=trend_threshold,
    outlier_method=outlier_method,
    iqr_multiplier=iqr_multiplier,
    z_threshold=z_threshold,
    correlation_threshold=correlation_threshold,
    medium_multiplier=medium_multiplier,
    high_multiplier=high_multiplier,
)

correlation_matrix = build_correlation_matrix(filtered, selected_indicators)

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("Overview")

total = len(insights)
high = sum(x["severity"] == "High" for x in insights)
medium = sum(x["severity"] == "Medium" for x in insights)
low = sum(x["severity"] == "Low" for x in insights)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total insights", total)
c2.metric("🔴 High", high)
c3.metric("🟠 Medium", medium)
c4.metric("🟢 Low", low)

# -----------------------------
# Insight list
# -----------------------------
st.subheader("💡 Generated Insights")

if insights:
    insights_df = pd.DataFrame(insights)

    display_cols = [
        "insight_id",
        "type",
        "indicator",
        "entity",
        "period",
        "value",
        "prev_value",
        "change_pct",
        "severity",
        "explanation",
    ]
    st.dataframe(
        insights_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = insights_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download insights.csv",
        data=csv_bytes,
        file_name="insights.csv",
        mime="text/csv",
    )
else:
    st.info(
        "No insights were flagged for the current filters and thresholds. "
        "Try widening the filters or lowering the detection thresholds."
    )

# -----------------------------
# Severity bar chart
# -----------------------------
st.subheader("📊 Severity Counts")

severity_df = pd.DataFrame(
    {
        "severity": ["Low", "Medium", "High"],
        "count": [low, medium, high],
    }
)

fig_severity = px.bar(
    severity_df,
    x="severity",
    y="count",
    text="count",
    title="Insight count by severity",
)
fig_severity.update_layout(yaxis_title="Number of insights")
st.plotly_chart(fig_severity, use_container_width=True)

# -----------------------------
# Correlation heatmap
# -----------------------------
st.subheader("🔥 Pearson Correlation Heatmap")

if correlation_matrix.shape[0] >= 2:
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title="Correlation matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    corr_csv = correlation_matrix.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download correlation_matrix.csv",
        data=corr_csv,
        file_name="correlation_matrix.csv",
        mime="text/csv",
    )
else:
    st.info("Select at least two indicators to display the correlation matrix.")

st.caption(
    "Limitation: the supplied sample contains only 2 months × 6 districts "
    "(12 rows), so Pearson correlations are indicative rather than stable "
    "evidence of real-world relationships."
)

# -----------------------------
# Per-district line chart
# -----------------------------
st.subheader("📈 Per-District Indicator Trend")

chart_indicators = selected_indicators or INDICATORS
chart_indicator = st.selectbox(
    "Choose indicator for line chart",
    chart_indicators,
)

chart_df = filtered.sort_values(["district", "month"]).copy()

if not chart_df.empty:
    fig_line = px.line(
        chart_df,
        x="month",
        y=chart_indicator,
        color="district",
        markers=True,
        title=f"{chart_indicator} by district",
    )
    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title=chart_indicator,
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("No rows match the selected filters.")

# -----------------------------
# Download current data
# -----------------------------
st.subheader("📥 Current filtered dataset")
st.dataframe(filtered, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download filtered dataset",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_healthcare_data.csv",
    mime="text/csv",
)

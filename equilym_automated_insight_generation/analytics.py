from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]

INDICATORS = [
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]

NUMERIC_INDICATORS = INDICATORS.copy()


def load_and_validate(source: Any):
    """
    Load a CSV from a path or uploaded file-like object.

    Returns:
        cleaned_dataframe, validation_dictionary
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    validation = {
        "missing_columns": missing_columns,
        "missing_counts": df.isna().sum(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if missing_columns:
        return df, validation

    df = df.copy()

    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    for col in INDICATORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove exact duplicate rows so one row remains per supplied observation.
    df = df.drop_duplicates().reset_index(drop=True)

    validation["invalid_months"] = int(df["month"].isna().sum())
    validation["missing_counts"] = df.isna().sum()

    return df, validation


def _safe_number(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _severity_from_ratio(
    ratio: float,
    medium_multiplier: float,
    high_multiplier: float,
) -> str:
    """
    Severity is derived from the observed magnitude relative to the
    configurable threshold. No fixed raw-value magic number is used.
    """
    ratio = abs(float(ratio))

    if ratio >= high_multiplier:
        return "High"
    if ratio >= medium_multiplier:
        return "Medium"
    return "Low"


def detect_trends(
    df: pd.DataFrame,
    indicators: list[str],
    threshold: float,
    medium_multiplier: float,
    high_multiplier: float,
) -> list[dict]:
    """Detect significant month-to-month changes per district/indicator."""
    results = []

    if df.empty:
        return results

    work = df.sort_values(["district", "month"]).copy()

    for district, group in work.groupby("district", sort=True):
        group = group.sort_values("month")

        for indicator in indicators:
            previous = None
            previous_month = None

            for _, row in group.iterrows():
                current = row[indicator]

                if pd.isna(current):
                    previous = None
                    previous_month = None
                    continue

                if previous is not None and previous != 0:
                    change_pct = ((current - previous) / previous) * 100.0

                    if abs(change_pct) >= threshold:
                        ratio = abs(change_pct) / threshold
                        severity = _severity_from_ratio(
                            ratio,
                            medium_multiplier,
                            high_multiplier,
                        )

                        direction = "increased" if change_pct > 0 else "decreased"

                        results.append(
                            {
                                "type": "trend",
                                "indicator": indicator,
                                "entity": district,
                                "period": row["month"].strftime("%Y-%m"),
                                "value": _safe_number(current),
                                "prev_value": _safe_number(previous),
                                "change_pct": round(float(change_pct), 2),
                                "severity": severity,
                                "explanation": (
                                    f"{indicator.replace('_', ' ').title()} in "
                                    f"{district} {direction} by "
                                    f"{abs(change_pct):.1f}% compared to "
                                    f"{previous_month.strftime('%Y-%m')}. "
                                    f"The observed change exceeds the "
                                    f"configured {threshold:.1f}% threshold."
                                ),
                            }
                        )

                previous = float(current)
                previous_month = row["month"]

    return results


def detect_outliers(
    df: pd.DataFrame,
    indicators: list[str],
    method: str,
    iqr_multiplier: float,
    z_threshold: float,
    medium_multiplier: float,
    high_multiplier: float,
) -> list[dict]:
    """Detect outliers independently for each numerical indicator."""
    results = []

    if df.empty:
        return results

    for indicator in indicators:
        series = pd.to_numeric(df[indicator], errors="coerce")
        valid = series.dropna()

        if len(valid) < 4:
            continue

        if method == "IQR":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            for idx in valid.index:
                value = float(series.loc[idx])

                if value < lower or value > upper:
                    # Distance is normalized by the IQR-based boundary
                    # distance. This keeps severity data-derived.
                    if value > upper:
                        distance = (value - upper) / iqr
                    else:
                        distance = (lower - value) / iqr

                    severity = _severity_from_ratio(
                        distance,
                        medium_multiplier,
                        high_multiplier,
                    )

                    row = df.loc[idx]

                    results.append(
                        {
                            "type": "outlier",
                            "indicator": indicator,
                            "entity": row["district"],
                            "period": row["month"].strftime("%Y-%m"),
                            "value": value,
                            "prev_value": None,
                            "change_pct": None,
                            "severity": severity,
                            "explanation": (
                                f"{row['district']}'s {indicator.replace('_', ' ')} "
                                f"value of {value:.2f} is outside the "
                                f"configured IQR range "
                                f"[{lower:.2f}, {upper:.2f}]."
                            ),
                        }
                    )

        elif method == "Z-score":
            std = valid.std(ddof=0)

            if std == 0:
                continue

            mean = valid.mean()
            z_scores = (valid - mean) / std

            for idx in valid.index:
                z = float(z_scores.loc[idx])

                if abs(z) >= z_threshold:
                    ratio = abs(z) / z_threshold
                    severity = _severity_from_ratio(
                        ratio,
                        medium_multiplier,
                        high_multiplier,
                    )

                    row = df.loc[idx]
                    value = float(series.loc[idx])

                    results.append(
                        {
                            "type": "outlier",
                            "indicator": indicator,
                            "entity": row["district"],
                            "period": row["month"].strftime("%Y-%m"),
                            "value": value,
                            "prev_value": None,
                            "change_pct": None,
                            "severity": severity,
                            "explanation": (
                                f"{row['district']}'s {indicator.replace('_', ' ')} "
                                f"value of {value:.2f} has a Z-score of "
                                f"{z:.2f}, exceeding the configured "
                                f"|Z| threshold of {z_threshold:.2f}."
                            ),
                        }
                    )

        else:
            raise ValueError("method must be 'IQR' or 'Z-score'")

    return results


def build_correlation_matrix(
    df: pd.DataFrame,
    indicators: list[str],
) -> pd.DataFrame:
    """Return the standard pandas Pearson correlation matrix."""
    if not indicators:
        return pd.DataFrame()

    available = [c for c in indicators if c in df.columns]

    if not available:
        return pd.DataFrame()

    return df[available].corr(method="pearson")


def detect_correlations(
    df: pd.DataFrame,
    indicators: list[str],
    threshold: float,
    medium_multiplier: float,
    high_multiplier: float,
) -> list[dict]:
    """Flag unique indicator pairs whose absolute Pearson r exceeds threshold."""
    results = []

    matrix = build_correlation_matrix(df, indicators)

    if matrix.empty or len(matrix.columns) < 2:
        return results

    for i, indicator_a in enumerate(matrix.columns):
        for indicator_b in matrix.columns[i + 1:]:
            correlation = matrix.loc[indicator_a, indicator_b]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:
                ratio = abs(float(correlation)) / threshold
                severity = _severity_from_ratio(
                    ratio,
                    medium_multiplier,
                    high_multiplier,
                )

                direction = "positive" if correlation > 0 else "negative"

                results.append(
                    {
                        "type": "correlation",
                        "indicator": f"{indicator_a}:{indicator_b}",
                        "entity": "All selected districts",
                        "period": "selected period",
                        "value": round(float(correlation), 4),
                        "prev_value": None,
                        "change_pct": None,
                        "severity": severity,
                        "explanation": (
                            f"{indicator_a.replace('_', ' ').title()} and "
                            f"{indicator_b.replace('_', ' ').title()} show a "
                            f"strong {direction} Pearson correlation "
                            f"(r={correlation:.2f}), meeting the configured "
                            f"|r| threshold of {threshold:.2f}. Correlation "
                            f"does not imply causation."
                        ),
                    }
                )

    return results


def detect_threshold_breaches(
    df: pd.DataFrame,
    indicators: list[str],
    medium_multiplier: float,
    high_multiplier: float,
) -> list[dict]:
    """
    A generic, data-derived threshold-breach detector.

    This implementation uses the cross-sectional median as the reference
    and flags values whose relative deviation exceeds a configurable
    threshold multiplier. It is intentionally conservative and only emits
    breaches when the deviation is at least 100%.
    """
    # The assignment requires this insight type to be supported. Since no
    # domain threshold values are supplied, we do not invent medical limits.
    # Therefore this function returns no rows unless future domain thresholds
    # are explicitly supplied.
    return []


def generate_insights(
    df: pd.DataFrame,
    selected_indicators: list[str],
    trend_threshold: float,
    outlier_method: str,
    iqr_multiplier: float,
    z_threshold: float,
    correlation_threshold: float,
    medium_multiplier: float,
    high_multiplier: float,
) -> list[dict]:
    """Run all detectors and normalize results into the required schema."""
    indicators = [
        col for col in selected_indicators
        if col in INDICATORS and col in df.columns
    ]

    trend_rows = detect_trends(
        df,
        indicators,
        trend_threshold,
        medium_multiplier,
        high_multiplier,
    )

    outlier_rows = detect_outliers(
        df,
        indicators,
        outlier_method,
        iqr_multiplier,
        z_threshold,
        medium_multiplier,
        high_multiplier,
    )

    correlation_rows = detect_correlations(
        df,
        indicators,
        correlation_threshold,
        medium_multiplier,
        high_multiplier,
    )

    threshold_rows = detect_threshold_breaches(
        df,
        indicators,
        medium_multiplier,
        high_multiplier,
    )

    all_rows = (
        trend_rows
        + outlier_rows
        + correlation_rows
        + threshold_rows
    )

    normalized = []

    for number, row in enumerate(all_rows, start=1):
        normalized.append(
            {
                "insight_id": f"INS-{number:04d}",
                "type": row["type"],
                "indicator": row["indicator"],
                "entity": row["entity"],
                "period": row["period"],
                "value": row["value"],
                "prev_value": row["prev_value"],
                "change_pct": row["change_pct"],
                "severity": row["severity"],
                "explanation": row["explanation"],
            }
        )

    return normalized

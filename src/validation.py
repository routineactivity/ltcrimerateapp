from __future__ import annotations

import pandas as pd


def validate_monthly_clean(
    df: pd.DataFrame,
    *,
    require_columns: tuple[str, ...] = ("GLOBALID", "MajorText", "MinorText", "WardName", "WardCode"),
    month_min: int = 201004,
    month_max: int = 202601,
) -> None:
    """
    Validate ward-level monthly output (crime__monthly_clean.csv).
    Raises ValueError with a clear message if something is wrong.
    """
    missing = [c for c in require_columns if c not in df.columns]
    if missing:
        raise ValueError(f"monthly_clean missing required columns: {missing}")

    # GLOBALID must be present
    if df["GLOBALID"].isna().any():
        raise ValueError("monthly_clean has missing GLOBALID values")

    # Ensure there are monthly columns in expected range
    month_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit() and len(c) == 6]
    if not month_cols:
        raise ValueError("monthly_clean has no yyyymm monthly columns")

    # Check month bounds
    month_ints = sorted(int(c) for c in month_cols)
    if month_ints[0] > month_min or month_ints[-1] < month_max:
        raise ValueError(
            f"monthly_clean month coverage unexpected. "
            f"Found {month_ints[0]}..{month_ints[-1]} expected to include {month_min}..{month_max}"
        )

    # Monthly values should be numeric and non-negative
    vals = df[month_cols]
    # allow ints/floats; coerce check
    if vals.isna().any().any():
        raise ValueError("monthly_clean has NaN values in monthly columns")
    if (vals < 0).any().any():
        raise ValueError("monthly_clean has negative values in monthly columns")


def validate_constituency_counts(
    df: pd.DataFrame,
    *,
    min_year: int = 2011,
    max_year: int = 2025,
) -> None:
    """
    Validate constituency-year counts output.
    Expected columns: PCON24CD, PCON24NM, MajorText, MinorText, year, crime_count
    """
    required = ["PCON24CD", "PCON24NM", "MajorText", "MinorText", "year", "crime_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"constituency_counts missing required columns: {missing}")

    if df["PCON24CD"].isna().any():
        raise ValueError("constituency_counts has missing PCON24CD")
    if df["year"].isna().any():
        raise ValueError("constituency_counts has missing year")
    if df["crime_count"].isna().any():
        raise ValueError("constituency_counts has missing crime_count")

    y_min = int(df["year"].min())
    y_max = int(df["year"].max())
    if y_min < min_year or y_max > max_year:
        raise ValueError(f"constituency_counts has years {y_min}..{y_max} but expected {min_year}..{max_year}")

    if (df["crime_count"] < 0).any():
        raise ValueError("constituency_counts has negative crime_count")


def validate_constituency_rates(
    df: pd.DataFrame,
    *,
    min_year: int = 2011,
    max_year: int = 2025,
) -> None:
    """
    Validate constituency-year rates output.
    Expected columns: PCON24CD, year, crime_count, population, rate_per_1000
    """
    required = ["PCON24CD", "year", "crime_count", "population", "rate_per_1000"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"constituency_rates missing required columns: {missing}")

    if df["PCON24CD"].isna().any():
        raise ValueError("constituency_rates has missing PCON24CD")

    y_min = int(df["year"].min())
    y_max = int(df["year"].max())
    if y_min < min_year or y_max > max_year:
        raise ValueError(f"constituency_rates has years {y_min}..{y_max} but expected {min_year}..{max_year}")

    if df["population"].isna().any():
        raise ValueError("constituency_rates has missing population values")
    if (df["population"] <= 0).any():
        raise ValueError("constituency_rates has non-positive population values")

    if df["rate_per_1000"].isna().any():
        raise ValueError("constituency_rates has missing rate_per_1000 values")

    if (df["crime_count"] < 0).any():
        raise ValueError("constituency_rates has negative crime_count")
    if (df["rate_per_1000"] < 0).any():
        raise ValueError("constituency_rates has negative rate_per_1000")
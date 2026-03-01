# processing.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

MONTH_COL_PATTERN = re.compile(r"^\d{6}$")  # yyyymm


def _find_month_cols(df: pd.DataFrame, start: int = 201004, end: int = 202601) -> list[str]:
    month_cols: list[str] = []
    for c in df.columns:
        if isinstance(c, str) and MONTH_COL_PATTERN.match(c):
            v = int(c)
            if start <= v <= end:
                month_cols.append(c)
    return sorted(month_cols, key=lambda x: int(x))


def build_crime_monthly_clean(
    wardcrime: pd.DataFrame,
    wards: pd.DataFrame,
    *,
    crime_wardcode_col: str = "WardCode",
    wards_wardcode_col: str = "WardCode",
    globalid_col: str = "GLOBALID",
    wardname_col: str = "WardName",
    majortext_col: str = "MajorText",
    minortext_col: str = "MinorText",
    month_start: int = 201004,
    month_end: int = 202601,
) -> pd.DataFrame:
    # Identify month columns
    month_cols = _find_month_cols(wardcrime, start=month_start, end=month_end)
    if not month_cols:
        raise ValueError(f"No month cols found between {month_start} and {month_end}.")

    # Coerce month cols to numeric
    wc = wardcrime.copy()
    for c in month_cols:
        wc[c] = pd.to_numeric(wc[c], errors="coerce").fillna(0)

    # Keep only needed columns from crime (explicitly ignore any crime WardName)
    wc = wc[[crime_wardcode_col, majortext_col, minortext_col] + month_cols]

    # Prepare ward lookup (WardCode -> GLOBALID, plus metadata)
    wl = wards[[globalid_col, wards_wardcode_col, wardname_col]].drop_duplicates()

    # Join GLOBALID onto crime by WardCode
    wc = wc.merge(
        wl[[globalid_col, wards_wardcode_col]],
        how="left",
        left_on=crime_wardcode_col,
        right_on=wards_wardcode_col,
        validate="m:1",
    )

    if wards_wardcode_col != crime_wardcode_col:
        wc = wc.drop(columns=[wards_wardcode_col])

    # Fail fast if any WardCode didn’t match
    if wc[globalid_col].isna().any():
        n = int(wc[globalid_col].isna().sum())
        raise ValueError(f"{n} crime rows did not match a GLOBALID. Check WardCode formatting.")

    # Aggregate by GLOBALID (NOT by ward name)
    group_cols = [globalid_col, majortext_col, minortext_col]
    agg = wc.groupby(group_cols, as_index=False)[month_cols].sum()

    # Add WardName + WardCode from wardlist via GLOBALID
    meta = wl.drop_duplicates(subset=[globalid_col])[[globalid_col, wardname_col, wards_wardcode_col]]
    out = agg.merge(meta, how="left", on=globalid_col, validate="m:1")

    if wards_wardcode_col != "WardCode":
        out = out.rename(columns={wards_wardcode_col: "WardCode"})

    # Order columns
    out = out[[globalid_col, majortext_col, minortext_col] + month_cols + [wardname_col, "WardCode"]]
    return out

def add_pcon_geography(
    crime_monthly: pd.DataFrame,
    ward_pcon_lookup: pd.DataFrame,
    *,
    crime_globalid_col: str = "GLOBALID",
    lookup_globalid_col: str = "GlobalID",
    pcon_cd_col: str = "PCON24CD",
    pcon_nm_col: str = "PCON24NM",
) -> pd.DataFrame:
    needed = {lookup_globalid_col, pcon_cd_col, pcon_nm_col}
    missing = needed - set(ward_pcon_lookup.columns)
    if missing:
        raise KeyError(f"ward_pcon_lookup missing columns: {sorted(missing)}")

    lk = (
        ward_pcon_lookup[[lookup_globalid_col, pcon_cd_col, pcon_nm_col]]
        .drop_duplicates(subset=[lookup_globalid_col])
        .rename(columns={lookup_globalid_col: crime_globalid_col})
    )

    out = crime_monthly.merge(lk, how="left", on=crime_globalid_col, validate="m:1")
    if out[pcon_cd_col].isna().any():
        n = int(out[pcon_cd_col].isna().sum())
        raise ValueError(f"{n} rows did not match to a PCON via GlobalID/GLOBALID.")
    return out


def monthly_to_constituency_year(
    crime_monthly_with_pcon: pd.DataFrame,
    *,
    month_start: int = 201004,
    month_end: int = 202601,
    min_year: int = 2011,
    max_year: int = 2025,
    pcon_cd_col: str = "PCON24CD",
    pcon_nm_col: str = "PCON24NM",
    majortext_col: str = "MajorText",
    minortext_col: str = "MinorText",
) -> pd.DataFrame:
    """
    Converts monthly yyyymm columns into constituency-year counts.
    Drops years outside [min_year, max_year].
    """

    month_cols = _find_month_cols(crime_monthly_with_pcon, start=month_start, end=month_end)
    if not month_cols:
        raise ValueError("No monthly yyyymm columns found to aggregate.")

    id_cols = [pcon_cd_col, pcon_nm_col, majortext_col, minortext_col]

    long = crime_monthly_with_pcon[id_cols + month_cols].melt(
        id_vars=id_cols,
        value_vars=month_cols,
        var_name="yyyymm",
        value_name="count",
    )

    long["count"] = pd.to_numeric(long["count"], errors="coerce").fillna(0)
    long["year"] = long["yyyymm"].str.slice(0, 4).astype(int)

    # DROP 2010 and 2026
    long = long[(long["year"] >= min_year) & (long["year"] <= max_year)]

    annual = (
        long.groupby(id_cols + ["year"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "crime_count"})
    )

    return annual


def make_rates_per_1000(
    crime_const_year: pd.DataFrame,
    pcon_population: pd.DataFrame,
    *,
    pcon_cd_col: str = "PCON24CD",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Joins population estimates and computes crime rate per 1,000 population.

    Population input format expected:
      - PCON24CD column
      - cy2011, cy2012, ... columns (calendar year population)
    """
    if pcon_cd_col not in pcon_population.columns:
        raise KeyError(f"Population file missing {pcon_cd_col}")

    pop_year_cols = [c for c in pcon_population.columns if isinstance(c, str) and c.startswith("cy") and c[2:].isdigit()]
    if not pop_year_cols:
        raise ValueError("No population columns found like cy2011, cy2012, ...")

    pop_long = pcon_population[[pcon_cd_col] + pop_year_cols].melt(
        id_vars=[pcon_cd_col],
        value_vars=pop_year_cols,
        var_name="cy",
        value_name="population",
    )
    pop_long["year"] = pop_long["cy"].str.replace("cy", "", regex=False).astype(int)
    pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")

    out = crime_const_year.merge(
        pop_long[[pcon_cd_col, "year", "population"]],
        how="left",
        left_on=[pcon_cd_col, year_col],
        right_on=[pcon_cd_col, "year"],
        validate="m:1",
    )

    # If population missing (e.g., 2010 or 2026), rate becomes NaN
    out["rate_per_1000"] = (out["crime_count"] / out["population"]) * 1000

    return out


def run_constituency_pipeline(
    crime_monthly_clean_path: str | Path,
    ward_pcon_lookup_path: str | Path,
    pcon_population_path: str | Path,
    *,
    out_counts_path: str | Path | None = None,
    out_rates_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end:
      crime__monthly_clean.csv -> constituency-year counts -> constituency-year rates per 1000
    """
    crime_monthly = pd.read_csv(crime_monthly_clean_path)
    ward_pcon = pd.read_csv(ward_pcon_lookup_path)
    pop = pd.read_csv(pcon_population_path)

    crime_with_pcon = add_pcon_geography(crime_monthly, ward_pcon)
    crime_const_year = monthly_to_constituency_year(crime_with_pcon)
    crime_const_year_rates = make_rates_per_1000(crime_const_year, pop)

    if out_counts_path is not None:
        Path(out_counts_path).parent.mkdir(parents=True, exist_ok=True)
        crime_const_year.to_csv(out_counts_path, index=False)

    if out_rates_path is not None:
        Path(out_rates_path).parent.mkdir(parents=True, exist_ok=True)
        crime_const_year_rates.to_csv(out_rates_path, index=False)

    return crime_const_year, crime_const_year_rates
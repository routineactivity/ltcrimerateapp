from pathlib import Path
import pandas as pd

from logging_config import setup_logging
from validation import (
    validate_monthly_clean,
    validate_constituency_counts,
    validate_constituency_rates,
)

from data_processing import (
    build_crime_monthly_clean,
    run_constituency_pipeline,
)


def main() -> None:
    log = setup_logging()  # creates console + file logger

    # Project root (…/ltcrimerates)
    BASE_DIR = Path(__file__).resolve().parents[1]

    raw_dir = BASE_DIR / "data" / "raw"
    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: ward-level monthly clean
    wardcrime_path = raw_dir / "wardcrime_20102026.csv"
    wards_path = raw_dir / "wardlistwithglobalid.csv"
    monthly_out_path = processed_dir / "crime__monthly_clean.csv"

    log.info(f"Stage 1 starting. Reading: {wardcrime_path} and {wards_path}")
    wardcrime = pd.read_csv(wardcrime_path)
    wards = pd.read_csv(wards_path)

    crime__monthly_clean = build_crime_monthly_clean(wardcrime, wards)
    validate_monthly_clean(crime__monthly_clean)  # fail fast if bad

    crime__monthly_clean.to_csv(monthly_out_path, index=False)
    log.info(f"Stage 1 complete. Wrote: {monthly_out_path} (rows={len(crime__monthly_clean):,}, cols={len(crime__monthly_clean.columns):,})")

    # Optional preview
    log.info("Stage 1 preview:\n%s", crime__monthly_clean.head().to_string(index=False))

    # ---- Stage 2: constituency-year counts + rates (2011–2025 only)
    ward_pcon_lookup_path = raw_dir / "ward_pcon_lookup.csv"
    pcon_population_path = raw_dir / "pcon_population_data.csv"

    counts_out_path = processed_dir / "crime_constituency_year_counts.csv"
    rates_out_path = processed_dir / "crime_constituency_year_rates_per_1000.csv"

    log.info(f"Stage 2 starting. Reading: {ward_pcon_lookup_path} and {pcon_population_path}")

    counts, rates = run_constituency_pipeline(
        crime_monthly_clean_path=monthly_out_path,
        ward_pcon_lookup_path=ward_pcon_lookup_path,
        pcon_population_path=pcon_population_path,
        out_counts_path=counts_out_path,
        out_rates_path=rates_out_path,
    )

    validate_constituency_counts(counts)
    validate_constituency_rates(rates)

    log.info(f"Stage 2 complete. Wrote: {counts_out_path} (rows={len(counts):,})")
    log.info(f"Stage 2 complete. Wrote: {rates_out_path} (rows={len(rates):,})")
    log.info("Years in rates: %s to %s", int(rates["year"].min()), int(rates["year"].max()))

    # Optional preview
    log.info("Counts preview:\n%s", counts.head().to_string(index=False))
    log.info("Rates preview:\n%s", rates.head().to_string(index=False))


if __name__ == "__main__":
    main()
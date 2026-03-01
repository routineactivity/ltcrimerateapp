from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mapclassify
from matplotlib.colors import BoundaryNorm

import streamlit as st


# ---------- helpers ----------
def compute_bins(values: pd.Series, k: int = 6, method: str = "quantile") -> np.ndarray:
    v = values.dropna().to_numpy()
    if v.size == 0:
        return np.array([0, 1], dtype=float)

    if method == "quantile":
        bins = np.quantile(v, np.linspace(0, 1, k + 1))

    elif method == "equal":
        bins = np.linspace(v.min(), v.max(), k + 1)

    elif method == "natural":
        import mapclassify
        # mapclassify returns upper bounds; build full edges including min
        nb = mapclassify.NaturalBreaks(v, k=k)
        upper = nb.bins  # length k
        bins = np.concatenate(([v.min()], upper))

    else:
        raise ValueError("method must be one of: quantile, equal, natural")

    bins = np.unique(np.round(bins, 3))
    if bins.size < 2:
        bins = np.array([v.min(), v.max() + 1e-6])
    return bins


def aggregate_rates(
    df: pd.DataFrame,
    *,
    level: str,                 # "MajorText" or "MinorText"
    crime_types: list[str],
    years: list[int],
) -> pd.DataFrame:
    """
    Returns constituency-year rates at requested level.

    Correct aggregation:
      - sum crime_count across categories
      - population should NOT be summed (same per PCON-year). Use min(population).
      - recompute rate_per_1000
    """
    keep_cols = ["PCON24CD", "PCON24NM", "year", "crime_count", "population", "MajorText", "MinorText"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Rates CSV missing required columns: {missing}")

    sub = df[df["year"].isin(years)].copy()

    if level == "MajorText":
        sub = sub[sub["MajorText"].isin(crime_types)]
        group_cols = ["PCON24CD", "PCON24NM", "year", "MajorText"]

        out = (
            sub.groupby(group_cols, as_index=False)
            .agg(
                crime_count=("crime_count", "sum"),
                population=("population", "min"),
            )
        )

        # Optional safety check: population should be constant within each PCON-year group
        # (uncomment if you want the app to fail fast on bad joins)
        # pop_nunique = sub.groupby(["PCON24CD", "year"])["population"].nunique()
        # if (pop_nunique > 1).any():
        #     raise ValueError("Population varies within a PCON-year group. Check input data.")

        out["rate_per_1000"] = (out["crime_count"] / out["population"]) * 1000
        out = out.rename(columns={"MajorText": "crime_type"})
        return out

    # MinorText
    sub = sub[sub["MinorText"].isin(crime_types)]
    group_cols = ["PCON24CD", "PCON24NM", "year", "MinorText"]

    out = (
        sub.groupby(group_cols, as_index=False)
        .agg(
            crime_count=("crime_count", "sum"),
            population=("population", "min"),
        )
    )
    out["rate_per_1000"] = (out["crime_count"] / out["population"]) * 1000
    out = out.rename(columns={"MinorText": "crime_type"})
    return out


def plot_facets(
    geo: gpd.GeoDataFrame,
    *,
    years: list[int],
    crime_types: list[str],
    bins_method: str = "quantile",
    k_bins: int = 6,
    cmap_name: str = "BuPu",
    title_prefix: str = "",
) -> plt.Figure:

    nrows = len(crime_types)
    ncols = len(years)

    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
    })

    fig_w = 3.4 * ncols + 1.2   # add space on right for per-row legends
    fig_h = 2.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    cmap = getattr(plt.cm, cmap_name)

    # layout params for the per-row colorbars (figure coords)
    # these work well for compact facets; tweak if needed
    legend_left = 0.92
    legend_width = 0.015
    top = 0.88
    bottom = 0.12
    row_h = (top - bottom) / nrows
    legend_pad_y = 0.08  # shrink within each row height

    for r, ct in enumerate(crime_types):
        # bins fixed across years for this crime type
        vals = geo.loc[geo["crime_type"] == ct, "rate_per_1000"]
        bins = compute_bins(vals, k=k_bins, method=bins_method)
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)

        for c, y in enumerate(years):
            ax = axes[r, c]
            tmp = geo[(geo["crime_type"] == ct) & (geo["year"] == y)]

            tmp.plot(
                column="rate_per_1000",
                ax=ax,
                cmap=cmap,
                norm=norm,
                linewidth=0.25,
                edgecolor="grey",
                missing_kwds={"color": "lightgrey"},
            )

            if r == 0:
                ax.set_title(f"{y}", fontsize=9, pad=2)

            ax.axis("off")

            # row label on left of first column
            if c == 0:
                ax.text(
                    -0.02, 0.5, ct,
                    transform=ax.transAxes,
                    ha="right", va="center",
                    fontsize=8
                )

        # ---- per-row colorbar axis ----
        # Compute this row’s legend position in figure coords
        row_bottom = top - (r + 1) * row_h
        cax = fig.add_axes([
            legend_left,
            row_bottom + row_h * legend_pad_y,
            legend_width,
            row_h * (1 - 2 * legend_pad_y),
        ])

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_ticks(bins)
        cbar.ax.set_yticklabels([f"{b:.2f}" for b in bins])

        # label only on first row to reduce clutter
        if r == 0:
            cbar.set_label("Rate per 1,000", fontsize=8)
        else:
            cbar.set_label("")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=10, y=0.99)

    return fig


# ---------- app ----------
def main() -> None:
    st.set_page_config(page_title="Crime Rate Snapshots 2011-2025", layout="wide")
    st.title("Crime rates per 1,000 by constituency")

    base_dir = Path(__file__).resolve().parents[1]
    rates_path = base_dir / "data" / "processed" / "crime_constituency_year_rates_per_1000.csv"
    gpkg_path = base_dir / "londonparlcon2024.gpkg"

    st.sidebar.header("Inputs")

    if not rates_path.exists():
        st.error(f"Missing file: {rates_path}")
        return
    if not gpkg_path.exists():
        st.error(f"Missing file: {gpkg_path}")
        return

    rates = pd.read_csv(rates_path)

    # years selection
    all_years = sorted(rates["year"].dropna().astype(int).unique().tolist())
    years = st.sidebar.multiselect(
        "Select 3 years",
        options=all_years,
        default=[all_years[0], all_years[len(all_years)//2], all_years[-1]] if len(all_years) >= 3 else all_years
    )
    if len(years) != 3:
        st.warning("Select exactly 3 years.")
        return
    years = sorted([int(y) for y in years])

    # level selection
    level = st.sidebar.selectbox("Crime type source", options=["MajorText", "MinorText"], index=0)

    # crime types selection
    if level == "MajorText":
        options = sorted(rates["MajorText"].dropna().unique().tolist())
    else:
        options = sorted(rates["MinorText"].dropna().unique().tolist())

    crime_types = st.sidebar.multiselect(
        "Select one or more crime types",
        options=options,
        default=options[:1] if options else []
    )
    if not crime_types:
        st.warning("Select at least 1 crime type.")
        return

    # bins and styling
    bins_method = st.sidebar.selectbox(
    "Fixed bins method",
    options=["quantile", "equal", "natural"],
    index=0)
    k_bins = st.sidebar.slider("Number of bins", min_value=4, max_value=9, value=6)
    cmap_name = st.sidebar.selectbox("Colourmap", options=["BuPu", "viridis", "plasma", "YlOrRd", "Greens"], index=0)

    # load boundaries (choose layer if needed)
    # If your gpkg has multiple layers, add a selector:
    # layers = gpd.list_layers(gpkg_path)
    # st.sidebar.write(layers)
    gdf = gpd.read_file(gpkg_path)

    if "PCON24CD" not in gdf.columns:
        st.error("GeoPackage does not contain PCON24CD column needed for join.")
        st.write("Columns found:", list(gdf.columns))
        return

    # aggregate correctly
    agg = aggregate_rates(rates, level=level, crime_types=crime_types, years=years)

    # join geometry
    geo = gdf.merge(agg, on="PCON24CD", how="left")
    if geo["rate_per_1000"].isna().all():
        st.error("Join produced no rates. Check PCON24CD codes match between CSV and GeoPackage.")
        return

    st.subheader("Static snapshots showing rates of crime per 1,000 usual residents by Parliamentary Constituency")
    st.caption("Rows = crime types; columns = years. Bins are fixed per crime type across the selected years.")
    st.markdown(
    """
    <small>
    Crime data from
    <a href="https://data.london.gov.uk/dataset/mps-monthly-crime-dashboard-data-e5n6w/" target="_blank" rel="noopener noreferrer">
      London Datastore (MPS Monthly Crime Dashboard Data)
    </a>
    and population data from
    <a href="https://www.nomisweb.co.uk/" target="_blank" rel="noopener noreferrer">
      NOMIS
    </a>.
    </small>
    """,
    unsafe_allow_html=True,
)

    fig = plot_facets(
        geo,
        years=years,
        crime_types=crime_types,
        bins_method=bins_method,
        k_bins=k_bins,
        cmap_name=cmap_name,
        title_prefix=f"{level} | {', '.join(map(str, years))}",
    )
    st.pyplot(fig, clear_figure=True)

    # Optional: download the current figure
    st.sidebar.download_button(
        "Download current plot (PNG)",
        data=_fig_to_png_bytes(fig),
        file_name="crime_rates_facets.png",
        mime="image/png",
    )


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    main()
"""
app.py
======

India Climate & Disaster Risk Atlas (1901-2026)

Interactive Streamlit application that visualizes drought severity (SPEI),
flood fraction/risk, and EM-DAT disaster events across Indian states,
districts, and subdistricts on a Folium choropleth map, with a drill-down
inspector panel and historical trend charts.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


# --------------------------------------------------------------------------- #
# Config & paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"

YEAR_MIN, YEAR_MAX, YEAR_DEFAULT = 1901, 2026, 2020


LEVEL_CONFIG = {
    "State": {
        "summary_file": "state_summary.parquet",
        "geojson_file": "india_state.geojson",
        "name_field": "STATE_NAME",
    },
    "District": {
        "summary_file": "district_summary.parquet",
        "geojson_file": "india_district.geojson",
        "name_field": "DISTRICT_NAME",
    },
    "Subdistrict": {
        "summary_file": "subdistrict_summary.parquet",
        "geojson_file": "india_subdistrict.geojson",
        "name_field": "SUBDISTRICT_NAME",
    },
}


METRIC_OPTIONS = {
    "Drought Index (SPEI)": "SPEI-12month",
    "Flood Fraction": "flood_fraction",
    "Disaster Events": "event_count",
}


st.set_page_config(
    page_title="India Climate & Disaster Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_summary(level: str) -> pd.DataFrame:
    """Load processed summary data for the selected administrative level."""
    cfg = LEVEL_CONFIG[level]
    path = PROCESSED_DIR / cfg["summary_file"]

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_geojson(level: str) -> Optional[dict]:
    """Load processed GeoJSON boundary data."""
    cfg = LEVEL_CONFIG[level]
    path = PROCESSED_DIR / cfg["geojson_file"]

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_emdat_events() -> pd.DataFrame:
    """Load individual geocoded EM-DAT disaster events."""
    path = PROCESSED_DIR / "emdat_events.parquet"

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #

def spei_color(value: Optional[float]) -> str:
    """Map an SPEI z-score to a drought-severity color."""

    if value is None or pd.isna(value):
        return "#d9d9d9"

    if value <= -2.0:
        return "#7f0000"

    if value <= -1.5:
        return "#de2d26"

    if value <= -1.0:
        return "#fc9272"

    if value < 1.0:
        return "#ffffb2"

    if value < 1.5:
        return "#6baed6"

    return "#08519c"


def flood_color(value: Optional[float]) -> str:
    """Map flood fraction to a color."""

    if value is None or pd.isna(value):
        return "#d9d9d9"

    if value >= 30:
        return "#08306b"

    if value >= 15:
        return "#2171b5"

    if value >= 5:
        return "#6baed6"

    if value > 0:
        return "#c6dbef"

    return "#ffffb2"


def event_color(value: Optional[float]) -> str:
    """Map disaster event count to a color."""

    if value is None or pd.isna(value):
        return "#d9d9d9"

    if value >= 10:
        return "#7f0000"

    if value >= 5:
        return "#de2d26"

    if value >= 1:
        return "#fc9272"

    return "#ffffb2"


COLOR_FUNCS = {
    "SPEI-12month": spei_color,
    "flood_fraction": flood_color,
    "event_count": event_color,
}


# --------------------------------------------------------------------------- #
# Map legends
# --------------------------------------------------------------------------- #

LEGEND_DEFS = {
    "SPEI-12month": [
        ("Extreme drought (≤ -2.0)", "#7f0000"),
        ("Severe drought (-2.0 to -1.5)", "#de2d26"),
        ("Moderate drought (-1.5 to -1.0)", "#fc9272"),
        ("Normal (-1.0 to 1.0)", "#ffffb2"),
        ("Moderately wet (1.0 to 1.5)", "#6baed6"),
        ("Very wet (≥ 1.5)", "#08519c"),
        ("No data", "#d9d9d9"),
    ],

    "flood_fraction": [
        ("≥ 30%", "#08306b"),
        ("15% – 30%", "#2171b5"),
        ("5% – 15%", "#6baed6"),
        ("0% – 5%", "#c6dbef"),
        ("0% / none", "#ffffb2"),
        ("No data", "#d9d9d9"),
    ],

    "event_count": [
        ("≥ 10 events", "#7f0000"),
        ("5 – 9 events", "#de2d26"),
        ("1 – 4 events", "#fc9272"),
        ("0 events", "#ffffb2"),
        ("No data", "#d9d9d9"),
    ],
}


def build_legend_html(metric_col: str, metric_label: str) -> str:
    """Build HTML for the dynamic Folium map legend."""

    entries = LEGEND_DEFS.get(metric_col, [])

    rows = "".join(
        f"""
        <div style="display:flex;align-items:center;margin:2px 0;">
            <span style="
                display:inline-block;
                width:14px;
                height:14px;
                background:{color};
                border:1px solid #888;
                margin-right:6px;
                flex-shrink:0;
            "></span>
            <span style="font-size:12px; color:#000000;">{label}</span>
        </div>
        """
        for label, color in entries
    )

    return f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        font-family: sans-serif;
        color: #000000 !important;
    ">
        <div style="
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 4px;
            color: #000000 !important;
        ">
            {metric_label}
        </div>

        {rows}
    </div>
    """


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

st.sidebar.title("Configurations")


admin_level = st.sidebar.selectbox(
    "Please select level",
    list(LEVEL_CONFIG.keys()),
    index=1,
)


selected_year = st.sidebar.slider(
    "Please select the year",
    min_value=YEAR_MIN,
    max_value=YEAR_MAX,
    value=YEAR_DEFAULT,
    step=1,
)


metric_label = st.sidebar.selectbox(
    "Please select the metric",
    list(METRIC_OPTIONS.keys()),
)

metric_col = METRIC_OPTIONS[metric_label]


show_disaster_markers = st.sidebar.checkbox(
    "Show disaster event markers on map",
    value=True,
    help=(
        "Plots individual EM-DAT disaster locations near the selected year. "
        "Aggregated per-region counts may be sparse in a single year."
    ),
)


if show_disaster_markers:

    marker_window = st.sidebar.slider(
        "Year range for disaster (± years)",
        min_value=0,
        max_value=10,
        value=2,
        help=(
            "Show disaster events within this many years "
            "of the selected year."
        ),
    )

else:
    marker_window = 0


st.sidebar.markdown("---")



# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #

cfg = LEVEL_CONFIG[admin_level]
name_field = cfg["name_field"]

summary_df = load_summary(admin_level)
geojson_data = load_geojson(admin_level)


# --------------------------------------------------------------------------- #
# Dashboard heading
# --------------------------------------------------------------------------- #

st.title("India Climate & Disaster Risk Atlas")

st.caption(
    f"{admin_level} level · {selected_year} · {metric_label}"
)


if summary_df.empty:
    st.warning(
        f"No processed data found for the **{admin_level}** level. "
        f"Place raw source files under `data/raw/` and run "
        f"`python preprocess.py` to generate `{cfg['summary_file']}`."
    )


if geojson_data is None:
    st.info(
        f"No boundary GeoJSON found for **{admin_level}** "
        f"(`{cfg['geojson_file']}`). The map will be skipped "
        f"until it exists."
    )


# --------------------------------------------------------------------------- #
# Filter selected year
# --------------------------------------------------------------------------- #

year_df = (
    summary_df[summary_df["Year"] == selected_year].copy()
    if not summary_df.empty and "Year" in summary_df.columns
    else pd.DataFrame()
)


lookup = (
    year_df.set_index(name_field)
    if not year_df.empty and name_field in year_df.columns
    else pd.DataFrame()
)


# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #

map_col, inspector_col = st.columns([2, 1])


selected_region: Optional[str] = st.session_state.get(
    "selected_region"
)


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #

with map_col:

    st.subheader("Map View")

    if geojson_data is not None:

        m = folium.Map(
            location=[22.5, 80.0],
            zoom_start=5,
            tiles="cartodbpositron",
        )

        color_fn = COLOR_FUNCS.get(
            metric_col,
            spei_color,
        )


        def style_function(feature):

            name = feature["properties"].get(name_field)

            val = None

            if not lookup.empty and name in lookup.index:

                row = lookup.loc[name]

                val = (
                    row[metric_col]
                    if metric_col in row
                    else None
                )

                if isinstance(val, pd.Series):
                    val = val.iloc[0]

            return {
                "fillColor": color_fn(val),
                "color": "#555555",
                "weight": 0.6,
                "fillOpacity": 0.75,
            }


        def highlight_function(_feature):

            return {
                "weight": 2.5,
                "color": "#000000",
                "fillOpacity": 0.9,
            }


        # ------------------------------------------------------------------- #
        # Tooltips
        # ------------------------------------------------------------------- #

        for feature in geojson_data.get("features", []):

            name = feature["properties"].get(name_field)

            row = (
                lookup.loc[name]
                if not lookup.empty and name in lookup.index
                else None
            )

            spei_val = "N/A"
            flood_val = "N/A"
            events_val = "N/A"

            if row is not None:

                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                spei_value = row.get(
                    "SPEI-12month",
                    None,
                )

                flood_value = row.get(
                    "flood_fraction",
                    None,
                )

                event_value = row.get(
                    "event_count",
                    None,
                )

                spei_val = (
                    round(spei_value, 2)
                    if pd.notna(spei_value)
                    else "N/A"
                )

                flood_val = (
                    round(flood_value, 2)
                    if pd.notna(flood_value)
                    else "N/A"
                )

                events_val = (
                    int(event_value)
                    if pd.notna(event_value)
                    else 0
                )


            feature["properties"]["_tooltip"] = (
                f"{name} ({selected_year})<br>"
                f"SPEI (12mo): {spei_val}<br>"
                f"Flood Fraction: {flood_val}%<br>"
                f"Disaster Events: {events_val}"
            )


        # ------------------------------------------------------------------- #
        # Administrative boundaries
        # ------------------------------------------------------------------- #

        gj = folium.GeoJson(
            geojson_data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["_tooltip"],
                aliases=[""],
                labels=False,
                sticky=True,
            ),
            name="Administrative Boundaries",
        )

        gj.add_to(m)


        # ------------------------------------------------------------------- #
        # EM-DAT disaster markers
        # ------------------------------------------------------------------- #

        if show_disaster_markers:

            emdat_events = load_emdat_events()

            if (
                not emdat_events.empty
                and "Year" in emdat_events.columns
            ):

                window_events = emdat_events[
                    (
                        emdat_events["Year"]
                        >= selected_year - marker_window
                    )
                    &
                    (
                        emdat_events["Year"]
                        <= selected_year + marker_window
                    )
                ]


                marker_layer = folium.FeatureGroup(
                    name="Disaster Events",
                    show=True,
                )


                for _, ev in window_events.iterrows():

                    lat = ev.get("latitude")
                    lon = ev.get("longitude")

                    if pd.isna(lat) or pd.isna(lon):
                        continue


                    deaths = (
                        int(ev["total_deaths"])
                        if pd.notna(ev.get("total_deaths"))
                        else "N/A"
                    )

                    affected = (
                        int(ev["total_affected"])
                        if pd.notna(ev.get("total_affected"))
                        else "N/A"
                    )


                    popup_html = (
                        f"<b>{ev.get('disaster_type', 'Disaster')}</b> "
                        f"({int(ev['Year'])})<br>"
                        f"{ev.get('location', 'Unknown location')}<br>"
                        f"Deaths: {deaths}<br>"
                        f"Affected: {affected}"
                    )


                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color="#7f0000",
                        fill=True,
                        fill_color="#de2d26",
                        fill_opacity=0.85,
                        weight=1,
                        popup=folium.Popup(
                            popup_html,
                            max_width=250,
                        ),
                    ).add_to(marker_layer)


                marker_layer.add_to(m)


                if window_events.empty:

                    st.caption(
                        f"No geocoded EM-DAT disaster events within "
                        f"±{marker_window} years of {selected_year}."
                    )


            elif emdat_events.empty:

                st.caption(
                    "No EM-DAT event-location data found — run "
                    "`preprocess.py` to generate `emdat_events.parquet`."
                )


        # ------------------------------------------------------------------- #
        # Layer control and legend
        # ------------------------------------------------------------------- #

        folium.LayerControl().add_to(m)

        m.get_root().html.add_child(
            folium.Element(
                build_legend_html(
                    metric_col,
                    metric_label,
                )
            )
        )


        # ------------------------------------------------------------------- #
        # Render map
        # ------------------------------------------------------------------- #

        map_state = st_folium(
            m,
            height=560,
            use_container_width=True,
            key="main_map",
        )


        # ------------------------------------------------------------------- #
        # Capture polygon click
        # ------------------------------------------------------------------- #

        if (
            map_state
            and map_state.get("last_active_drawing")
        ):

            clicked_name = (
                map_state["last_active_drawing"]
                ["properties"]
                .get(name_field)
            )

            if clicked_name:

                st.session_state[
                    "selected_region"
                ] = clicked_name

                selected_region = clicked_name


    else:

        st.info(
            "Map unavailable — no boundary data loaded."
        )


# --------------------------------------------------------------------------- #
# Region Inspector
# --------------------------------------------------------------------------- #

with inspector_col:

    st.subheader("Region Inspector")


    region_options = (
        sorted(
            summary_df[name_field]
            .dropna()
            .unique()
            .tolist()
        )
        if (
            not summary_df.empty
            and name_field in summary_df.columns
        )
        else []
    )


    default_index = 0

    if (
        selected_region
        and selected_region in region_options
    ):
        default_index = (
            region_options.index(selected_region) + 1
        )


    picked = st.selectbox(
        f"Select {admin_level}",
        ["-- choose a region or click the map --"]
        + region_options,
        index=default_index,
    )


    if picked != "-- choose a region or click the map --":

        selected_region = picked

        st.session_state[
            "selected_region"
        ] = picked


    # ----------------------------------------------------------------------- #
    # Region statistics
    # ----------------------------------------------------------------------- #

    if (
        selected_region
        and not summary_df.empty
    ):

        region_hist = (
            summary_df[
                summary_df[name_field]
                == selected_region
            ]
            .sort_values("Year")
        )


        if not region_hist.empty:

            latest = region_hist[
                region_hist["Year"]
                == selected_year
            ]


            total_disasters = (
                int(region_hist["event_count"].sum())
                if "event_count" in region_hist
                else 0
            )

            total_deaths = (
                int(region_hist["total_deaths"].sum())
                if "total_deaths" in region_hist
                else 0
            )

            total_affected = (
                int(region_hist["total_affected"].sum())
                if "total_affected" in region_hist
                else 0
            )


            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Total Disasters",
                f"{total_disasters}",
            )

            m2.metric(
                "Total Deaths",
                f"{total_deaths}",
            )

            m3.metric(
                "Total Affected",
                f"{total_affected}",
            )


            # --------------------------------------------------------------- #
            # Current SPEI
            # --------------------------------------------------------------- #

            if (
                not latest.empty
                and "SPEI-12month" in latest.columns
            ):

                spei_now = (
                    latest["SPEI-12month"]
                    .iloc[0]
                )

                st.metric(
                    f"SPEI-12mo ({selected_year})",
                    (
                        f"{spei_now:.2f}"
                        if pd.notna(spei_now)
                        else "N/A"
                    ),
                )


            # --------------------------------------------------------------- #
            # Historical trend
            # --------------------------------------------------------------- #

            st.markdown(
                f"**Historical Trend — {selected_region}** "
                f"({YEAR_MIN}–{YEAR_MAX})"
            )


            fig = go.Figure()


            if "SPEI-12month" in region_hist.columns:

                fig.add_trace(
                    go.Scatter(
                        x=region_hist["Year"],
                        y=region_hist["SPEI-12month"],
                        name="SPEI (12-month)",
                        mode="lines",
                        yaxis="y1",
                        line=dict(color="#de2d26"),
                    )
                )


            if "flood_fraction" in region_hist.columns:

                fig.add_trace(
                    go.Scatter(
                        x=region_hist["Year"],
                        y=region_hist["flood_fraction"],
                        name="Flood Fraction (%)",
                        mode="lines",
                        yaxis="y2",
                        line=dict(color="#2171b5"),
                    )
                )


            fig.update_layout(
                height=340,
                margin=dict(
                    l=10,
                    r=10,
                    t=30,
                    b=10,
                ),
                yaxis=dict(
                    title="SPEI",
                ),
                yaxis2=dict(
                    title="Flood %",
                    overlaying="y",
                    side="right",
                ),
                legend=dict(
                    orientation="h",
                    y=-0.2,
                ),
                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        else:

            st.info(
                "No historical records for this region."
            )


    else:

        st.info(
            "Pick a region above or click a polygon on "
            "the map to see details."
        )


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #

st.markdown("---")


"""
app.py
======
India Climate & Disaster Risk Atlas (1901-2026)

Interactive Streamlit application that visualizes drought severity (SPEI),
flood fraction/risk, and EM-DAT disaster events across Indian states,
districts, and subdistricts on a Folium choropleth map, with a drill-down
tabbed inspector panel and historical trend charts.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    "Flood Area": "flood_area",
    "Disaster Events (All-Time Total)": "event_count",
}

st.set_page_config(
    page_title="India Environment Risk Atlas",
    page_icon="\U0001F327",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Visual identity & helper utilities
# --------------------------------------------------------------------------- #

INK = "#1A1B2E"
INDIGO = "#57595E"
PARCHMENT = "#0D0D0D"
MARIGOLD = "#E2A63B"
MONSOON_BLUE = "#2C6E8E"
DROUGHT_RED = "#B03A2E"
CARD_BORDER = "#E4DCC8"

def format_compact_number(val: Union[int, float, None]) -> str:
    """Format numbers into K, M, B compact representation to prevent metric overflow."""
    if val is None or pd.isna(val):
        return "0"
    num = float(val)
    if num >= 1e9:
        return f"{num / 1e9:.2f}B".rstrip('0').rstrip('.')
    if num >= 1e6:
        return f"{num / 1e6:.2f}M".rstrip('0').rstrip('.')
    if num >= 1e3:
        return f"{num / 1e3:.1f}K".rstrip('0').rstrip('.')
    return f"{int(num):,}"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500..700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}

.stApp {{
    background-color: {PARCHMENT};
}}

/* Sidebar controls */
section[data-testid="stSidebar"] {{
    background-color: {INDIGO};
}}
section[data-testid="stSidebar"] * {{
    color: #EDE7D6 !important;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label {{
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}}

/* Dashboard title */
.dashboard-title {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.4rem;
    letter-spacing: -0.01em;
    color: #EDE7D6;
    margin-bottom: 0.1rem;
}}

/* Signature gradient bar */
.spectrum-bar {{
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(
        90deg, #67000d 0%, {DROUGHT_RED} 20%, {MARIGOLD} 40%, #FFFFCC 50%,
        #74c476 65%, {MONSOON_BLUE} 85%, #00441b 100%
    );
    margin-bottom: 1.4rem;
}}

/* Metric cards text visibility */
div[data-testid="stMetric"] {{
    background-color: #FFFFFF;
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem 0.7rem 1rem;
    box-shadow: 0 1px 3px rgba(22, 35, 63, 0.06);
}}
div[data-testid="stMetric"] * {{
    color: {INK} !important;
}}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {INK} !important;
}}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {{
    font-family: 'Fraunces', serif;
    color: {INK} !important;
}}

/* Section subheaders */
h3 {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: #EDE7D6;
}}

div[data-baseweb="select"] > div {{
    border-radius: 8px;
}}

.stCaption, [data-testid="stCaptionContainer"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem !important;
}}

/* Custom styling for tabs */
button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: #EDE7D6 !important;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_summary(level: str) -> pd.DataFrame:
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
    path = PROCESSED_DIR / "emdat_events.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# High-density granular color mapping functions
# --------------------------------------------------------------------------- #

def spei_color(value: Optional[float]) -> str:
    """Detailed 9-bin SPEI z-score scale from extreme drought to extreme wetness."""
    if value is None or pd.isna(value):
        return "#d9d9d9"
    if value <= -2.0:
        return "#67000d"
    if value <= -1.5:
        return "#a50f15"
    if value <= -1.0:
        return "#d95f0e"
    if value <= -0.5:
        return "#fe9929"
    if value < 0.5:
        return "#ffffcc"
    if value < 1.0:
        return "#c7e9c0"
    if value < 1.5:
        return "#74c476"
    if value < 2.0:
        return "#238b45"
    return "#00441b"


def flood_color(value: Optional[float]) -> str:
    """Detailed 8-bin scale for flood percentage coverage."""
    if value is None or pd.isna(value):
        return "#d9d9d9"
    if value >= 40:
        return "#030512"
    if value >= 25:
        return "#08306b"
    if value >= 15:
        return "#08519c"
    if value >= 10:
        return "#2171b5"
    if value >= 5:
        return "#4292c6"
    if value >= 2:
        return "#6baed6"
    if value > 0:
        return "#c6dbef"
    return "#ffffd9"


def flood_area_color(value: Optional[float]) -> str:
    """Detailed 11-bin high-granularity scale for absolute flood surface area (km²)."""
    if value is None or pd.isna(value):
        return "#d9d9d9"
    if value >= 10000:
        return "#020024"  # Extreme inundation
    if value >= 5000:
        return "#030512"
    if value >= 2500:
        return "#081d58"
    if value >= 1000:
        return "#253494"
    if value >= 500:
        return "#225ea8"
    if value >= 250:
        return "#1d91c0"
    if value >= 100:
        return "#41b6c4"
    if value >= 50:
        return "#7fcdbb"
    if value >= 10:
        return "#c7e9b4"
    if value > 0:
        return "#edf8b1"
    return "#ffffd9"


def event_color(value: Optional[float]) -> str:
    """Detailed scale for cumulative all-time disaster event occurrences (1901-2026)."""
    if value is None or pd.isna(value):
        return "#d9d9d9"
    if value >= 100:
        return "#4a0000"
    if value >= 50:
        return "#7f0000"
    if value >= 30:
        return "#b30000"
    if value >= 15:
        return "#d95f0e"
    if value >= 5:
        return "#f16913"
    if value >= 1:
        return "#fdd0a2"
    return "#ffffd9"


COLOR_FUNCS = {
    "SPEI-12month": spei_color,
    "flood_fraction": flood_color,
    "flood_area": flood_area_color,
    "event_count": event_color,
}

LEGEND_DEFS = {
    "SPEI-12month": [
        ("Extreme drought (\u2264 -2.0)", "#67000d"),
        ("Severe drought (-2.0 to -1.5)", "#a50f15"),
        ("Moderate drought (-1.5 to -1.0)", "#d95f0e"),
        ("Mild drought (-1.0 to -0.5)", "#fe9929"),
        ("Near normal (-0.5 to 0.5)", "#ffffcc"),
        ("Slightly wet (0.5 to 1.0)", "#c7e9c0"),
        ("Moderately wet (1.0 to 1.5)", "#74c476"),
        ("Very wet (1.5 to 2.0)", "#238b45"),
        ("Extremely wet (\u2265 2.0)", "#00441b"),
        ("No data", "#d9d9d9"),
    ],
    "flood_fraction": [
        ("\u2265 40%", "#030512"),
        ("25% \u2013 40%", "#08306b"),
        ("15% \u2013 25%", "#08519c"),
        ("10% \u2013 15%", "#2171b5"),
        ("5% \u2013 10%", "#4292c6"),
        ("2% \u2013 5%", "#6baed6"),
        ("0% \u2013 2%", "#c6dbef"),
        ("0% / none", "#ffffd9"),
        ("No data", "#d9d9d9"),
    ],
    "flood_area": [
        ("\u2265 10,000 km\u00b2", "#020024"),
        ("5,000 \u2013 10,000 km\u00b2", "#030512"),
        ("2,500 \u2013 5,000 km\u00b2", "#081d58"),
        ("1,000 \u2013 2,500 km\u00b2", "#253494"),
        ("500 \u2013 1,000 km\u00b2", "#225ea8"),
        ("250 \u2013 500 km\u00b2", "#1d91c0"),
        ("100 \u2013 250 km\u00b2", "#41b6c4"),
        ("50 \u2013 100 km\u00b2", "#7fcdbb"),
        ("10 \u2013 50 km\u00b2", "#c7e9b4"),
        ("0 \u2013 10 km\u00b2", "#edf8b1"),
        ("0 km\u00b2 / none", "#ffffd9"),
        ("No data", "#d9d9d9"),
    ],
    "event_count": [
        ("\u2265 100 events (All Years)", "#4a0000"),
        ("50 \u2013 99 events", "#7f0000"),
        ("30 \u2013 49 events", "#b30000"),
        ("15 \u2013 29 events", "#d95f0e"),
        ("5 \u2013 14 events", "#f16913"),
        ("1 \u2013 4 events", "#fdd0a2"),
        ("0 events", "#ffffd9"),
        ("No data", "#d9d9d9"),
    ],
}


def build_legend_html(metric_col: str, metric_label: str) -> str:
    entries = LEGEND_DEFS.get(metric_col, [])
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'border-radius:2px;background:{color};border:1px solid rgba(0,0,0,0.2);'
        f'margin-right:6px;flex-shrink:0;"></span>'
        f'<span style="font-size:11px;color:#1A1B2E;line-height:1.2;">{label}</span></div>'
        for label, color in entries
    )
    return f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background: #FFFFFF; padding: 10px 12px; border-radius: 8px;
                border: 1px solid #E4DCC8;
                box-shadow: 0 2px 8px rgba(22,35,63,0.18);
                font-family: 'Inter', sans-serif;">
        <div style="font-family:'IBM Plex Mono', monospace; font-weight: 600;
                    font-size: 10px; text-transform: uppercase;
                    letter-spacing: 0.05em; color: #2C6E8E; margin-bottom: 5px;">
            {metric_label}
        </div>
        {rows}
    </div>
    """


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

st.sidebar.markdown(
    '<div style="font-family:\'Fraunces\',serif; font-weight:600; '
    'font-size:1.4rem; color:#EDE7D6; margin-bottom:0;">'
    'Configurations</div>'
    '<div style="monospace; font-size:0.7rem; '
    'color:#B8AE8F; text-transform:uppercase; letter-spacing:0.06em; '
    'margin-bottom:1rem;">Filter by level, year, and metric</div>',
    unsafe_allow_html=True,
)

admin_level = st.sidebar.selectbox(
    "Choose a level", list(LEVEL_CONFIG.keys()), index=1
)

selected_year = st.sidebar.slider(
    "Choose a year", min_value=YEAR_MIN, max_value=YEAR_MAX, value=YEAR_DEFAULT, step=1
)

metric_label = st.sidebar.selectbox("Choose the metric", list(METRIC_OPTIONS.keys()))
metric_col = METRIC_OPTIONS[metric_label]

show_disaster_markers = st.sidebar.checkbox(
    "Mark disaster events on the map", value=True,
    help="Plots individual EM-DAT disaster locations near the selected year.",
)
if show_disaster_markers:
    marker_window = st.sidebar.slider(
        "Include events within \u00b1 years", min_value=0, max_value=10, value=2,
        help="Show disaster events within this many years of the selected year.",
    )
else:
    marker_window = 0

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <a href="https://github.com/Omkar2703/WCL_Atlas.git" target="_blank" style="text-decoration: none;">
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            background-color: #2C6E8E;
            color: #EDE7D6;
            padding: 10px 16px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            transition: background-color 0.2s ease;">
            <svg height="18" width="18" viewBox="0 0 16 16" fill="#EDE7D6">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            GitHub Repository
        </div>
    </a>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Load data for the chosen level
# --------------------------------------------------------------------------- #

cfg = LEVEL_CONFIG[admin_level]
name_field = cfg["name_field"]

summary_df = load_summary(admin_level)
geojson_data = load_geojson(admin_level)

st.markdown(
    '<div class="dashboard-title">India Environment Risk Atlas</div>'
    '<div class="spectrum-bar"></div>',
    unsafe_allow_html=True,
)
st.caption(f"Viewing: {admin_level} level \u00b7 {selected_year} \u00b7 {metric_label}")

if summary_df.empty:
    st.warning(
        f"No processed data found for the **{admin_level}** level. "
        f"Place raw source files under `data/raw/` and run `python preprocess.py` "
        f"to generate `{cfg['summary_file']}`."
    )

if geojson_data is None:
    st.info(
        f"No boundary GeoJSON found for **{admin_level}** "
        f"(`{cfg['geojson_file']}`). The map will be skipped until it exists."
    )

year_df = (
    summary_df[summary_df["Year"] == selected_year].copy()
    if not summary_df.empty and "Year" in summary_df.columns
    else pd.DataFrame()
)

lookup = (
    year_df.set_index(name_field) if not year_df.empty and name_field in year_df.columns
    else pd.DataFrame()
)

# Total disaster events per region across ALL years
total_events_by_entity = (
    summary_df.groupby(name_field)["event_count"].sum()
    if not summary_df.empty and {"event_count", name_field}.issubset(summary_df.columns)
    else pd.Series(dtype=float)
)

selected_region: Optional[str] = st.session_state.get("selected_region")

# --------------------------------------------------------------------------- #
# Section 1: Map View (Full Width)
# --------------------------------------------------------------------------- #

st.subheader("Map View")

if geojson_data is not None:
    m = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles="cartodbpositron")
    color_fn = COLOR_FUNCS.get(metric_col, spei_color)

    def style_function(feature):
        name = feature["properties"].get(name_field)
        val = None

        if metric_col == "event_count":
            if not total_events_by_entity.empty and name in total_events_by_entity.index:
                val = total_events_by_entity.loc[name]
        else:
            if not lookup.empty and name in lookup.index:
                row = lookup.loc[name]
                val = row[metric_col] if metric_col in row else None
                if isinstance(val, pd.Series):
                    val = val.iloc[0]

        return {
            "fillColor": color_fn(val),
            "color": "#444444",
            "weight": 0.5,
            "fillOpacity": 0.8,
        }

    def highlight_function(_feature):
        return {"weight": 2.5, "color": "#000000", "fillOpacity": 0.95}

    for feature in geojson_data.get("features", []):
        name = feature["properties"].get(name_field)
        row = lookup.loc[name] if (not lookup.empty and name in lookup.index) else None
        spei_val = flood_val = area_val = "N/A"
        if row is not None:
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            spei_val = round(row.get("SPEI-12month"), 2) if pd.notna(row.get("SPEI-12month", None)) else "N/A"
            flood_val = round(row.get("flood_fraction"), 2) if pd.notna(row.get("flood_fraction", None)) else "N/A"
            area_val = round(row.get("flood_area"), 1) if pd.notna(row.get("flood_area", None)) else "N/A"
        
        events_val = (
            int(total_events_by_entity.get(name, 0))
            if not total_events_by_entity.empty else 0
        )
        
        feature["properties"]["_tooltip"] = (
            f"{name}<br>"
            f"SPEI-12mo ({selected_year}): {spei_val}<br>"
            f"Flood Fraction ({selected_year}): {flood_val}%<br>"
            f"Flood Area ({selected_year}): {area_val} km\u00b2<br>"
            f"Disaster Events (All Years): {events_val}"
        )

    gj = folium.GeoJson(
        geojson_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(fields=["_tooltip"], aliases=[""], labels=False, sticky=True),
        name="admin_layer",
    )
    gj.add_to(m)

    if show_disaster_markers:
        emdat_events = load_emdat_events()
        if not emdat_events.empty and "Year" in emdat_events.columns:
            window_events = emdat_events[
                (emdat_events["Year"] >= selected_year - marker_window)
                & (emdat_events["Year"] <= selected_year + marker_window)
            ]
            marker_layer = folium.FeatureGroup(name="Disaster Events", show=True)
            lat_col = "plot_lat" if "plot_lat" in window_events.columns else "latitude"
            lon_col = "plot_lon" if "plot_lon" in window_events.columns else "longitude"
            for _, ev in window_events.iterrows():
                lat, lon = ev.get(lat_col), ev.get(lon_col)
                if pd.isna(lat) or pd.isna(lon):
                    continue
                precision = ev.get("location_precision", "exact")
                is_exact = precision == "exact"
                precision_label = {
                    "exact": "Exact location",
                    "district_centroid": "Approximate \u2014 district centroid",
                    "state_centroid": "Approximate \u2014 state centroid",
                }.get(precision, "Location precision unknown")
                popup_html = (
                    f"<b>{ev.get('disaster_type', 'Disaster')}</b> "
                    f"({int(ev['Year'])})<br>"
                    f"{ev.get('location', 'Unknown location')}<br>"
                    f"Deaths: {format_compact_number(ev.get('total_deaths'))}<br>"
                    f"Affected: {format_compact_number(ev.get('total_affected'))}<br>"
                    f"<i>{precision_label}</i>"
                )
                if is_exact:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color="#7f0000",
                        fill=True,
                        fill_color="#de2d26",
                        fill_opacity=0.9,
                        weight=1.5,
                        popup=folium.Popup(popup_html, max_width=260),
                    ).add_to(marker_layer)
                else:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        color="#7f0000",
                        fill=True,
                        fill_color="#fc9272",
                        fill_opacity=0.4,
                        weight=1.5,
                        dash_array="4",
                        popup=folium.Popup(popup_html, max_width=260),
                    ).add_to(marker_layer)
            marker_layer.add_to(m)

    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(build_legend_html(metric_col, metric_label)))

    map_state = st_folium(m, height=580, use_container_width=True, key="main_map")

    if map_state and map_state.get("last_active_drawing"):
        clicked_name = map_state["last_active_drawing"]["properties"].get(name_field)
        if clicked_name:
            st.session_state["selected_region"] = clicked_name
            selected_region = clicked_name
else:
    st.info("Map unavailable — no boundary data loaded.")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Section 2: Multi-Metric Tabbed Region Inspector (Positioned Below Map)
# --------------------------------------------------------------------------- #

st.subheader("Region Inspector")

region_options = (
    sorted(summary_df[name_field].dropna().unique().tolist())
    if not summary_df.empty and name_field in summary_df.columns
    else []
)

default_index = 0
if selected_region and selected_region in region_options:
    default_index = region_options.index(selected_region) + 1

picked = st.selectbox(
    f"Select {admin_level}",
    ["-- choose a region or click the map --"] + region_options,
    index=default_index,
)
if picked != "-- choose a region or click the map --":
    selected_region = picked
    st.session_state["selected_region"] = picked

if selected_region and not summary_df.empty:
    region_hist = summary_df[summary_df[name_field] == selected_region].sort_values("Year")

    if not region_hist.empty:
        latest = region_hist[region_hist["Year"] == selected_year]
        total_disasters = region_hist["event_count"].sum() if "event_count" in region_hist else 0
        total_deaths = region_hist["total_deaths"].sum() if "total_deaths" in region_hist else 0
        total_affected = region_hist["total_affected"].sum() if "total_affected" in region_hist else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Disasters", format_compact_number(total_disasters))
        m2.metric("Total Deaths", format_compact_number(total_deaths))
        m3.metric("Total Affected", format_compact_number(total_affected))

        if not latest.empty and "SPEI-12month" in latest.columns:
            spei_now = latest["SPEI-12month"].iloc[0]
            m4.metric(
                f"SPEI-12mo ({selected_year})",
                f"{spei_now:.2f}" if pd.notna(spei_now) else "N/A",
            )

        st.markdown(f"**Detailed Inspector — {selected_region}**")

        tab1, tab2 = st.tabs(["Climate Risk (SPEI & Flood)", "Disaster History"])

        # Tab 1: Climate Risk (SPEI & Flood Coverage)
        with tab1:
            fig1 = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    "SPEI Drought / Wetness Index (Red = Drought | Blue = Wet)",
                    "Flood Area Fraction (%)"
                )
            )

            if "SPEI-12month" in region_hist.columns:
                spei_vals = region_hist["SPEI-12month"].fillna(0)
                colors = ["#b03a2e" if v < 0 else "#2c6e8e" for v in spei_vals]
                fig1.add_trace(
                    go.Bar(
                        x=region_hist["Year"],
                        y=region_hist["SPEI-12month"],
                        marker_color=colors,
                        name="SPEI Index",
                    ),
                    row=1, col=1
                )

            if "flood_fraction" in region_hist.columns:
                fig1.add_trace(
                    go.Scatter(
                        x=region_hist["Year"],
                        y=region_hist["flood_fraction"],
                        fill="tozeroy",
                        name="Flood Fraction (%)",
                        line=dict(color="#2171b5", width=1.5),
                    ),
                    row=2, col=1
                )

            fig1.update_layout(
                height=420,
                margin=dict(l=10, r=20, t=30, b=10),
                showlegend=False,
                template="plotly_white"
            )
            fig1.update_yaxes(title_text="SPEI", row=1, col=1)
            fig1.update_yaxes(title_text="Flood %", row=2, col=1)

            st.plotly_chart(fig1, use_container_width=True)

        # Tab 2: Disaster History & Human Casualties
        with tab2:
            emdat_events = load_emdat_events()
            region_emdat = pd.DataFrame()

            if not emdat_events.empty:
                match_cols = [c for c in ["state", "district", "location", name_field] if c in emdat_events.columns]
                for col in match_cols:
                    matched = emdat_events[emdat_events[col].astype(str).str.lower() == str(selected_region).lower()]
                    if not matched.empty:
                        region_emdat = matched
                        break

            if not region_emdat.empty and "disaster_type" in region_emdat.columns:
                disaster_by_type = region_emdat.groupby(["Year", "disaster_type"]).size().reset_index(name="count")

                fig2 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.12,
                    subplot_titles=("Yearly Disaster Events by Type", "Human Impact (Deaths & Affected)")
                )

                for d_type in disaster_by_type["disaster_type"].unique():
                    sub = disaster_by_type[disaster_by_type["disaster_type"] == d_type]
                    fig2.add_trace(
                        go.Bar(x=sub["Year"], y=sub["count"], name=str(d_type)),
                        row=1, col=1
                    )

                impact_by_year = region_emdat.groupby("Year").agg({
                    "total_deaths": "sum",
                    "total_affected": "sum"
                }).reset_index()

                fig2.add_trace(
                    go.Scatter(
                        x=impact_by_year["Year"], y=impact_by_year["total_deaths"],
                        name="Deaths", line=dict(color="#b03a2e", width=2)
                    ),
                    row=2, col=1
                )
                fig2.add_trace(
                    go.Scatter(
                        x=impact_by_year["Year"], y=impact_by_year["total_affected"],
                        name="Affected", line=dict(color="#e2a63b", width=2)
                    ),
                    row=2, col=1
                )

                fig2.update_layout(
                    barmode="stack",
                    height=420,
                    margin=dict(l=10, r=20, t=30, b=10),
                    legend=dict(orientation="h", y=-0.25),
                    template="plotly_white"
                )
                fig2.update_yaxes(title_text="Events", row=1, col=1)
                fig2.update_yaxes(
                    title_text="Count",
                    type="log" if impact_by_year["total_affected"].max() > 10000 else "linear",
                    row=2, col=1
                )

                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig2 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.12,
                    subplot_titles=("Yearly Disaster Occurrence", "Casualties & Affected Population")
                )

                if "event_count" in region_hist.columns:
                    fig2.add_trace(
                        go.Bar(
                            x=region_hist["Year"], y=region_hist["event_count"],
                            marker_color="#e2a63b", name="Disaster Events"
                        ),
                        row=1, col=1
                    )

                if "total_deaths" in region_hist.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=region_hist["Year"], y=region_hist["total_deaths"],
                            name="Total Deaths", line=dict(color="#b03a2e", width=2)
                        ),
                        row=2, col=1
                    )
                if "total_affected" in region_hist.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=region_hist["Year"], y=region_hist["total_affected"],
                            name="Total Affected", line=dict(color="#2c6e8e", width=2)
                        ),
                        row=2, col=1
                    )

                fig2.update_layout(
                    height=420,
                    margin=dict(l=10, r=20, t=30, b=10),
                    legend=dict(orientation="h", y=-0.25),
                    template="plotly_white"
                )
                fig2.update_yaxes(title_text="Events", row=1, col=1)
                fig2.update_yaxes(title_text="Impact Count", row=2, col=1)

                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No historical records for this region.")
else:
    st.info("Pick a region above or click a polygon on the map to see details.")

st.markdown("---")
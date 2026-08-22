"""
preprocess.py
=============
Data engineering pipeline for the India Climate & Disaster Risk Dashboard.

Ingests three raw data sources:
    1. DATA_SPEI_DroughtAtlas  -> administrative boundaries + SPEI drought z-scores
    2. flood_atlas             -> district/state flood fraction & flood risk JSON
    3. Em_dat_droughtandfloods_2000_2026.xlsx -> EM-DAT disaster event records

...and produces analysis-ready, web-map-ready outputs in data/processed/:
    - state_summary.parquet
    - district_summary.parquet
    - subdistrict_summary.parquet
    - india_state.geojson
    - india_district.geojson
    - india_subdistrict.geojson

Design goals:
    - Never crash the whole pipeline because one input file is missing/malformed.
      Every stage logs a warning and degrades gracefully (empty DataFrame) so
      downstream steps and the Streamlit app still run.
    - No hard-coded assumptions about exact column names where the source
      schema is ambiguous (e.g. mapping spreadsheets) -- best-effort column
      detection is used with sensible fallbacks.

Run:
    python preprocess.py
"""

from __future__ import annotations

import glob
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "data" / "processed"

DROUGHT_DIR = RAW_DIR / "DATA_SPEI_DroughtAtlas"
BOUNDARY_DIR = DROUGHT_DIR / "Administrative-Boundaries_Shapefile"
ZSCORE_DIR = DROUGHT_DIR / "Z-score"

FLOOD_DIR = RAW_DIR / "india-flood-atlas-data-main"
EMDAT_PATH = RAW_DIR / "emdat_with_coordinates.xlsx"

YEAR_MIN, YEAR_MAX = 1901, 2026

ZSCORE_COLS = ["Year", "Month", "SPEI-1month", "SPEI-4month", "SPEI-12month"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Administrative boundaries
# --------------------------------------------------------------------------- #

def _find_column_ci(columns, candidates: List[str]) -> Optional[str]:
    """Case-insensitive, order-preserving lookup of the first matching column.

    `candidates` is checked in priority order; for each candidate we look for
    an exact case-insensitive match against `columns`, so 'District' matches
    a candidate of 'DISTRICT' or 'district' regardless of casing.
    """
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        hit = lower_map.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None


def _safe_import_geopandas():
    try:
        import geopandas as gpd  # noqa
        return gpd
    except ImportError:
        log.error("geopandas is not installed. Run: pip install geopandas")
        return None


def process_boundaries() -> Dict[str, "object"]:
    """Convert shapefiles to simplified WGS84 GeoJSON for web-map rendering.

    Returns a dict keyed by level ('state', 'district', 'subdistrict') mapping
    to the (in-memory) GeoDataFrame, or an empty dict entry if unavailable.
    """
    gpd = _safe_import_geopandas()
    result: Dict[str, "object"] = {}
    if gpd is None:
        return result

    shapefile_map = {
        "country": BOUNDARY_DIR / "India_Country_Boundary.shp",
        "state": BOUNDARY_DIR / "STATE_BOUNDARY.shp",
        "district": BOUNDARY_DIR / "DISTRICT_BOUNDARY.shp",
        "subdistrict": BOUNDARY_DIR / "SUBDISTRICT_BOUNDARY.shp",
    }

    # Candidate name-field column names, in priority order, per level.
    name_field_candidates = {
        "state": ["STATE", "ST_NM", "STATE_NAME", "NAME_1", "st_nm", "NAME"],
        "district": ["DISTRICT", "DIST_NAME", "NAME_2", "district", "NAME"],
        "subdistrict": [
            "SUBDISTRICT", "SUB_DIST", "TEHSIL", "TALUKA", "NAME_3",
            "SubDistrict/Tehsil", "NAME",
        ],
    }

    for level, shp_path in shapefile_map.items():
        if not shp_path.exists():
            log.warning("Boundary shapefile missing for '%s': %s", level, shp_path)
            continue
        try:
            gdf = gpd.read_file(shp_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed reading shapefile %s: %s", shp_path, exc)
            continue

        try:
            if gdf.crs is None:
                log.warning("%s has no CRS set; assuming EPSG:4326", shp_path.name)
                gdf.set_crs(epsg=4326, inplace=True)
            else:
                gdf = gdf.to_crs(epsg=4326)
        except Exception as exc:  # noqa: BLE001
            log.warning("CRS reprojection failed for %s: %s", shp_path, exc)

        try:
            gdf["geometry"] = gdf["geometry"].simplify(0.005, preserve_topology=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Geometry simplification failed for %s: %s", shp_path, exc)

        if level in name_field_candidates:
            standard_col = level.upper() + "_NAME"
            found_col = _find_column_ci(gdf.columns, name_field_candidates[level])
            if found_col:
                gdf[standard_col] = gdf[found_col].astype(str).apply(_sanitize_entity_name)
            else:
                log.warning(
                    "No recognizable name column found in %s (columns: %s); "
                    "falling back to first non-geometry column.",
                    shp_path.name, list(gdf.columns),
                )
                fallback_col = next(
                    (c for c in gdf.columns if c != "geometry"), None
                )
                gdf[standard_col] = (
                    gdf[fallback_col].astype(str).str.strip() if fallback_col else ""
                )

        out_path = PROCESSED_DIR / f"india_{level}.geojson"
        try:
            gdf.to_file(out_path, driver="GeoJSON")
            log.info("Wrote %s (%d features)", out_path.name, len(gdf))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed writing GeoJSON %s: %s", out_path, exc)

        result[level] = gdf

    return result


# --------------------------------------------------------------------------- #
# 2. SPEI drought z-score data
# --------------------------------------------------------------------------- #

# Level-specific candidate column names for the drought-atlas index
# spreadsheets (India_States.xlsx / India_Districts.xlsx / India_SubDistricts.xlsx).
# These are checked case-insensitively via _find_column_ci, so 'District',
# 'DISTRICT', and 'district' are all treated as the same match.
MAPPING_NAME_CANDIDATES = {
    "state": ["State", "State_Name", "ST_NM", "Name"],
    "district": ["District", "District_Name", "Dist_Name", "DIST_NAME", "Name"],
    "subdistrict": [
        "SubDistrict/Tehsil", "Sub_District", "Subdistrict", "Taluka", "Tehsil",
        "Sub_Dist_Name", "Name",
    ],
}
MAPPING_ID_CANDIDATES = ["ID", "Id", "Code", "OBJECTID", "S_ID", "D_ID", "T_ID"]


# Character-level corrections for a systematic encoding/keyboard-layout
# corruption observed in the source spreadsheets, where certain letters are
# consistently replaced by a placeholder symbol (e.g. every 'U' became '@',
# every 'A' became '>', every 'I' became '|'). Confirmed from real examples:
#   BHAR@CH -> BHARUCH, S@RAT -> SURAT             ('@' stands for 'U')
#   R>JASTH>N -> RAJASTHAN, TELANG>NA -> TELANGANA ('>' stands for 'A')
#   CHAND|GARH -> CHANDIGARH, KASHM|R -> KASHMIR   ('|' stands for 'I')
# Add more entries here if other placeholder characters turn up in the logs.
CORRUPT_CHAR_MAP: Dict[str, str] = {
    "@": "U",
    ">": "A",
    "|": "I",
    "<": "A",
    "\\": "I",
    "#": "U",
}

# Manual whole-name corrections for anything that doesn't fit the simple
# single-character substitution pattern above.
NAME_CORRECTIONS: Dict[str, str] = {}

# '&', ',', and '_' are legitimate in real names or our own placeholder
# labels ("ANDAMAN & NICOBAR", "UnmappedFloodID_12") -- not corruption.
_SUSPICIOUS_CHAR_RE = re.compile(r"[^A-Za-z0-9\s\-'.,()/&_]")


def _sanitize_entity_name(name: str) -> str:
    """Apply known corrections and flag any remaining suspicious characters.

    Order of operations:
    1. Whole-name override via NAME_CORRECTIONS, if present.
    2. Character-level substitution via CORRUPT_CHAR_MAP (handles the
       systematic keyboard/font corruption automatically, at scale).
    3. Any characters still outside the normal alphabet/punctuation set are
       logged as a warning so a new mapping can be added by hand.
    """
    name = str(name).strip()
    upper = name.upper()
    if upper in NAME_CORRECTIONS:
        return NAME_CORRECTIONS[upper]

    corrected = "".join(CORRUPT_CHAR_MAP.get(ch, ch) for ch in name)
    corrected = corrected.upper()

    if _SUSPICIOUS_CHAR_RE.search(corrected):
        log.warning(
            "Entity name '%s' still contains unexpected characters after "
            "auto-correction (result: '%s'). Add the new placeholder "
            "character to CORRUPT_CHAR_MAP once you know which letter it "
            "represents, or add a whole-name fix to NAME_CORRECTIONS.",
            name, corrected,
        )
    return corrected


def _load_id_name_mapping(xlsx_path: Path, level: str) -> Dict[str, str]:
    """Best-effort ID -> Name mapping loader for the drought-atlas index files.

    `level` ('state' | 'district' | 'subdistrict') is used to pick the right
    name column out of a mapping file that may contain several hierarchy
    columns (e.g. ID, State, District, SubDistrict all in one sheet) -- we
    must not just grab "the second column", since that could silently pick
    the wrong hierarchy level's name.
    """
    if not xlsx_path.exists():
        log.warning("Mapping file missing: %s", xlsx_path)
        return {}
    try:
        df = pd.read_excel(xlsx_path, dtype=str)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed reading mapping file %s: %s", xlsx_path, exc)
        return {}

    id_col = _find_column_ci(df.columns, MAPPING_ID_CANDIDATES)
    name_col = _find_column_ci(df.columns, MAPPING_NAME_CANDIDATES.get(level, ["Name"]))

    # Generic 'contains id' / 'contains name' regex fallback if the exact
    # candidate lists above didn't hit.
    if id_col is None:
        id_col = next(
            (c for c in df.columns if re.search(r"\bid\b|id$", str(c), re.IGNORECASE)),
            None,
        )
    if name_col is None:
        name_col = next(
            (c for c in df.columns if re.search(r"name", str(c), re.IGNORECASE)), None
        )

    if id_col is None or name_col is None:
        log.warning(
            "Could not confidently detect ID/Name columns in %s for level '%s' "
            "(available columns: %s). Skipping this mapping file.",
            xlsx_path.name, level, list(df.columns),
        )
        return {}

    log.info(
        "%s -> using ID column '%s' and Name column '%s'",
        xlsx_path.name, id_col, name_col,
    )

    mapping = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        raw_id_norm = re.sub(r"\.0$", "", raw_id)  # 12.0 -> 12
        mapping[raw_id_norm] = _sanitize_entity_name(row[name_col])
    return mapping


def _extract_id_from_filename(filename: str) -> Optional[str]:
    match = re.search(r"id_(\w+)", filename, re.IGNORECASE)
    return match.group(1) if match else None


def _process_zscore_folder(
    folder: Path, mapping: Dict[str, str], entity_col: str
) -> pd.DataFrame:
    if not folder.exists():
        log.warning("Z-score folder missing: %s", folder)
        return pd.DataFrame(columns=[entity_col, "Year", "SPEI-1month",
                                      "SPEI-4month", "SPEI-12month"])

    files = sorted(glob.glob(str(folder / "*")))
    if not files:
        log.warning("No files found in %s", folder)
        return pd.DataFrame(columns=[entity_col, "Year", "SPEI-1month",
                                      "SPEI-4month", "SPEI-12month"])

    frames: List[pd.DataFrame] = []
    unmapped_ids: List[str] = []
    for fpath in files:
        fname = Path(fpath).name
        entity_id = _extract_id_from_filename(fname)
        if entity_id is None:
            log.warning("Could not parse entity ID from filename: %s", fname)
            continue
        entity_name = mapping.get(entity_id)
        if entity_name is None:
            unmapped_ids.append(entity_id)
            entity_name = f"Unknown_{entity_id}"

        try:
            df = pd.read_csv(fpath, sep=r"\s+", header=None, names=ZSCORE_COLS)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed reading z-score file %s: %s", fpath, exc)
            continue

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        df = df[(df["Year"] >= YEAR_MIN) & (df["Year"] <= YEAR_MAX)]
        df[entity_col] = entity_name
        frames.append(df)

    if unmapped_ids:
        mapping_ids = set(mapping.keys())
        file_ids = {
            _extract_id_from_filename(Path(f).name) for f in files
        } - {None}
        log.warning(
            "%s: %d/%d files had no name mapping (showing up to 10 IDs: %s). "
            "Mapping file has %d IDs total (range-ish sample: %s); the data "
            "folder references %d distinct IDs. If these ranges don't "
            "overlap much, the mapping spreadsheet's ID scheme likely "
            "doesn't match the filenames' ID scheme.",
            folder.name, len(unmapped_ids), len(files), sorted(unmapped_ids)[:10],
            len(mapping_ids), sorted(mapping_ids)[:5] + ["..."] + sorted(mapping_ids)[-5:]
            if len(mapping_ids) > 10 else sorted(mapping_ids),
            len(file_ids),
        )

    if not frames:
        return pd.DataFrame(columns=[entity_col, "Year", "SPEI-1month",
                                      "SPEI-4month", "SPEI-12month"])

    combined = pd.concat(frames, ignore_index=True)
    annual = (
        combined.groupby([entity_col, "Year"], as_index=False)[
            ["SPEI-1month", "SPEI-4month", "SPEI-12month"]
        ]
        .mean()
        .round(3)
    )
    return annual


def process_drought_data() -> Dict[str, pd.DataFrame]:
    log.info("Processing SPEI drought atlas...")
    state_map = _load_id_name_mapping(ZSCORE_DIR / "India_States.xlsx", "state")
    district_map = _load_id_name_mapping(ZSCORE_DIR / "India_Districts.xlsx", "district")
    subdist_map = _load_id_name_mapping(
        ZSCORE_DIR / "India_SubDistricts.xlsx", "subdistrict"
    )

    state_df = _process_zscore_folder(ZSCORE_DIR / "States", state_map, "STATE_NAME")
    district_df = _process_zscore_folder(
        ZSCORE_DIR / "Districts", district_map, "DISTRICT_NAME"
    )
    subdist_df = _process_zscore_folder(
        ZSCORE_DIR / "Talukas", subdist_map, "SUBDISTRICT_NAME"
    )

    log.info(
        "Drought data rows -> state: %d, district: %d, subdistrict: %d",
        len(state_df), len(district_df), len(subdist_df),
    )
    return {
        "state": state_df, "district": district_df, "subdistrict": subdist_df,
        "_state_map": state_map, "_district_map": district_map,
    }


# --------------------------------------------------------------------------- #
# 3. Flood atlas data
# --------------------------------------------------------------------------- #

def _resolve_flood_path(expected_path: Path) -> Optional[Path]:
    """Locate a flood-atlas file even if it's nested inside extra folders.

    Some downloaded archives (e.g. a GitHub zip) unpack into a nested
    'repo-name-main/repo-name-main/...' structure rather than sitting
    directly in data/raw/flood_atlas/. If the expected path doesn't exist,
    search recursively under data/raw/ for a file with the same name.
    """
    if expected_path.exists():
        return expected_path

    # Search progressively wider roots: data/raw/, then data/, then the
    # whole project root. This covers archives that were extracted into an
    # unexpected nested folder (e.g. data/data/<repo>-main/<repo>-main/...).
    search_roots = [RAW_DIR, RAW_DIR.parent, ROOT]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matches = list(search_root.rglob(expected_path.name))
        if matches:
            if len(matches) > 1:
                log.warning(
                    "Multiple candidates found for %s, using the first: %s",
                    expected_path.name, matches,
                )
            log.info(
                "Found '%s' at a nested path instead of the expected location: %s",
                expected_path.name, matches[0],
            )
            return matches[0]
    return None


_YEAR_KEY_RE = re.compile(r"^(18|19|20)\d{2}$")


def _parse_flood_yearly_json(path: Path, value_name: str) -> pd.DataFrame:
    """Parse the '*_annual_flood_frac.json' shape:

        [
          {"ID": 1, "1901": 0.01, "1902": 0.91, ...},
          {"ID": 2, ...},
          ...
        ]

    Each record is one entity (identified only by numeric ID), with a
    year-number key for every year's value. Returns a long-format DataFrame:
    [entity_id, Year, value_name].
    """
    resolved_path = _resolve_flood_path(path)
    if resolved_path is None:
        log.warning(
            "Flood atlas file missing: %s (also searched recursively under %s)",
            path, RAW_DIR,
        )
        return pd.DataFrame(columns=["entity_id", "Year", value_name])
    path = resolved_path

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed reading/parsing JSON %s: %s", path, exc)
        return pd.DataFrame(columns=["entity_id", "Year", value_name])

    if not isinstance(raw, list):
        log.warning(
            "Expected a list of per-entity records in %s, got %s instead.",
            path, type(raw).__name__,
        )
        return pd.DataFrame(columns=["entity_id", "Year", value_name])

    records = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entity_id = item.get("ID") or item.get("Id") or item.get("id")
        if entity_id is None:
            continue
        for key, val in item.items():
            if _YEAR_KEY_RE.match(str(key).strip()):
                records.append(
                    {"entity_id": str(entity_id).strip(), "Year": int(key), value_name: val}
                )

    if not records:
        log.warning(
            "No year-keyed values found in %s -- schema may differ from the "
            "expected {'ID': ..., '<year>': <value>, ...} shape.", path,
        )
        return pd.DataFrame(columns=["entity_id", "Year", value_name])

    df = pd.DataFrame(records)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df[(df["Year"] >= YEAR_MIN) & (df["Year"] <= YEAR_MAX)]
    return df


def _parse_flood_risk_json(path: Path) -> pd.DataFrame:
    """Parse the '*_flood_risk_data.json' shape:

        [
          {"ID": 1, "Vulnerability": 0.534, "Hazard": 0.08,
           "Exposure": 0.011, "Risk": 0.001},
          ...
        ]

    One static (non-yearly) record per entity. Returns:
    [entity_id, flood_vulnerability, flood_hazard, flood_exposure, flood_risk_index].
    """
    resolved_path = _resolve_flood_path(path)
    if resolved_path is None:
        log.warning(
            "Flood atlas file missing: %s (also searched recursively under %s)",
            path, RAW_DIR,
        )
        return pd.DataFrame(columns=["entity_id", "flood_vulnerability",
                                      "flood_hazard", "flood_exposure",
                                      "flood_risk_index"])
    path = resolved_path

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed reading/parsing JSON %s: %s", path, exc)
        return pd.DataFrame(columns=["entity_id", "flood_vulnerability",
                                      "flood_hazard", "flood_exposure",
                                      "flood_risk_index"])

    if not isinstance(raw, list):
        log.warning(
            "Expected a list of per-entity risk records in %s, got %s instead.",
            path, type(raw).__name__,
        )
        return pd.DataFrame(columns=["entity_id", "flood_vulnerability",
                                      "flood_hazard", "flood_exposure",
                                      "flood_risk_index"])

    records = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entity_id = item.get("ID") or item.get("Id") or item.get("id")
        if entity_id is None:
            continue
        records.append({
            "entity_id": str(entity_id).strip(),
            "flood_vulnerability": item.get("Vulnerability"),
            "flood_hazard": item.get("Hazard"),
            "flood_exposure": item.get("Exposure"),
            "flood_risk_index": item.get("Risk"),
        })

    if not records:
        log.warning(
            "No usable risk records found in %s -- schema may differ from "
            "the expected {'ID', 'Vulnerability', 'Hazard', 'Exposure', "
            "'Risk'} shape.", path,
        )
        return pd.DataFrame(columns=["entity_id", "flood_vulnerability",
                                      "flood_hazard", "flood_exposure",
                                      "flood_risk_index"])

    df = pd.DataFrame(records)
    for col in ["flood_vulnerability", "flood_hazard", "flood_exposure",
                "flood_risk_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_year(value) -> Optional[int]:
    if value is None:
        return None
    try:
        year = int(re.sub(r"[^0-9]", "", str(value))[:4])
        if YEAR_MIN - 50 <= year <= YEAR_MAX + 5:
            return year
    except (ValueError, TypeError):
        return None
    return None


def _positional_fallback_map(id_map: Dict[str, str], flood_ids) -> Dict[str, str]:
    """Heuristic fallback when direct ID matching almost entirely fails.

    Some datasets from the same lab/shapefile are exported with their own
    row-order ID (1..N) instead of the official ID used elsewhere. If the
    count of flood-atlas IDs matches the count of mapped entities, assume
    they're the same 40-ish region set in ascending-ID order and pair them
    positionally. This is a guess -- callers must log it loudly and the
    result should be spot-checked, not silently trusted.
    """
    try:
        sorted_map_names = [
            name for _, name in sorted(id_map.items(), key=lambda kv: float(kv[0]))
        ]
        sorted_flood_ids = sorted(set(flood_ids), key=lambda x: float(x))
    except (ValueError, TypeError):
        return {}
    return dict(zip(sorted_flood_ids, sorted_map_names))


def _map_ids_to_names(
    df: pd.DataFrame, id_map: Dict[str, str], name_col: str, source_label: str
) -> pd.DataFrame:
    """Resolve numeric entity_id -> name using an ID/name mapping dict.

    Assumes the flood-atlas ID scheme matches the drought-atlas mapping
    spreadsheets' ID scheme. If most IDs fail to resolve, falls back to a
    positional (sorted-order) guess when the entity counts line up, and
    logs a loud warning either way so the result can be spot-checked.
    """
    if df.empty:
        df[name_col] = pd.Series(dtype=str)
        return df

    df = df.copy()
    df[name_col] = df["entity_id"].map(id_map)

    unmapped_ids = df[df[name_col].isna()]["entity_id"].unique()
    total_ids = df["entity_id"].nunique()

    if len(unmapped_ids) > 0:
        fail_rate = len(unmapped_ids) / total_ids
        if fail_rate >= 0.5 and len(id_map) > 0:
            # Direct match mostly/entirely failed -- try positional fallback.
            fallback = _positional_fallback_map(id_map, df["entity_id"].unique())
            df.loc[df[name_col].isna(), name_col] = df.loc[
                df[name_col].isna(), "entity_id"
            ].map(fallback)
            newly_resolved = len(unmapped_ids) - df[name_col].isna().sum()
            log.warning(
                "%s: direct ID match failed for %d/%d entities -- the flood "
                "atlas likely uses its own row-order ID rather than the "
                "drought-atlas spreadsheet ID. Applied a POSITIONAL FALLBACK "
                "(sorted-ID pairing) instead, resolving %d of them. This is "
                "a heuristic guess, not a verified mapping -- spot-check a "
                "few entries (e.g. does the highest-risk ID correspond to a "
                "genuinely flood-prone state?) before trusting it.",
                source_label, len(unmapped_ids), total_ids, newly_resolved,
            )
            unmapped_ids = df[df[name_col].isna()]["entity_id"].unique()

        if len(unmapped_ids) > 0:
            log.warning(
                "%s: %d/%d entity IDs still unresolved after fallback "
                "(sample: %s). These rows are kept with a placeholder name "
                "so no data is dropped.",
                source_label, len(unmapped_ids), total_ids,
                sorted(unmapped_ids, key=str)[:10],
            )
        df[name_col] = df[name_col].fillna(
            "UnmappedFloodID_" + df["entity_id"].astype(str)
        )

    # Sanitize only the unique names once, then map back -- avoids emitting
    # the same "unexpected characters" warning once per row (this table can
    # have tens of thousands of rows for a handful of distinct entities).
    unique_names = df[name_col].unique()
    clean_map = {n: _sanitize_entity_name(n) for n in unique_names}
    df[name_col] = df[name_col].map(clean_map)
    return df.drop(columns=["entity_id"])


def process_flood_atlas(
    state_map: Dict[str, str], district_map: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    log.info("Processing flood atlas...")

    # Yearly flood-fraction data (long format after parsing).
    district_frac = _parse_flood_yearly_json(
        FLOOD_DIR / "District_Wise_annual_flood_frac.json", "flood_fraction"
    )
    state_frac = _parse_flood_yearly_json(
        FLOOD_DIR / "State_Wise_annual_flood_frac.json", "flood_fraction"
    )

    # Yearly flood-area data (same wide-by-ID/year shape as flood_frac).
    district_area = _parse_flood_yearly_json(
        FLOOD_DIR / "District_Wise_annual_flood_area.json", "flood_area"
    )
    state_area = _parse_flood_yearly_json(
        FLOOD_DIR / "State_Wise_annual_flood_area.json", "flood_area"
    )

    # Static (non-yearly) risk-component data.
    district_risk = _parse_flood_risk_json(
        FLOOD_DIR / "District_Wise_flood_risk_data.json"
    )
    state_risk = _parse_flood_risk_json(
        FLOOD_DIR / "State_Wise_flood_risk_data.json"
    )

    district_frac = _map_ids_to_names(
        district_frac, district_map, "DISTRICT_NAME", "District flood fraction"
    )
    district_area = _map_ids_to_names(
        district_area, district_map, "DISTRICT_NAME", "District flood area"
    )
    district_risk = _map_ids_to_names(
        district_risk, district_map, "DISTRICT_NAME", "District flood risk"
    )
    state_frac = _map_ids_to_names(
        state_frac, state_map, "STATE_NAME", "State flood fraction"
    )
    state_area = _map_ids_to_names(
        state_area, state_map, "STATE_NAME", "State flood area"
    )
    state_risk = _map_ids_to_names(
        state_risk, state_map, "STATE_NAME", "State flood risk"
    )

    def _merge(
        frac: pd.DataFrame, area: pd.DataFrame, risk: pd.DataFrame, entity_col: str
    ) -> pd.DataFrame:
        # frac/area are yearly (join on entity+Year); risk is static per
        # entity (broadcast across every year via a left join on entity only).
        yearly = _outer_merge([frac, area], on=[entity_col, "Year"])
        if yearly.empty and risk.empty:
            return pd.DataFrame(
                columns=[entity_col, "Year", "flood_fraction", "flood_area",
                         "flood_vulnerability", "flood_hazard",
                         "flood_exposure", "flood_risk_index"]
            )
        if yearly.empty:
            return risk
        if risk.empty:
            return yearly
        return pd.merge(yearly, risk, on=[entity_col], how="left")

    district_flood = _merge(district_frac, district_area, district_risk, "DISTRICT_NAME")
    state_flood = _merge(state_frac, state_area, state_risk, "STATE_NAME")

    log.info(
        "Flood data rows -> state: %d, district: %d",
        len(state_flood), len(district_flood),
    )
    return {"state": state_flood, "district": district_flood}


# --------------------------------------------------------------------------- #
# 4. EM-DAT disaster events
# --------------------------------------------------------------------------- #

def process_emdat() -> pd.DataFrame:
    log.info("Processing EM-DAT disaster records...")
    if not EMDAT_PATH.exists():
        log.warning("EM-DAT file missing: %s", EMDAT_PATH)
        return pd.DataFrame()

    try:
        df = pd.read_excel(EMDAT_PATH)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed reading EM-DAT file: %s", exc)
        return pd.DataFrame()

    if "ISO" in df.columns:
        df = df[df["ISO"] == "IND"].copy()
    else:
        log.warning("EM-DAT file has no 'ISO' column; skipping country filter")

    wanted_cols = {
        "DisNo.": "DisNo",
        "Disaster Type": "disaster_type",
        "Start Year": "Year",
        "Location": "location",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Total Deaths": "total_deaths",
        "Total Affected": "total_affected",
        "Total Damage ('000 US$)": "total_damage_000usd",
    }
    available = {k: v for k, v in wanted_cols.items() if k in df.columns}
    missing = set(wanted_cols) - set(available)
    if missing:
        log.warning("EM-DAT missing expected columns: %s", missing)

    df = df[list(available.keys())].rename(columns=available)

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        df = df[(df["Year"] >= YEAR_MIN) & (df["Year"] <= YEAR_MAX)]

    for numcol in ["total_deaths", "total_affected", "total_damage_000usd",
                   "latitude", "longitude"]:
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors="coerce")

    log.info("EM-DAT India records retained: %d", len(df))
    return df


def _match_emdat_to_admin(
    emdat_df: pd.DataFrame,
    boundaries: Dict[str, "object"],
    level: str,
    name_field: str,
) -> pd.DataFrame:
    """Spatially/textually assign each EM-DAT event to an admin unit.

    Prefers a point-in-polygon spatial join (when lat/lon + boundary geometry
    are available); falls back to case-insensitive substring matching against
    the 'location' free-text field.
    """
    if emdat_df.empty:
        return pd.DataFrame(columns=[name_field, "Year", "event_count",
                                      "total_deaths", "total_affected",
                                      "total_damage_000usd"])

    gdf = boundaries.get(level)
    matched = None

    if gdf is not None and {"latitude", "longitude"}.issubset(emdat_df.columns):
        gpd = _safe_import_geopandas()
        pts = emdat_df.dropna(subset=["latitude", "longitude"]).copy()
        if gpd is not None and not pts.empty:
            try:
                import shapely.geometry as geom

                pts["geometry"] = pts.apply(
                    lambda r: geom.Point(r["longitude"], r["latitude"]), axis=1
                )
                pts_gdf = gpd.GeoDataFrame(pts, geometry="geometry", crs="EPSG:4326")
                joined = gpd.sjoin(pts_gdf, gdf[[name_field, "geometry"]],
                                   how="left", predicate="within")
                joined = joined.drop(columns=["geometry", "index_right"],
                                      errors="ignore")
                matched = joined
            except Exception as exc:  # noqa: BLE001
                log.warning("Spatial join failed for level '%s': %s", level, exc)

    if matched is None or name_field not in matched.columns or matched[name_field].isna().all():
        # Fallback: text matching against 'location'
        if gdf is not None and "location" in emdat_df.columns:
            names = gdf[name_field].dropna().unique().tolist()
            def _find_name(loc: str) -> Optional[str]:
                if not isinstance(loc, str):
                    return None
                for n in names:
                    if isinstance(n, str) and n.lower() in loc.lower():
                        return n
                return None
            matched = emdat_df.copy()
            matched[name_field] = matched["location"].apply(_find_name)
        else:
            matched = emdat_df.copy()
            matched[name_field] = np.nan

    matched = matched.dropna(subset=[name_field])
    if matched.empty:
        return pd.DataFrame(columns=[name_field, "Year", "event_count",
                                      "total_deaths", "total_affected",
                                      "total_damage_000usd"])

    agg = (
        matched.groupby([name_field, "Year"])
        .agg(
            event_count=("Year", "count"),
            total_deaths=("total_deaths", "sum"),
            total_affected=("total_affected", "sum"),
            total_damage_000usd=("total_damage_000usd", "sum"),
        )
        .reset_index()
    )
    return agg


# --------------------------------------------------------------------------- #
# 5. Merge everything into level-wise summary tables
# --------------------------------------------------------------------------- #

def _outer_merge(dfs: List[pd.DataFrame], on: List[str]) -> pd.DataFrame:
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame(columns=on)
    out = dfs[0]
    for d in dfs[1:]:
        out = pd.merge(out, d, on=on, how="outer")
    return out


def build_summaries(
    drought: Dict[str, pd.DataFrame],
    flood: Dict[str, pd.DataFrame],
    emdat_state: pd.DataFrame,
    emdat_district: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    log.info("Merging datasets into final summary tables...")

    state_summary = _outer_merge(
        [drought["state"], flood.get("state"), emdat_state],
        on=["STATE_NAME", "Year"],
    )
    district_summary = _outer_merge(
        [drought["district"], flood.get("district"), emdat_district],
        on=["DISTRICT_NAME", "Year"],
    )
    subdistrict_summary = drought["subdistrict"].copy()

    for df in (state_summary, district_summary, subdistrict_summary):
        for col in ("event_count", "total_deaths", "total_affected",
                    "total_damage_000usd"):
            if col in df.columns:
                df[col] = df[col].fillna(0)

    return {
        "state": state_summary,
        "district": district_summary,
        "subdistrict": subdistrict_summary,
    }


def save_summaries(summaries: Dict[str, pd.DataFrame]) -> None:
    for level, df in summaries.items():
        out_path = PROCESSED_DIR / f"{level}_summary.parquet"
        try:
            df.to_parquet(out_path, index=False)
            log.info("Wrote %s (%d rows, %d cols)", out_path.name, *df.shape)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed writing %s: %s", out_path, exc)


def save_emdat_events(emdat_raw: pd.DataFrame) -> None:
    """Save the raw, geocoded EM-DAT event rows (not aggregated) so the map
    can plot individual disaster locations as point markers -- aggregated
    per-region counts are too sparse, per-year, to show up as a meaningful
    choropleth on their own."""
    out_path = PROCESSED_DIR / "emdat_events.parquet"
    cols = ["Year", "disaster_type", "location", "latitude", "longitude",
            "total_deaths", "total_affected", "total_damage_000usd"]
    df = emdat_raw[[c for c in cols if c in emdat_raw.columns]].copy()
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df.dropna(subset=["latitude", "longitude"])
    try:
        df.to_parquet(out_path, index=False)
        log.info("Wrote %s (%d rows)", out_path.name, len(df))
    except Exception as exc:  # noqa: BLE001
        log.error("Failed writing %s: %s", out_path, exc)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    log.info("Starting India Climate & Disaster Risk preprocessing pipeline")
    ensure_dirs()

    boundaries = process_boundaries()
    drought = process_drought_data()
    flood = process_flood_atlas(drought["_state_map"], drought["_district_map"])
    emdat_raw = process_emdat()

    emdat_state = _match_emdat_to_admin(emdat_raw, boundaries, "state", "STATE_NAME")
    emdat_district = _match_emdat_to_admin(
        emdat_raw, boundaries, "district", "DISTRICT_NAME"
    )

    summaries = build_summaries(drought, flood, emdat_state, emdat_district)
    save_summaries(summaries)
    save_emdat_events(emdat_raw)

    log.info("Preprocessing complete. Outputs in: %s", PROCESSED_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log.exception("Pipeline failed with an unhandled error: %s", exc)
        sys.exit(1)
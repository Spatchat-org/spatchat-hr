# storage.py
import os
import shutil
import json
import zipfile
import pandas as pd
from collections import defaultdict
from shapely.geometry import mapping

# ---- Global analysis state (shared across modules) ----
# MCP: {animal_id: {percent: {"polygon": shapely.Polygon, "area": float}}}
mcp_results = {}

# KDE: {animal_id: {percent: {"contour": shapely (Multi)Polygon or None,
#                             "area": float,
#                             "geotiff": str (path),
#                             "geojson": str (optional path)}}}
kde_results = {}

# LoCoH: full result dict from estimators/locoh.compute_locoh(...)
# {
#   "method": "k"|"a"|"r",
#   "isopleths": [50,95,...],
#   "animals": {
#       "<animal_id>": {
#           "n_points": int,
#           "isopleths": [
#               {"isopleth": 50, "area_sq_km": float, "geometry": <GeoJSON>},
#               ...
#           ],
#           "facets": [
#               {"cum_percent": 50, "area_sq_km": float, "geometry": <GeoJSON>},
#               ...
#           ]
#       }, ...
#   }
# }
locoh_results = None  # or dict

# dBBMM: {animal_id: {"geotiff": str, "isopleths": [{percent, area_sq_km, geometry}]}}
dbbmm_results = {}

# requested sets (kept for UI and summaries)
requested_percents = set()        # MCP
requested_kde_percents = set()    # KDE
requested_dbbmm_percents = set()  # dBBMM

# Cached dataset and headers
cached_df = None
cached_headers = []

# --- transient detection summary for chat ---
last_detection_summary = ""

# ---- Accessors expected by app.py ----
def get_cached_df():
    return cached_df

def set_cached_df(df):
    global cached_df
    cached_df = df

def get_cached_headers():
    return cached_headers

def set_cached_headers(headers):
    global cached_headers
    cached_headers = list(headers or [])

def set_detection_summary(text: str):
    global last_detection_summary
    last_detection_summary = text or ""

def get_detection_summary() -> str:
    return last_detection_summary

def get_locoh_results():
    return locoh_results

def set_locoh_results(res: dict | None):
    global locoh_results
    locoh_results = res

def get_dbbmm_results():
    return dbbmm_results

def set_dbbmm_results(res: dict | None):
    dbbmm_results.clear()
    if isinstance(res, dict):
        dbbmm_results.update(res)

# ---- Lifecycle helpers ----
def clear_all_results():
    """
    Reset analysis outputs and the outputs/ folder WITHOUT rebinding globals.
    This preserves references imported elsewhere (e.g., in app.py).
    """
    mcp_results.clear()
    kde_results.clear()
    dbbmm_results.clear()
    requested_percents.clear()
    requested_kde_percents.clear()
    requested_dbbmm_percents.clear()

    global locoh_results
    locoh_results = None

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------------------------------
# Estimator-specific writers (consolidated outputs only)
# --------------------------------------------------------------------------------------
def _write_mcp_assets(rows_accum: list[tuple], outdir: str):
    """
    rows_accum += (animal_id, 'MCP-<percent>', area_km2)
    Writes a single merged GeoJSON for all MCP polygons if present.
    """
    features = []
    for animal, percents in mcp_results.items():
        for percent, v in percents.items():
            area = float(v.get("area", 0.0))
            rows_accum.append((animal, f"MCP-{percent}", area))
            poly = v.get("polygon")
            if poly is not None:
                features.append({
                    "type": "Feature",
                    "properties": {"animal_id": str(animal), "percent": int(percent), "area_km2": area},
                    "geometry": mapping(poly)
                })

    if features:
        fc = {"type": "FeatureCollection", "features": features}
        with open(os.path.join(outdir, "mcps_all.geojson"), "w") as f:
            json.dump(fc, f)

def _write_kde_assets(rows_accum: list[tuple], outdir: str):
    """
    rows_accum += (animal_id, 'KDE-<percent>', area_km2)
    Writes consolidated files only:
      - kde_index.json (areas + pointers to rasters/contours)
      - kdes_all.geojson (ALL animals × isopleths, if geometries are available)
    Falls back to in-memory shapely contours if per-animal GeoJSON paths are absent.
    """
    index = {"animals": {}}
    any_kde = False

    # Accumulator for consolidated output
    features_all = []
    # NEW: track per-animal GeoJSONs we ingest so we can delete them after consolidation
    to_delete_paths = set()
    outdir_abs = os.path.abspath(outdir)

    for animal, percents in kde_results.items():
        index["animals"][animal] = {}
        for percent, v in percents.items():
            any_kde = True
            area = float(v.get("area", 0.0))
            rows_accum.append((animal, f"KDE-{percent}", area))
            index["animals"][animal][str(percent)] = {
                "area_km2": area,
                "geotiff": v.get("geotiff"),
                "geojson": v.get("geojson"),
            }

            # Prefer existing GeoJSON path if present; else serialize contour geometry
            feat_list = []
            gj_path = v.get("geojson")
            if gj_path and os.path.exists(gj_path):
                try:
                    with open(gj_path, "r") as f:
                        gj = json.load(f)
                    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
                        feat_list = gj.get("features", []) or []
                    elif isinstance(gj, dict) and gj.get("type") == "Feature":
                        feat_list = [gj]
                    # Mark for deletion only if it's inside outputs/
                    gj_abs = os.path.abspath(gj_path)
                    if gj_abs.startswith(outdir_abs + os.sep):
                        to_delete_paths.add(gj_abs)
                except Exception:
                    feat_list = []

            if not feat_list:
                contour = v.get("contour")
                if contour is not None:
                    try:
                        feat_list = [{
                            "type": "Feature",
                            "properties": {},
                            "geometry": mapping(contour),
                        }]
                    except Exception:
                        feat_list = []

            # Normalize properties and accumulate
            for feat in feat_list:
                if not isinstance(feat, dict):
                    continue
                props = feat.setdefault("properties", {})
                props["animal_id"] = str(animal)
                props["percent"] = int(percent)
                props["area_km2"] = area
                features_all.append(feat)

    if any_kde:
        with open(os.path.join(outdir, "kde_index.json"), "w") as f:
            json.dump(index, f, indent=2)

    if features_all:
        fc_all = {"type": "FeatureCollection", "features": features_all}
        with open(os.path.join(outdir, "kdes_all.geojson"), "w") as f:
            json.dump(fc_all, f)

        # NEW: remove per-animal KDE GeoJSONs we ingested so they won't be zipped
        for p in sorted(to_delete_paths):
            try:
                os.remove(p)
            except OSError:
                pass

def _write_locoh_assets(rows_accum: list[tuple], outdir: str):
    """
    rows_accum += (animal_id, 'LoCoH-<isopleth>', area_km2)
    Writes consolidated files only:
      - locoh_results.json               (full object: envelopes + facets)
      - locoh_envelopes_all.geojson      (ALL animals × isopleths)
      - locoh_facets_all.geojson         (ALL animals facets)
    """
    if not locoh_results or not isinstance(locoh_results, dict):
        return

    # Full results (for reproducibility/programmatic use)
    with open(os.path.join(outdir, "locoh_results.json"), "w") as f:
        json.dump(locoh_results, f)

    animals = (locoh_results.get("animals") or {})

    # Accumulators for consolidated files
    envelope_features = []   # all animals×isopleths
    facets_features = []     # all animals facets

    for animal_id, data in animals.items():
        # Envelopes (single-piece per isopleth)
        for item in data.get("isopleths", []):
            iso = int(item.get("isopleth"))
            area = float(item.get("area_sq_km", 0.0))
            rows_accum.append((animal_id, f"LoCoH-{iso}", area))

            gj = item.get("geometry")
            if gj:
                envelope_features.append({
                    "type": "Feature",
                    "properties": {"animal_id": str(animal_id), "isopleth": iso, "area_km2": area},
                    "geometry": gj,
                })

        # Facets (tiny hulls; many features)
        for fct in (data.get("facets") or []):
            facets_features.append({
                "type": "Feature",
                "properties": {
                    "animal_id": str(animal_id),
                    "cum_percent": int(fct.get("cum_percent", 0)),
                    "area_km2": float(fct.get("area_sq_km", 0.0)),
                },
                "geometry": fct.get("geometry"),
            })

    # Consolidated files
    if envelope_features:
        fc_env_all = {"type": "FeatureCollection", "features": envelope_features}
        with open(os.path.join(outdir, "locoh_envelopes_all.geojson"), "w") as f:
            json.dump(fc_env_all, f)

    if facets_features:
        fc_fac_all = {"type": "FeatureCollection", "features": facets_features}
        with open(os.path.join(outdir, "locoh_facets_all.geojson"), "w") as f:
            json.dump(fc_fac_all, f)

def _write_dbbmm_assets(rows_accum: list[tuple], outdir: str):
    """
    rows_accum += (animal_id, 'dBBMM-<percent>', area_km2)
    Writes consolidated files only:
      - dbbmm_index.json      (areas + pointers)
      - dbbmms_all.geojson    (ALL animals × isopleths)
    """
    index = {"animals": {}}
    any_bb = False
    features_all = []

    for animal, data in dbbmm_results.items():
        any_bb = True

        # Accept dict or DBBMMResult dataclass
        if isinstance(data, dict):
            geotiff = data.get("geotiff")
            iso_list = data.get("isopleths", []) or []
        else:
            geotiff = getattr(data, "geotiff", None)
            iso_list = getattr(data, "isopleths", []) or []

        index["animals"][animal] = {"geotiff": geotiff, "isopleths": []}

        for item in iso_list:
            p = int(item.get("percent"))
            area = float(item.get("area_sq_km", 0.0))
            index["animals"][animal]["isopleths"].append({"percent": p, "area_km2": area})
            rows_accum.append((animal, f"dBBMM-{p}", area))

            gj = item.get("geometry")
            if gj:
                features_all.append({
                    "type": "Feature",
                    "properties": {"animal_id": str(animal), "percent": p, "area_km2": area},
                    "geometry": gj,
                })

    if any_bb:
        with open(os.path.join(outdir, "dbbmm_index.json"), "w") as f:
            json.dump(index, f, indent=2)

    if features_all:
        fc_all = {"type": "FeatureCollection", "features": features_all}
        with open(os.path.join(outdir, "dbbmms_all.geojson"), "w") as f:
            json.dump(fc_all, f)

# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def save_all_mcps_zip():
    """
    Writes (under ./outputs):
      - mcps_all.geojson
      - kde_index.json, kdes_all.geojson
      - locoh_results.json, locoh_envelopes_all.geojson, locoh_facets_all.geojson
      - dbbmm_index.json, dbbmms_all.geojson
      - home_range_areas.csv
      - spatchat_results.zip              (zip of everything in outputs/)
    Returns path to the zip.
    """
    os.makedirs("outputs", exist_ok=True)
    rows: list[tuple] = []
    outdir = "outputs"

    # Per-estimator writers (consolidated only)
    _write_mcp_assets(rows, outdir)
    _write_kde_assets(rows, outdir)
    _write_locoh_assets(rows, outdir)
    _write_dbbmm_assets(rows, outdir)

    # Areas CSV (one table for all estimators)
    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        df.sort_values(["animal_id", "type"], inplace=True)
        df.to_csv(os.path.join(outdir, "home_range_areas.csv"), index=False)

    # Zip everything in outputs/
    archive = os.path.join(outdir, "spatchat_results.zip")
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(outdir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, outdir)
                zipf.write(full_path, arcname=rel_path)

    print("ZIP written:", archive)
    return archive

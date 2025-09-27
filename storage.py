# storage.py
import os
import shutil
import json
import zipfile
import pandas as pd
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
#           ]
#       }, ...
#   }
# }
locoh_results = None  # or dict

# requested sets (kept for UI and summaries)
requested_percents = set()        # MCP
requested_kde_percents = set()    # KDE

# Cached dataset and headers
cached_df = None
cached_headers = []

# --- transient detection summary for chat ---
last_detection_summary = ""
_dataset_brief = ""

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

def set_dataset_brief(text: str):
    global _dataset_brief
    _dataset_brief = text or ""

def get_dataset_brief() -> str:
    return _dataset_brief

def set_detection_summary(text: str):
    global last_detection_summary
    last_detection_summary = text or ""

def get_detection_summary() -> str:
    return last_detection_summary

# ---- LoCoH accessors (used by app.py optionally) ----
def get_locoh_results():
    return locoh_results

def set_locoh_results(res: dict | None):
    global locoh_results
    locoh_results = res

# ---- Lifecycle helpers ----
def clear_all_results():
    """
    Reset analysis outputs and the outputs/ folder WITHOUT rebinding globals.
    This preserves references imported elsewhere (e.g., in app.py).
    """
    mcp_results.clear()
    kde_results.clear()
    requested_percents.clear()
    requested_kde_percents.clear()

    global locoh_results
    locoh_results = None

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------------------------------
# Estimator-specific writers (small, focused)
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
                    "properties": {"animal_id": animal, "percent": int(percent), "area_km2": area},
                    "geometry": mapping(poly)
                })

    if features:
        geojson = {"type": "FeatureCollection", "features": features}
        with open(os.path.join(outdir, "mcps_all.geojson"), "w") as f:
            json.dump(geojson, f)

def _write_kde_assets(rows_accum: list[tuple], outdir: str):
    """
    rows_accum += (animal_id, 'KDE-<percent>', area_km2)
    We do not duplicate large rasters here (they already exist on disk);
    but we can write a compact index.json for convenience.
    """
    index = {"animals": {}}
    any_kde = False

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

    if any_kde:
        with open(os.path.join(outdir, "kde_index.json"), "w") as f:
            json.dump(index, f, indent=2)

def _write_locoh_assets(rows_accum: list[tuple], outdir: str):
    """
    rows_accum += (animal_id, 'LoCoH-<isopleth>', area_km2)
    Writes locoh_results.json, plus optional per-animal/isopleth FeatureCollections.
    """
    if not locoh_results or not isinstance(locoh_results, dict):
        return

    # Write the full LoCoH result for reproducibility
    with open(os.path.join(outdir, "locoh_results.json"), "w") as f:
        json.dump(locoh_results, f)

    animals = (locoh_results.get("animals") or {})
    for animal_id, data in animals.items():
        for item in data.get("isopleths", []):
            iso = int(item.get("isopleth"))
            area = float(item.get("area_sq_km", 0.0))
            rows_accum.append((animal_id, f"LoCoH-{iso}", area))

            # Optional: write one small GeoJSON per isopleth for GIS users
            gj = item.get("geometry")
            if gj:
                feat = {
                    "type": "Feature",
                    "properties": {"animal_id": animal_id, "isopleth": iso, "area_km2": area},
                    "geometry": gj,
                }
                fc = {"type": "FeatureCollection", "features": [feat]}
                fname = f"locoh_{str(animal_id).replace(' ', '_')}_{iso}.geojson"
                with open(os.path.join(outdir, fname), "w") as f:
                    json.dump(fc, f)

# --------------------------------------------------------------------------------------
# Orchestrator (keeps your existing name/signature for compatibility)
# --------------------------------------------------------------------------------------
def save_all_mcps_zip():
    """
    Writes (under ./outputs):
      - mcps_all.geojson                  (if MCP exists)
      - kde_index.json                    (if KDE exists; points to external rasters/contours)
      - locoh_results.json                (if LoCoH exists)
      - locoh_<animal>_<iso>.geojson      (optional per-isopleth exports)
      - home_range_areas.csv              (areas for MCP + KDE + LoCoH)
      - spatchat_results.zip              (zip of everything in outputs/)
    Returns path to the zip.
    """
    os.makedirs("outputs", exist_ok=True)
    rows: list[tuple] = []
    outdir = "outputs"

    # Per-estimator writers
    _write_mcp_assets(rows, outdir)
    _write_kde_assets(rows, outdir)
    _write_locoh_assets(rows, outdir)

    # Areas CSV (one table for all estimators)
    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        # Stable ordering: animal, then type
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

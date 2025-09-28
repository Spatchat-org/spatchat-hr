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

dbbmm_results = {}              # {animal_id: {"geotiff": str, "isopleths": [{percent, area_sq_km, geometry}]}}

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
    Writes:
      - locoh_results.json               (full object: envelopes + facets)
      - locoh_<animal>_<iso>.geojson     (per-animal, per-isopleth ENVELOPE)
      - locoh_facets_<animal>.geojson    (per-animal FACETS: tiny local hulls)
      - locoh_envelopes.geojson          (ALL animals×isopleth envelopes in one file)
      - locoh_facets.geojson             (ALL facets in one file)
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
        # ----- Envelopes (single-piece per isopleth) -----
        for item in data.get("isopleths", []):
            iso = int(item.get("isopleth"))
            area = float(item.get("area_sq_km", 0.0))
            rows_accum.append((animal_id, f"LoCoH-{iso}", area))

            gj = item.get("geometry")
            if gj:
                feat_env = {
                    "type": "Feature",
                    "properties": {"animal_id": animal_id, "isopleth": iso, "area_km2": area},
                    "geometry": gj,
                }
                envelope_features.append(feat_env)

                # Per-animal, per-isopleth file (kept for convenience)
                fc_env_one = {"type": "FeatureCollection", "features": [feat_env]}
                fname_env_one = f"locoh_{str(animal_id).replace(' ', '_')}_{iso}.geojson"
                with open(os.path.join(outdir, fname_env_one), "w") as f:
                    json.dump(fc_env_one, f)

        # ----- Facets (tiny hulls; many features) -----
        animal_facets = []
        for fct in (data.get("facets") or []):
            feat_facet = {
                "type": "Feature",
                "properties": {
                    "animal_id": animal_id,
                    "cum_percent": int(fct.get("cum_percent", 0)),
                    "area_km2": float(fct.get("area_sq_km", 0.0)),
                },
                "geometry": fct.get("geometry"),
            }
            animal_facets.append(feat_facet)
            facets_features.append(feat_facet)

        # Per-animal facets file (only if present)
        if animal_facets:
            fc_facets_one = {"type": "FeatureCollection", "features": animal_facets}
            fname_facets_one = f"locoh_facets_{str(animal_id).replace(' ', '_')}.geojson"
            with open(os.path.join(outdir, fname_facets_one), "w") as f:
                json.dump(fc_facets_one, f)

    # ----- Consolidated files -----
    if envelope_features:
        fc_env_all = {"type": "FeatureCollection", "features": envelope_features}
        with open(os.path.join(outdir, "locoh_envelopes.geojson"), "w") as f:
            json.dump(fc_env_all, f)

    if facets_features:
        fc_fac_all = {"type": "FeatureCollection", "features": facets_features}
        with open(os.path.join(outdir, "locoh_facets.geojson"), "w") as f:
            json.dump(fc_fac_all, f)

def _write_dbbmm_assets(rows_accum: list[tuple], outdir: str):
    index = {"animals": {}}
    any_bb = False

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
            rows_accum.append((animal, f"dBBMM-{p}", area))
            index["animals"][animal]["isopleths"].append({"percent": p, "area_km2": area})

            gj = item.get("geometry")
            if gj:
                fc = {"type": "FeatureCollection", "features": [{
                    "type": "Feature",
                    "properties": {"animal_id": animal, "percent": p, "area_km2": area},
                    "geometry": gj,
                }]}
                fname = f"dbbmm_{str(animal).replace(' ', '_')}_{p}.geojson"
                with open(os.path.join(outdir, fname), "w") as f:
                    json.dump(fc, f)

    if any_bb:
        with open(os.path.join(outdir, "dbbmm_index.json"), "w") as f:
            json.dump(index, f, indent=2)

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

# storage.py
import os
import shutil
import json
import zipfile
import pandas as pd
from shapely.geometry import mapping

# ---- Global analysis state (shared across modules) ----
mcp_results = {}                 # {animal_id: {percent: {"polygon": shapely.Polygon, "area": float}}}
kde_results = {}                 # {animal_id: {percent: {"contour": shapely (Multi)Polygon, "area": float, "geotiff": str, "geojson": str}}}
requested_percents = set()       # {int}
requested_kde_percents = set()   # {int}

cached_df = None                 # pandas.DataFrame or None
cached_headers = []              # list[str]

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

# ---- Lifecycle helpers ----
def clear_all_results():
    """
    Reset analysis outputs and the outputs/ folder WITHOUT rebinding globals.
    This preserves references imported elsewhere (e.g., in app.py).
    """
    # mutate, don't rebind
    mcp_results.clear()
    kde_results.clear()
    requested_percents.clear()
    requested_kde_percents.clear()

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

def save_all_mcps_zip():
    """
    Writes:
      - outputs/mcps_all.geojson       (MCP polygons, if any)
      - outputs/home_range_areas.csv   (areas for MCP & KDE)
      - outputs/spatchat_results.zip   (zip of everything in outputs/)
    Returns path to the zip.
    """
    os.makedirs("outputs", exist_ok=True)
    features = []
    rows = []

    # MCP features
    if any(mcp_results.values()):
        for animal, percents in mcp_results.items():
            for percent, v in percents.items():
                features.append({
                    "type": "Feature",
                    "properties": {"animal_id": animal, "percent": percent, "area_km2": v["area"]},
                    "geometry": mapping(v["polygon"])
                })
                rows.append((animal, f"MCP-{percent}", v["area"]))

        geojson = {"type": "FeatureCollection", "features": features}
        with open(os.path.join("outputs", "mcps_all.geojson"), "w") as f:
            json.dump(geojson, f)

    # KDE areas
    for animal, percents in kde_results.items():
        for percent, v in percents.items():
            rows.append((animal, f"KDE-{percent}", v["area"]))

    # Areas CSV
    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        df.to_csv(os.path.join("outputs", "home_range_areas.csv"), index=False)

    # Zip everything in outputs/
    archive = "outputs/spatchat_results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk("outputs"):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, "outputs")
                zipf.write(full_path, arcname=rel_path)

    print("ZIP written:", archive)
    return archive

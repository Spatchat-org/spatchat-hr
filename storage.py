import os, shutil, json, zipfile
import pandas as pd
from shapely.geometry import mapping

# Global analysis state (mirrors your original globals)
mcp_results = {}
kde_results = {}
requested_percents = set()
requested_kde_percents = set()
cached_df = None
cached_headers = []

def clear_all_results():
    global mcp_results, kde_results, requested_percents, requested_kde_percents
    mcp_results = {}
    kde_results = {}
    requested_percents = set()
    requested_kde_percents = set()
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

def save_all_mcps_zip():
    os.makedirs("outputs", exist_ok=True)
    from . import storage  # self-import for globals
    features, rows = [], []

    if any(storage.mcp_results.values()):
        for animal, percents in storage.mcp_results.items():
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

    for animal, percents in storage.kde_results.items():
        for percent, v in percents.items():
            rows.append((animal, f"KDE-{percent}", v["area"]))

    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        df.to_csv(os.path.join("outputs", "home_range_areas.csv"), index=False)

    archive = "outputs/spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk("outputs"):
            for file in files:
                if file.endswith(".zip"): continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, "outputs")
                zipf.write(full_path, arcname=rel_path)
    print("ZIP written:", archive)
    return archive

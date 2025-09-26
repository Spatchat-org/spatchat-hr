# storage.py
"""
In-memory app state for SpatChat-HR.
Simple module-level globals so it works on HF Spaces without a DB.
"""

import os
import shutil
import json
import zipfile
import pandas as pd
from shapely.geometry import mapping

# -------------------------
# Cached upload
# -------------------------
_cached_df = None
_cached_headers = []

def get_cached_df():
    return _cached_df

def set_cached_df(df):
    global _cached_df
    _cached_df = df

def get_cached_headers():
    return list(_cached_headers)

def set_cached_headers(headers):
    global _cached_headers
    _cached_headers = list(headers or [])

# -------------------------
# Analysis artifacts
# -------------------------
# MCP results: {animal_id: {percent: {"polygon": shapely.Polygon, "area": float_km2}}}
mcp_results = {}

# KDE results: {animal_id: {percent: {"contour": (Multi)Polygon, "area": float_km2,
#                                     "geotiff": path, "geojson": path}}}
kde_results = {}

# What user asked for (to control map layering/order)
requested_percents = set()
requested_kde_percents = set()

def clear_all_results():
    """Reset all computed artifacts and clean outputs/ directory."""
    global mcp_results, kde_results, requested_percents, requested_kde_percents
    mcp_results = {}
    kde_results = {}
    requested_percents = set()
    requested_kde_percents = set()
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

def save_all_mcps_zip():
    """Write combined GeoJSON/CSV + pack everything in outputs/ into a ZIP."""
    os.makedirs("outputs", exist_ok=True)

    features, rows = [], []

    # Aggregate MCPs
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

    # Aggregate KDEs
    for animal, percents in kde_results.items():
        for percent, v in percents.items():
            rows.append((animal, f"KDE-{percent}", v["area"]))

    # Areas CSV
    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        df.to_csv(os.path.join("outputs", "home_range_areas.csv"), index=False)

    # Zip all artifacts under outputs/
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

__all__ = [
    # cached upload
    "get_cached_df", "set_cached_df",
    "get_cached_headers", "set_cached_headers",
    # analysis state
    "mcp_results", "kde_results",
    "requested_percents", "requested_kde_percents",
    # ops
    "clear_all_results", "save_all_mcps_zip",
]

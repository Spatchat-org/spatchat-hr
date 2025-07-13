import os
import json
import gradio as gr
import pandas as pd
import folium
import shutil
import random
import numpy as np
from pyproj import Transformer
from together import Together
from dotenv import load_dotenv
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, mapping, MultiPolygon
import zipfile
import re
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.neighbors import KernelDensity
import tempfile

print("Starting SpatChat (multi-MCP/KDE, robust download version)")

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
    # Clean outputs folder and reset everything for new session
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

# ========== LLM SETUP (Together API) ==========
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP, KDE, dBBMM, AKDE, etc.), reply ONLY in JSON using this format:
{"tool": "home_range", "method": "mcp", "levels": [95, 50]}
- method: one of "mcp", "kde", "akde", "bbmm", "dbbmm"
- levels: list of percentages for the home range (default [95] if user doesn't specify)
- Optionally, include animal_id if the user specifies a particular animal.
For any other questions, answer as an expert movement ecologist in plain text (keep to 2-3 sentences).
"""
FALLBACK_PROMPT = """
You are SpatChat, a wildlife movement expert.
If you can't map a request to a home range tool, just answer naturally.
Keep replies under three sentences.
"""

def ask_llm(chat_history, user_input):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.0
    ).choices[0].message.content
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7
        ).choices[0].message.content
        return None, conv

def render_empty_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

def fit_map_to_bounds(m, df):
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    if np.isfinite([min_lat, max_lat, min_lon, max_lon]).all():
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m

def looks_like_latlon(df, x_col, y_col):
    try:
        x_vals = df[x_col].astype(float)
        y_vals = df[y_col].astype(float)
        if x_vals.between(-180, 180).all() and y_vals.between(-90, 90).all():
            return "lonlat"
        if x_vals.between(-90, 90).all() and y_vals.between(-180, 180).all():
            return "latlon"
    except Exception:
        return None
    return None

def looks_invalid_latlon(df, lat_col, lon_col):
    try:
        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)
        return not (lat.between(-90, 90).all() and lon.between(-180, 180).all())
    except Exception:
        return True

def handle_upload_initial(file):
    global cached_df, cached_headers, mcp_results, requested_percents
    mcp_results = {}
    requested_percents = set()
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)
    try:
        df = pd.read_csv(filename)
        cached_df = df
        cached_headers = list(df.columns)
    except Exception as e:
        return [], *(gr.update(visible=False) for _ in range(8)), render_empty_map(), gr.update(visible=False)
    lower_cols = [col.lower() for col in df.columns]
    if "latitude" in lower_cols and "longitude" in lower_cols:
        lat_col = df.columns[lower_cols.index("latitude")]
        lon_col = df.columns[lower_cols.index("longitude")]
        if looks_invalid_latlon(df, lat_col, lon_col):
            return [
                {"role": "assistant", "content": "Your columns are labeled `latitude` and `longitude`, but the values do not look like geographic coordinates. Please confirm your coordinate system below."}
            ], \
            gr.update(choices=cached_headers, value=lon_col, visible=True), \
            gr.update(choices=cached_headers, value=lat_col, visible=True), \
            gr.update(visible=True), \
            render_empty_map(), \
            *(gr.update(visible=True) for _ in range(4)), gr.update(visible=False)
        else:
            return [
                {"role": "assistant", "content": "CSV uploaded. Latitude and longitude detected. You may now proceed to create home ranges."}
            ], *(gr.update(visible=False) for _ in range(3)), handle_upload_confirm("longitude", "latitude", ""), *(gr.update(visible=False) for _ in range(4)), gr.update(visible=False)

    x_names = ["x", "easting", "lon", "longitude"]
    y_names = ["y", "northing", "lat", "latitude"]
    found_x = next((col for col in df.columns if col.lower() in x_names), df.columns[0])
    found_y = next((col for col in df.columns if col.lower() in y_names and col != found_x), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    if found_x == found_y and len(df.columns) > 1:
        found_y = df.columns[1 if df.columns[0] == found_x else 0]
    latlon_guess = looks_like_latlon(df, found_x, found_y)
    if latlon_guess:
        df["longitude"] = df[found_x] if latlon_guess == "lonlat" else df[found_y]
        df["latitude"] = df[found_y] if latlon_guess == "lonlat" else df[found_x]
        cached_df = df
        return [
            {"role": "assistant", "content": f"CSV uploaded. `{found_x}`/`{found_y}` interpreted as latitude/longitude."}
        ], *(gr.update(visible=False) for _ in range(3)), handle_upload_confirm("longitude", "latitude", ""), *(gr.update(visible=False) for _ in range(4)), gr.update(visible=False)

    return [
        {"role": "assistant", "content": f"CSV uploaded. Your coordinates do not appear to be latitude/longitude. Please specify X (easting), Y (northing), and the CRS/UTM zone below."}
    ], \
    gr.update(choices=cached_headers, value=found_x, visible=True), \
    gr.update(choices=cached_headers, value=found_y, visible=True), \
    gr.update(visible=True), render_empty_map(), *(gr.update(visible=True) for _ in range(4)), gr.update(visible=False)

def parse_crs_input(crs_input):
    crs_input = str(crs_input).strip().upper()
    if crs_input.startswith("EPSG:"):
        return int(crs_input.split(":")[1])
    if crs_input.isdigit():
        return 32600 + int(crs_input)
    if len(crs_input) >= 2 and crs_input[:-1].isdigit() and crs_input[-1] in ("N", "S"):
        zone = int(crs_input[:-1])
        return 32600 + zone if crs_input[-1] == "N" else 32700 + zone
    raise ValueError("Invalid CRS input. Use EPSG code or UTM zone like '33N'.")

def handle_upload_confirm(x_col, y_col, crs_input):
    global cached_df
    df = cached_df.copy()
    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Selected coordinate columns not found in data.</p>"
    if x_col.lower() in ["longitude", "lon"] and y_col.lower() in ["latitude", "lat"]:
        df['longitude'] = df[x_col]
        df['latitude'] = df[y_col]
    else:
        if not crs_input or crs_input.strip() == "":   
            return "<p>Please enter a CRS or UTM zone before confirming.</p>"
        try:
            epsg = parse_crs_input(crs_input)
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            df['longitude'], df['latitude'] = transformer.transform(
                df[x_col].values, df[y_col].values
            )
        except Exception as e:
            return f"<p>Failed to convert coordinates: {e}</p>"
    has_timestamp = "timestamp" in df.columns
    has_animal_id = "animal_id" in df.columns
    if has_timestamp:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not has_animal_id:
        df["animal_id"] = "sample"
    cached_df = df
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=9, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)
    points_layer = folium.FeatureGroup(name="Points", show=True)
    lines_layer = folium.FeatureGroup(name="Tracks", show=True)
    animal_ids = df["animal_id"].unique()
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}
    for animal in animal_ids:
        track = df[df["animal_id"] == animal]
        coords = list(zip(track["latitude"], track["longitude"]))
        color = color_map[animal]
        if has_timestamp:
            track = track.sort_values("timestamp")
            folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=animal).add_to(lines_layer)
        for _, row in track.iterrows():
            label = f"{animal}" + (f"<br>{row['timestamp']}" if has_timestamp else "")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                popup=label,
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(points_layer)
    points_layer.add_to(m)
    if has_timestamp:
        lines_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    return m._repr_html_()

def parse_levels_from_text(text):
    levels = [int(val) for val in re.findall(r'\b([1-9][0-9]?|100)\b', text)]
    if not levels:
        return [95]
    return sorted(set(levels))

# ========== MCP Functions ==========
def mcp_polygon(latitudes, longitudes, percent=95):
    points = np.column_stack((longitudes, latitudes))
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    n_keep = max(3, int(len(points) * (percent / 100.0)))
    keep_idx = np.argsort(dists)[:n_keep]
    points_kept = points[keep_idx]
    if len(points_kept) < 3:
        return None
    hull = ConvexHull(points_kept)
    hull_points = points_kept[hull.vertices]
    return hull_points

def add_mcps(df, percent_list):
    global mcp_results
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            if animal not in mcp_results:
                mcp_results[animal] = {}
            if percent not in mcp_results[animal]:
                track = df[df["animal_id"] == animal]
                hull_points = mcp_polygon(track['latitude'].values, track['longitude'].values, percent)
                if hull_points is not None:
                    poly = Polygon([(lon, lat) for lon, lat in hull_points])
                    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
                    coords_proj = [transformer.transform(lon, lat) for lon, lat in hull_points]
                    poly_proj = Polygon(coords_proj)
                    area_km2 = poly_proj.area / 1e6
                    mcp_results[animal][percent] = {"polygon": poly, "area": area_km2}

# ========== KDE Section ==========
def kde_home_range(latitudes, longitudes, percent=95, grid_size=200):
    lon0, lat0 = np.mean(longitudes), np.mean(latitudes)
    utm_zone = int((lon0 + 180) // 6) + 1
    epsg_utm = 32600 + utm_zone if lat0 >= 0 else 32700 + utm_zone
    to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg_utm}", always_xy=True)
    to_latlon = Transformer.from_crs(f"epsg:{epsg_utm}", "epsg:4326", always_xy=True)
    x, y = to_utm.transform(longitudes, latitudes)
    xy = np.vstack([x, y]).T

    n = xy.shape[0]
    if n > 1:
        stds = np.std(xy, axis=0, ddof=1)
        h = np.power(4 / (3 * n), 1 / 5) * np.mean(stds)
        if h < 1:
            h = 30.0
    else:
        h = 30.0

    margin = 3 * h
    xmin, xmax = x.min() - margin, x.max() + margin
    ymin, ymax = y.min() - margin, y.max() + margin
    x_grid = np.linspace(xmin, xmax, grid_size)
    y_grid = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    kde = KernelDensity(bandwidth=h, kernel="gaussian")
    kde.fit(xy)
    Z = np.exp(kde.score_samples(grid_points)).reshape(X.shape)

    cell_area = (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])
    Z /= (Z.sum() * cell_area)

    Z_flat = Z.flatten()
    idx_desc = np.argsort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_flat[idx_desc] * cell_area)
    threshold = Z_flat[idx_desc][np.searchsorted(cumsum, percent / 100.0)]
    mask = Z >= threshold

    Z_masked = np.where(mask, Z, 0)
    total_prob = Z_masked.sum() * cell_area
    if total_prob > 0:
        Z_masked /= total_prob

    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons = []
    for contour in contours:
        px, py = contour[:, 1], contour[:, 0]
        utm_xs = np.interp(px, np.arange(grid_size), x_grid)
        utm_y## truncated due to length ##

demo.launch(ssr_mode=False)

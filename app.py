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
from shapely.geometry import Polygon, mapping
import zipfile
import re
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage import measure
import tempfile

print("Starting SpatChat (multi-MCP/KDE, robust download version)")

# ====== GLOBAL STORAGE ======
mcp_results = {}    # animal_id -> {percent: {"polygon": Polygon, "area": area_km2}}
kde_results = {}    # animal_id -> {percent: {"contour": Polygon/MultiPolygon, "area": area_km2, "geotiff": path, "geojson": path}}
requested_percents = set()
requested_kde_percents = set()
cached_df = None
cached_headers = []

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

def parse_levels_from_text(text):
    # Extracts all percentages 1-100 in string
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
    # Estimate bandwidth automatically, grid centered on data
    xy = np.vstack([longitudes, latitudes])
    kde = gaussian_kde(xy)
    xmin, xmax = longitudes.min(), longitudes.max()
    ymin, ymax = latitudes.min(), latitudes.max()
    x_grid = np.linspace(xmin, xmax, grid_size)
    y_grid = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    # find the density threshold for the desired percent
    Z_flat = Z.flatten()
    idx = np.argsort(Z_flat)[::-1]  # sort descending
    cumsum = np.cumsum(Z_flat[idx])
    cumsum /= cumsum[-1]
    threshold = Z_flat[idx][np.searchsorted(cumsum, percent / 100.0)]
    # Get mask
    mask = Z >= threshold
    # Find contours (in y, x grid)
    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons = []
    for contour in contours:
        poly_xy = np.array([ [X[int(p[0]), int(p[1])], Y[int(p[0]), int(p[1])]] for p in contour ])
        if len(poly_xy) >= 3:
            polygons.append(Polygon(poly_xy))
    if not polygons:
        return None, None, None, None
    mpoly = unary_union(polygons)
    # Area in degrees: just for demonstration
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    coords_proj = [transformer.transform(*pt) for pt in np.vstack([p.exterior.coords for p in polygons if isinstance(p, Polygon)])]
    poly_proj = Polygon(coords_proj)
    area_km2 = poly_proj.area / 1e6
    # Export raster as geotiff
    tiff_fp = tempfile.mktemp(suffix=f"_kde_{percent}.tif", dir="outputs")
    with rasterio.open(
        tiff_fp, "w",
        driver="GTiff",
        height=Z.shape[0], width=Z.shape[1],
        count=1, dtype=Z.dtype,
        crs="EPSG:4326",
        transform=from_origin(xmin, ymax, (xmax-xmin)/grid_size, (ymax-ymin)/grid_size)
    ) as dst:
        dst.write(Z, 1)
    # Export contour as geojson
    geojson_fp = tempfile.mktemp(suffix=f"_kde_{percent}.geojson", dir="outputs")
    with open(geojson_fp, "w") as f:
        json.dump(mapping(mpoly), f)
    return mpoly, area_km2, tiff_fp, geojson_fp, Z, x_grid, y_grid

def add_kdes(df, percent_list):
    global kde_results
    os.makedirs("outputs", exist_ok=True)
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            if animal not in kde_results:
                kde_results[animal] = {}
            if percent not in kde_results[animal]:
                track = df[df["animal_id"] == animal]
                mpoly, area_km2, tiff_fp, geojson_fp, *_ = kde_home_range(
                    track['latitude'].values, track['longitude'].values, percent
                )
                if mpoly is not None:
                    kde_results[animal][percent] = {
                        "contour": mpoly, "area": area_km2,
                        "geotiff": tiff_fp, "geojson": geojson_fp
                    }

# ========== Main Handlers ==========
def handle_chat(chat_history, user_message):
    global cached_df, mcp_results, kde_results, requested_percents, requested_kde_percents
    chat_history = list(chat_history)
    tool, llm_output = ask_llm(chat_history, user_message)
    mcp_list, kde_list = [], []
    method = None
    if tool and tool.get("tool") == "home_range":
        method = tool.get("method")
        levels = tool.get("levels", [95])
        levels = [int(p) for p in levels if 1 <= int(p) <= 100]
        if method == "mcp":
            mcp_list = levels
        elif method == "kde":
            kde_list = levels
    # Manual override from text
    if "mcp" in user_message.lower():
        mcp_list = parse_levels_from_text(user_message)
    if "kde" in user_message.lower():
        kde_list = parse_levels_from_text(user_message)
    # Defaults if blank
    if not mcp_list and not kde_list:
        mcp_list = [95]
    # Prepare
    if cached_df is None or "latitude" not in cached_df or "longitude" not in cached_df:
        chat_history.append({"role": "assistant", "content": "CSV must be uploaded with 'latitude' and 'longitude' columns."})
        return chat_history, gr.update(), None
    # Add home ranges
    if mcp_list:
        add_mcps(cached_df, mcp_list)
        requested_percents.update(mcp_list)
    if kde_list:
        add_kdes(cached_df, kde_list)
        requested_kde_percents.update(kde_list)
    # Build map
    df = cached_df
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=9)
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
    paths_layer = folium.FeatureGroup(name="Tracks", show=True)
    animal_ids = df["animal_id"].unique()
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}
    # Add points/paths
    for animal in animal_ids:
        track = df[df["animal_id"] == animal]
        color = color_map[animal]
        if "timestamp" in track.columns:
            track = track.sort_values("timestamp")
            coords = list(zip(track["latitude"], track["longitude"]))
            if len(coords) > 1:
                folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=f"{animal} Track").add_to(paths_layer)
        for idx, row in track.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{animal}"
            ).add_to(points_layer)
    points_layer.add_to(m)
    paths_layer.add_to(m)
    # Add MCPs
    for percent in requested_percents:
        for animal in animal_ids:
            if animal in mcp_results and percent in mcp_results[animal]:
                v = mcp_results[animal][percent]
                folium.Polygon(
                    locations=[(lat, lon) for lon, lat in np.array(v["polygon"].exterior.coords)],
                    color=color_map[animal],
                    fill=True,
                    fill_opacity=0.15 + 0.15 * (percent / 100),
                    popup=f"{animal} MCP {percent}%"
                ).add_to(m)
    # === KDE RASTER OVERLAYS & CONTOUR ===
    for percent in requested_kde_percents:
        for animal in animal_ids:
            if animal in kde_results and percent in kde_results[animal]:
                v = kde_results[animal][percent]
                # RASTER overlay as transparent image
                with rasterio.open(v["geotiff"]) as src:
                    arr = src.read(1)
                    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
                    # Map norm to RGBA (use matplotlib colormap)
                    cmap = plt.get_cmap('plasma')
                    rgba = (cmap(arr_norm) * 255).astype(np.uint8)
                    bounds = src.bounds
                    img = np.dstack([rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], (rgba[:, :, 3]*0.3).astype(np.uint8)])
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                        opacity=0.3,
                        name=f"{animal} KDE {percent}%",
                        interactive=False
                    ).add_to(m)
                # Contour as Polygon
                contour = v["contour"]
                if contour:
                    if isinstance(contour, MultiPolygon):
                        for poly in contour.geoms:
                            folium.Polygon(
                                locations=[(lat, lon) for lon, lat in poly.exterior.coords],
                                color=color_map[animal],
                                fill=True,
                                fill_opacity=0.2,
                                popup=f"{animal} KDE {percent}%"
                            ).add_to(m)
                    elif isinstance(contour, Polygon):
                        folium.Polygon(
                            locations=[(lat, lon) for lon, lat in contour.exterior.coords],
                            color=color_map[animal],
                            fill=True,
                            fill_opacity=0.2,
                            popup=f"{animal} KDE {percent}%"
                        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    map_html = m._repr_html_()
    # Summary message
    msg = []
    if mcp_list:
        msg.append(f"MCP home ranges ({', '.join(str(p) for p in mcp_list)}%) calculated.")
    if kde_list:
        msg.append(f"KDE home ranges ({', '.join(str(p) for p in kde_list)}%) calculated (raster & contours).")
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": " ".join(msg) + " Download all results below."})
    return chat_history, gr.update(value=map_html), "spatchat_results.zip"

# ========== ZIP Results ==========
def save_all_mcps_zip():
    os.makedirs("outputs", exist_ok=True)
    # Write MCPs as GeoJSON
    features = []
    for animal, percents in mcp_results.items():
        for percent, v in percents.items():
            features.append({
                "type": "Feature",
                "properties": {
                    "animal_id": animal,
                    "percent": percent,
                    "area_km2": v["area"]
                },
                "geometry": mapping(v["polygon"])
            })
    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path = os.path.join("outputs", "mcps_all.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)
    # Areas as CSV
    rows = []
    for animal, percents in mcp_results.items():
        for percent, v in percents.items():
            rows.append((animal, f"MCP-{percent}", v["area"]))
    for animal, percents in kde_results.items():
        for percent, v in percents.items():
            rows.append((animal, f"KDE-{percent}", v["area"]))
    df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
    csv_path = os.path.join("outputs", "home_range_areas.csv")
    df.to_csv(csv_path, index=False)
    archive = "spatchat_results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(geojson_path, arcname="mcps_all.geojson")
        zipf.write(csv_path, arcname="home_range_areas.csv")
        # Add all KDE outputs (GeoTIFFs + GeoJSONs)
        for animal, percents in kde_results.items():
            for percent, v in percents.items():
                if v.get("geotiff"):
                    zipf.write(v["geotiff"], arcname=f"kde_{animal}_{percent}.tif")
                if v.get("geojson"):
                    zipf.write(v["geojson"], arcname=f"kde_{animal}_{percent}.geojson")
    print("ZIP written:", archive)
    return archive

# ========== UI ==========
with gr.Blocks(title="SpatChat: Home Range Analysis") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🏠 SpatChat: Home Range Analysis {hr}  🦊🦉🐢")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=hr" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=hr" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=hr" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Home Range Analysis.</i>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role": "assistant", "content": "Welcome to SpatChat! Please upload a CSV containing coordinates (lat/lon or UTM) and optional timestamp/animal_id to begin."}]
            )
            user_input = gr.Textbox(
                label="Ask Spatchat",
                placeholder="Type commands...",
                lines=1
            )
            file_input = gr.File(
                label="Upload Movement CSV (.csv or .txt only)",
                file_types=[".csv", ".txt"]
            )
            x_col = gr.Dropdown(label="X column", choices=[], visible=False)
            y_col = gr.Dropdown(label="Y column", choices=[], visible=False)
            crs_input = gr.Text(label="CRS (e.g. '32633', '33N', or 'EPSG:32633')", visible=False)
            confirm_btn = gr.Button("Confirm Coordinate Settings", visible=False)
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False)
            download_btn = gr.DownloadButton(
                "📥 Download Results",
                save_all_mcps_zip,
                label="Download Results"
            )

    file_input.change(
        fn=lambda file: (
            [{"role": "assistant", "content": "File uploaded!"}], 
            gr.update(choices=[], visible=False),
            gr.update(choices=[], visible=False),
            gr.update(visible=False),
            render_empty_map(),
            gr.update(choices=[], visible=False),
            gr.update(choices=[], visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        ),
        inputs=file_input,
        outputs=[
            chatbot, x_col, y_col, crs_input, map_output,
            x_col, y_col, crs_input, confirm_btn
        ]
    )
    confirm_btn.click(
        fn=lambda x_col, y_col, crs_input: render_empty_map(),
        inputs=[x_col, y_col, crs_input],
        outputs=map_output
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output, download_btn]
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

demo.launch(ssr_mode=False)

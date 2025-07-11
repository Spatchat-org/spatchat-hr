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

print("Starting SpatChat (Python-only MCP)")

# ====== GLOBAL MCP STORAGE ======
mcp_results = {}  # animal_id -> {"polygon": Polygon, "area": area_km2}
mcp_percent = 95  # Store last-used percent for reporting and export
zip_results = "outputs/spatchat_results.zip"  # Always points to latest ZIP

# ========== LLM SETUP (Together API) ==========
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP, KDE, dBBMM, AKDE, etc.), reply ONLY in JSON using this format:
{"tool": "home_range", "method": "mcp", "level": 95}
- method: one of "mcp", "kde", "akde", "bbmm", "dbbmm"
- level: percentage for the home range (default 95 if user doesn't specify)
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
        # fallback: try conversational response
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7
        ).choices[0].message.content
        return None, conv

# ========== App Logic ==========
cached_df = None
cached_headers = []

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
        # Check for lon/lat orientation
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

def guess_utm_crs(df, x_col, y_col, sample_size=10):
    """
    Guess the most likely UTM zone and hemisphere for a dataframe
    with projected coordinates (e.g., easting/northing), and return the
    best zone and transformed sample lat/lon for confidence.
    """
    from pyproj import CRS

    x = df[x_col].astype(float).values
    y = df[y_col].astype(float).values

    # Use a sample for performance
    idxs = np.linspace(0, len(x) - 1, min(sample_size, len(x))).astype(int)
    x_sample = x[idxs]
    y_sample = y[idxs]

    # Smart hemisphere guess: UTM northings in S hemisphere start at 10,000,000
    if np.median(y) > 9_000_000:
        hemisphere_candidates = ['S']
    else:
        hemisphere_candidates = ['N']

    best_zone = None
    best_score = -np.inf
    best_epsg = None
    best_latlon = None

    # Optionally, restrict possible zones to plausible range for your area (e.g., Europe 28-38)
    for hemisphere in hemisphere_candidates:
        for zone in range(1, 61):
            epsg = 32600 + zone if hemisphere == 'N' else 32700 + zone
            try:
                transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
                lons, lats = transformer.transform(x_sample, y_sample)
                lons = np.array(lons)
                lats = np.array(lats)
                if not (np.isfinite(lons).all() and np.isfinite(lats).all()):
                    continue
                # Check plausible lat/lon ranges
                if not (np.all((lats >= -90) & (lats <= 90)) and np.all((lons >= -180) & (lons <= 180))):
                    continue
                # Score: smaller longitude error to zone center is better
                central_meridian = -183 + zone * 6
                lon_error = np.abs(np.mean(lons) - central_meridian)
                score = -lon_error
                if score > best_score:
                    best_score = score
                    best_zone = zone
                    best_epsg = epsg
                    best_latlon = (lats, lons)
            except Exception:
                continue

    if best_zone and best_latlon is not None:
        hemisphere = 'N' if best_epsg < 32700 else 'S'
        return {
            "zone": best_zone,
            "hemisphere": hemisphere,
            "epsg": best_epsg,
            "crs_label": f"{best_zone}{hemisphere}",
            "lats": best_latlon[0],
            "lons": best_latlon[1]
        }
    return None

def handle_upload_initial(file):
    global cached_df, cached_headers
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)
    try:
        df = pd.read_csv(filename)
        cached_df = df
        cached_headers = list(df.columns)
    except Exception as e:
        return [], *(gr.update(visible=False) for _ in range(8)), render_empty_map()
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
            *(gr.update(visible=True) for _ in range(4))
        else:
            return [
                {"role": "assistant", "content": "CSV uploaded. Latitude and longitude detected. You may now proceed to create home ranges."}
            ], *(gr.update(visible=False) for _ in range(3)), handle_upload_confirm("longitude", "latitude", ""), *(gr.update(visible=False) for _ in range(4))

    # Try common fallback columns
    fallback_cols = ["x", "y", "lon", "lat", "easting", "northing"]
    x_names = ["x", "easting", "lon", "longitude"]
    y_names = ["y", "northing", "lat", "latitude"]
    found_x = next((col for col in df.columns if col.lower() in x_names), df.columns[0])
    found_y = next((col for col in df.columns if col.lower() in y_names and col != found_x), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    latlon_guess = looks_like_latlon(df, found_x, found_y)
    if latlon_guess:
        df["longitude"] = df[found_x] if latlon_guess == "lonlat" else df[found_y]
        df["latitude"] = df[found_y] if latlon_guess == "lonlat" else df[found_x]
        cached_df = df
        return [
            {"role": "assistant", "content": f"CSV uploaded. `{found_x}`/`{found_y}` interpreted as latitude/longitude."}
        ], *(gr.update(visible=False) for _ in range(3)), handle_upload_confirm("longitude", "latitude", ""), *(gr.update(visible=False) for _ in range(4))

    guess = guess_utm_crs(df, found_x, found_y)
    if guess:
        transformer = Transformer.from_crs(guess["epsg"], 4326, always_xy=True)
        df["longitude"], df["latitude"] = transformer.transform(df[found_x].values, df[found_y].values)
        cached_df = df
        return [
            {"role": "assistant", "content": f"CSV uploaded. `{found_x}`/`{found_y}` appear to be UTM. Guessed zone **{guess['crs_label']}**. Please confirm or adjust below."}
        ], \
        gr.update(choices=cached_headers, value=found_x, visible=True), \
        gr.update(choices=cached_headers, value=found_y, visible=True), \
        gr.update(visible=True), handle_upload_confirm("longitude", "latitude", ""), \
        gr.update(value=guess["crs_label"], visible=True), \
        *(gr.update(visible=True) for _ in range(3))

    return [
        {"role": "assistant", "content": "CSV uploaded. Coordinate columns not clearly labeled. Please confirm and provide a CRS."}
    ], \
    gr.update(choices=cached_headers, value=found_x, visible=True), \
    gr.update(choices=cached_headers, value=found_y, visible=True), \
    gr.update(visible=True), render_empty_map(), *(gr.update(visible=True) for _ in range(4))

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
        try:
            epsg = parse_crs_input(crs_input)
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            df['longitude'], df['latitude'] = transformer.transform(df[x_col].values, df[y_col].values)
        except Exception as e:
            return f"<p>Failed to convert coordinates: {e}</p>"
    has_timestamp = "timestamp" in df.columns
    has_animal_id = "animal_id" in df.columns
    if has_timestamp:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not has_animal_id:
        df["animal_id"] = "sample"
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

def save_results_zip():
    # Save as GeoJSON
    features = []
    for animal_id, v in mcp_results.items():
        features.append({
            "type": "Feature",
            "properties": {"animal_id": animal_id, "area_km2": v["area"]},
            "geometry": mapping(v["polygon"])
        })
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    os.makedirs("outputs", exist_ok=True)
    geojson_path = os.path.join("outputs", "mcps.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)
    # CSV
    csv_path = os.path.join("outputs", "mcp_areas.csv")
    df = pd.DataFrame(
        [(aid, v["area"]) for aid, v in mcp_results.items()],
        columns=["animal_id", "area_km2"]
    )
    df.to_csv(csv_path, index=False)
    # ZIP
    zip_path = os.path.join("outputs", "spatchat_results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(geojson_path, arcname="mcps.geojson")
        zipf.write(csv_path, arcname="mcp_areas.csv")
    return zip_path

def handle_chat(chat_history, user_message):
    global cached_df, mcp_results, mcp_percent, zip_results
    # Respond to "area" or "mcp area"
    if "area" in user_message.lower():
        if not mcp_results:
            response = "No MCPs have been calculated yet."
        else:
            header = "| Animal ID | MCP Area (km²) |\n|---|---|"
            rows = "\n".join(f"| {aid} | {v['area']:.2f} |" for aid, v in mcp_results.items())
            response = f"### MCP Areas ({mcp_percent}% MCP)\n{header}\n{rows}"
        chat_history = chat_history + [{"role": "user", "content": user_message}]
        chat_history = chat_history + [{"role": "assistant", "content": response}]
        return chat_history, gr.update(), ""  # Last output resets user_input

    tool, llm_output = ask_llm(chat_history, user_message)
    if tool and tool.get("tool") == "home_range" and tool.get("method") == "mcp":
        percent = tool.get("level", 95)
        mcp_percent = percent
        df = cached_df
        if df is None or "latitude" not in df or "longitude" not in df:
            chat_history = chat_history + [{"role": "user", "content": user_message}]
            chat_history = chat_history + [{"role": "assistant", "content": "CSV must be uploaded with 'latitude' and 'longitude' columns."}]
            return chat_history, gr.update(), ""
        if "animal_id" not in df.columns:
            df["animal_id"] = "sample"
        mcp_results.clear()
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=9)
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
        mcps_layer = folium.FeatureGroup(name="MCP Polygons", show=True)
        paths_layer = folium.FeatureGroup(name="Tracks", show=True)
        animal_ids = df["animal_id"].unique()
        color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        for animal in animal_ids:
            track = df[df["animal_id"] == animal]
            color = color_map[animal]
            # Paths
            if len(track) > 1:
                coords = list(zip(track["latitude"], track["longitude"]))
                folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=f"{animal} Track").add_to(paths_layer)
            # Points
            for idx, row in track.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{animal}"
                ).add_to(points_layer)
            # MCP for each animal
            hull_points = mcp_polygon(track['latitude'].values, track['longitude'].values, percent)
            if hull_points is not None:
                coords_lonlat = [(lon, lat) for lon, lat in hull_points]
                coords_proj = [transformer.transform(lon, lat) for lon, lat in coords_lonlat]
                poly = Polygon(coords_proj)
                area_km2 = poly.area / 1e6
                mcp_results[animal] = {"polygon": Polygon(coords_lonlat), "area": area_km2}
                folium.Polygon(
                    locations=[(lat, lon) for lon, lat in hull_points],
                    color=color,
                    fill=True,
                    fill_opacity=0.2,
                    popup=f"{animal} MCP {percent}%"
                ).add_to(mcps_layer)
        points_layer.add_to(m)
        mcps_layer.add_to(m)
        paths_layer.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m = fit_map_to_bounds(m, df)
        map_html = m._repr_html_()
        # After calculation, save ZIP and update zip_results path
        global zip_results
        zip_results = save_results_zip()
        chat_history = chat_history + [{"role": "user", "content": user_message}]
        chat_history = chat_history + [{"role": "assistant", "content": f"MCP {percent}% home ranges calculated for each animal and displayed on the map. Click 'Download Results' below the map to export GeoJSON + CSV."}]
        return chat_history, gr.update(value=map_html), ""
    else:
        chat_history = chat_history + [{"role": "user", "content": user_message}]
        chat_history = chat_history + [{"role": "assistant", "content": llm_output.strip()}]
        return chat_history, gr.update(), ""

# --- Only returns the *latest* zip_results path for DownloadButton
def _get_zip_results():
    return zip_results if os.path.exists(zip_results) else None

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
        margin: 10px 50px 10px 10px;  /* top, right, bottom, left */
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
            # Only need this: no .click(), no outputs=download_btn
            download_btn = gr.DownloadButton("📥 Download Results", _get_zip_results)

    file_input.change(
        fn=handle_upload_initial,
        inputs=file_input,
        outputs=[
            chatbot, x_col, y_col, crs_input, map_output,
            x_col, y_col, crs_input, confirm_btn
        ]
    )
    confirm_btn.click(
        fn=handle_upload_confirm,
        inputs=[x_col, y_col, crs_input],
        outputs=map_output
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output, user_input]
    )

demo.launch()
